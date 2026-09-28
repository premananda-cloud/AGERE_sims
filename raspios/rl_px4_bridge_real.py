"""
rl_px4_bridge_real.py

REAL FLIGHT. Runs the numpy hover policy against a real flight controller
(default: serial connection on a Raspberry Pi). This is NOT a drop-in
swap for rl_px4_bridge_sim.py -- it adds guardrails a policy trained only
in HoverAviary has no way to know it needs:

  * Requires typed confirmation before arming (no --yes bypass, on purpose)
  * --dry-run: connect + compute actions, never arm/takeoff -- use this
    first, every time, for a new .npz or after any code change
  * Geofence: aborts to land if horizontal drift from the hover target
    exceeds --geofence-radius
  * Altitude ceiling: aborts to land if altitude exceeds --max-altitude
  * Telemetry watchdog: aborts to land if telemetry goes stale for
    longer than --telemetry-timeout (link glitch, USB hiccup, etc.)
  * --speed-scale (default 0.5): damps commanded speed on top of the
    policy's trained MAX_SPEED_MPS, without touching the policy's
    internal calibration. Raise this only after real flights at 0.5
    look boring and safe.

None of this replaces a human safety pilot. Keep a hand on an RC
transmitter in manual/position mode, ready to override, for every flight.
This is inference from a policy that has only ever seen a physics
simulator -- treat the first several flights as "will it try to do
something weird" tests, not as a working product demo.

Usage:
    # ALWAYS do this first, for any new model or code change:
    python rl_px4_bridge_real.py --model hover_policy.npz --dry-run

    # First real flight -- short, damped, tight geofence:
    python rl_px4_bridge_real.py --model hover_policy.npz \\
        --duration 10 --speed-scale 0.3 --geofence-radius 2 --max-altitude 1.5 \\
        --log-dir ../logs/real_run_1
"""

import argparse
import asyncio
import time

import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw

from rl_bridge_common import (
    CONTROL_RATE_HZ,
    NumpyPolicy,
    StepLogger,
    TelemetryCache,
    action_to_ned_velocity,
    build_observation,
    ned_position_to_train_frame,
)

DEFAULT_CONNECTION = "serial:///dev/ttyACM0:115200"
DEFAULT_ALTITUDE_M = 1.0
DEFAULT_DURATION_S = 15.0          # short by default for a first real flight
DEFAULT_SPEED_SCALE = 0.5          # damped by default; see module docstring
DEFAULT_GEOFENCE_RADIUS_M = 3.0
DEFAULT_MAX_ALTITUDE_M = 2.0
DEFAULT_TELEMETRY_TIMEOUT_S = 0.5


class SafetyAbort(Exception):
    """Raised to break out of the control loop into a landing sequence."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def confirm_arming(args) -> None:
    print()
    print("=" * 60)
    print("REAL FLIGHT -- PRE-ARM CHECKLIST")
    print("=" * 60)
    print(f"  Connection:       {args.connection}")
    print(f"  Model:            {args.model}")
    print(f"  Target altitude:  {args.altitude} m")
    print(f"  Duration:         {args.duration} s")
    print(f"  Speed scale:      {args.speed_scale} (of policy's {0.25} m/s max)")
    print(f"  Geofence radius:  {args.geofence_radius} m")
    print(f"  Max altitude:     {args.max_altitude} m")
    print(f"  Telemetry timeout:{args.telemetry_timeout} s")
    print("-" * 60)
    print("  [ ] Props clear of obstructions and people")
    print("  [ ] Flying in a space appropriately sized for the geofence above")
    print("  [ ] Safety pilot has RC transmitter in hand, ready to override")
    print("  [ ] You have visually confirmed --dry-run output looked sane")
    print("  [ ] You are complying with local drone regulations for this flight")
    print("=" * 60)
    typed = input("Type FLY to arm and take off, anything else to abort: ")
    if typed.strip() != "FLY":
        raise SystemExit("Arming not confirmed. Aborting before arm.")


def check_safety_envelope(cache: TelemetryCache, target_xyz: np.ndarray,
                           geofence_radius: float, max_altitude: float,
                           telemetry_timeout: float) -> None:
    if cache.age_s() > telemetry_timeout:
        raise SafetyAbort(f"telemetry stale for {cache.age_s():.2f}s "
                           f"(limit {telemetry_timeout}s)")

    pos = ned_position_to_train_frame(*cache.position_ned)
    horiz_drift = float(np.linalg.norm(pos[:2] - target_xyz[:2]))
    if horiz_drift > geofence_radius:
        raise SafetyAbort(f"geofence breached: {horiz_drift:.2f}m from target "
                           f"(limit {geofence_radius}m)")

    altitude = float(pos[2])
    if altitude > max_altitude:
        raise SafetyAbort(f"altitude ceiling breached: {altitude:.2f}m "
                           f"(limit {max_altitude}m)")


async def dry_run(args, policy: NumpyPolicy):
    print("=== DRY RUN (no arm, no takeoff) ===")
    drone = System()
    await drone.connect(system_address=args.connection)

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected")
            break

    cache = TelemetryCache()
    asyncio.ensure_future(cache.track_position_velocity(drone))
    asyncio.ensure_future(cache.track_attitude(drone))
    print("Waiting for first telemetry samples...")
    await cache.ready.wait()

    target_xyz = ned_position_to_train_frame(*cache.position_ned)
    print(f"Using current position as fake hover target: {target_xyz}")
    print(f"Computing {int(CONTROL_RATE_HZ * 3)} steps of policy output "
          f"(no commands sent to the vehicle)...\n")

    dt = 1.0 / CONTROL_RATE_HZ
    for step in range(int(CONTROL_RATE_HZ * 3)):
        obs = build_observation(cache, target_xyz)
        raw_action = policy.predict(obs)
        current_yaw_deg = np.degrees(cache.attitude_rpy_rad[2])
        setpoint = action_to_ned_velocity(raw_action, current_yaw_deg,
                                           speed_scale=args.speed_scale)
        if step % int(CONTROL_RATE_HZ) == 0:
            print(f"  step {step:4d}  raw_action={np.round(raw_action, 3)}  "
                  f"cmd_vel_ned=({setpoint.north_m_s:.3f}, "
                  f"{setpoint.east_m_s:.3f}, {setpoint.down_m_s:.3f})")
        await asyncio.sleep(dt)

    print("\nDry run complete. Sanity-check the raw_action / cmd_vel values "
          "above before ever running a real flight with this model.")


async def real_flight(args, policy: NumpyPolicy):
    confirm_arming(args)

    logger = StepLogger(args.log_dir)

    drone = System()
    await drone.connect(system_address=args.connection)

    print("Waiting for drone to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("Connected")
            break

    print("Waiting for GPS/home position...")
    async for health in drone.telemetry.health():
        if health.is_global_position_ok and health.is_home_position_ok:
            print("GPS OK")
            break

    cache = TelemetryCache()
    asyncio.ensure_future(cache.track_position_velocity(drone))
    asyncio.ensure_future(cache.track_attitude(drone))
    print("Waiting for first telemetry samples...")
    await cache.ready.wait()

    print(f"Setting takeoff altitude to {args.altitude} m")
    await drone.action.set_takeoff_altitude(args.altitude)

    print("Arming...")
    await drone.action.arm()

    print("Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(10)

    target_xyz = ned_position_to_train_frame(*cache.position_ned)
    print(f"Hover target captured (training frame, x/y/z): {target_xyz}")

    offboard = drone.offboard
    print("Priming Offboard with a few zero-velocity setpoints...")
    zero_vel = VelocityNedYaw(0.0, 0.0, 0.0, np.degrees(cache.attitude_rpy_rad[2]))
    for _ in range(10):
        await offboard.set_velocity_ned(zero_vel)
        await asyncio.sleep(0.05)

    try:
        await offboard.start()
    except OffboardError as e:
        print(f"Offboard start failed: {e._result.result}")
        await drone.action.land()
        return

    print(f"Offboard active. Running policy for up to {args.duration}s "
          f"at {CONTROL_RATE_HZ} Hz (speed_scale={args.speed_scale})...")
    dt = 1.0 / CONTROL_RATE_HZ
    start = time.monotonic()
    step_count = 0
    abort_reason = None

    try:
        while time.monotonic() - start < args.duration:
            loop_start = time.monotonic()
            t_s = loop_start - start

            check_safety_envelope(cache, target_xyz, args.geofence_radius,
                                   args.max_altitude, args.telemetry_timeout)

            obs = build_observation(cache, target_xyz)
            raw_action = policy.predict(obs)

            current_yaw_deg = np.degrees(cache.attitude_rpy_rad[2])
            setpoint = action_to_ned_velocity(raw_action, current_yaw_deg,
                                               speed_scale=args.speed_scale)
            await offboard.set_velocity_ned(setpoint)

            step_count += 1
            pos = ned_position_to_train_frame(*cache.position_ned)
            pos_error = float(np.linalg.norm(target_xyz - pos))

            logger.log_step(step=step_count, t_s=t_s, cache=cache,
                             pos_error=pos_error, raw_action=raw_action,
                             setpoint=setpoint)

            if step_count % int(CONTROL_RATE_HZ) == 0:
                print(f"  t={t_s:5.1f}s  pos_error={pos_error:.3f} m")

            elapsed = time.monotonic() - loop_start
            await asyncio.sleep(max(0.0, dt - elapsed))

    except SafetyAbort as e:
        abort_reason = e.reason
        print(f"\n!!! SAFETY ABORT: {abort_reason} -- landing now !!!\n")
    except Exception as e:
        abort_reason = f"unexpected error: {e}"
        print(f"Control loop error: {e}")

    finally:
        print("Stopping offboard, landing...")
        try:
            await offboard.stop()
        except OffboardError:
            pass
        await drone.action.land()

        async for in_air in drone.telemetry.in_air():
            if not in_air:
                print("Landed")
                break

        logger.close(
            loader="numpy-real",
            model_path=args.model,
            target_xyz=target_xyz,
            extra_meta={
                "episode_duration_s": args.duration,
                "target_altitude_m": args.altitude,
                "speed_scale": args.speed_scale,
                "geofence_radius_m": args.geofence_radius,
                "max_altitude_m": args.max_altitude,
                "abort_reason": abort_reason,
            },
        )


async def run(args):
    print(f"Loading policy weights from {args.model} ...")
    policy = NumpyPolicy(args.model)

    if args.dry_run:
        await dry_run(args, policy)
    else:
        await real_flight(args, policy)


def main():
    parser = argparse.ArgumentParser(
        description="Run hover policy on a real flight controller. Read the "
                     "module docstring before your first real flight.")
    parser.add_argument("--model", required=True, help="Path to hover_policy.npz")
    parser.add_argument("--connection", default=DEFAULT_CONNECTION,
                         help=f"MAVSDK connection string (default: {DEFAULT_CONNECTION})")
    parser.add_argument("--altitude", type=float, default=DEFAULT_ALTITUDE_M)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--speed-scale", type=float, default=DEFAULT_SPEED_SCALE,
                         help="Multiplier on top of the policy's trained speed limit (0-1 typical).")
    parser.add_argument("--geofence-radius", type=float, default=DEFAULT_GEOFENCE_RADIUS_M,
                         help="Abort+land if horizontal drift from hover target exceeds this (m).")
    parser.add_argument("--max-altitude", type=float, default=DEFAULT_MAX_ALTITUDE_M,
                         help="Abort+land if altitude exceeds this (m).")
    parser.add_argument("--telemetry-timeout", type=float, default=DEFAULT_TELEMETRY_TIMEOUT_S,
                         help="Abort+land if telemetry goes stale for longer than this (s).")
    parser.add_argument("--log-dir", default=None, help="If set, write steps.csv + summary.json here")
    parser.add_argument("--dry-run", action="store_true",
                         help="Connect and compute policy outputs WITHOUT arming or taking off.")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
