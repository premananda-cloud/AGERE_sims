"""
rl_px4_bridge_sim.py

SIMULATION ONLY. Runs the numpy hover policy against PX4 SITL (jMAVSim /
Gazebo) over the default offboard-API UDP port. This is the "safe to run
whenever, breaks nothing" version -- no props, no geofence, no arming
confirmation.

Use this to validate a newly-exported .npz against sim BEFORE it ever
touches rl_px4_bridge_real.py.

Usage:
    python rl_px4_bridge_sim.py --model hover_policy.npz
    python rl_px4_bridge_sim.py --model hover_policy.npz --log-dir ../logs/run_1
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

SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"
TARGET_ALTITUDE_M = 1.0
EPISODE_DURATION_S = 30.0


async def run(npz_path: str, log_dir: str | None):
    print("=== SIMULATION RUN (PX4 SITL) ===")
    print(f"Loading policy weights from {npz_path} ...")
    policy = NumpyPolicy(npz_path)
    logger = StepLogger(log_dir)

    drone = System()
    await drone.connect(system_address=SYSTEM_ADDRESS)

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

    print(f"Setting takeoff altitude to {TARGET_ALTITUDE_M} m")
    await drone.action.set_takeoff_altitude(TARGET_ALTITUDE_M)

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

    print(f"Offboard active. Running policy for {EPISODE_DURATION_S}s at {CONTROL_RATE_HZ} Hz...")
    dt = 1.0 / CONTROL_RATE_HZ
    start = time.monotonic()
    step_count = 0

    try:
        while time.monotonic() - start < EPISODE_DURATION_S:
            loop_start = time.monotonic()
            t_s = loop_start - start

            obs = build_observation(cache, target_xyz)
            raw_action = policy.predict(obs)

            current_yaw_deg = np.degrees(cache.attitude_rpy_rad[2])
            setpoint = action_to_ned_velocity(raw_action, current_yaw_deg)
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

    except Exception as e:
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
            loader="numpy-sim",
            model_path=npz_path,
            target_xyz=target_xyz,
            extra_meta={"episode_duration_s": EPISODE_DURATION_S,
                        "target_altitude_m": TARGET_ALTITUDE_M,
                        "speed_scale": 1.0},
        )


def main():
    parser = argparse.ArgumentParser(description="Run hover policy against PX4 SITL.")
    parser.add_argument("--model", required=True, help="Path to hover_policy.npz")
    parser.add_argument("--log-dir", default=None, help="If set, write steps.csv + summary.json here")
    args = parser.parse_args()
    asyncio.run(run(args.model, args.log_dir))


if __name__ == "__main__":
    main()
