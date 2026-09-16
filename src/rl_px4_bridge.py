"""
rl_px4_bridge.py

Bridges the trained hover PPO policy (hover_champion.zip, trained in
AGERE against gym-pybullet-drones) to a running PX4 SITL + Gazebo
instance, over MAVSDK.

This is a FIRST-PASS integration script meant to close the loop once, not
a polished flight-stack component. Three things below are marked VERIFY —
each is a real unknown carried over from what wasn't visible in the
training code, not a stylistic choice. Get these wrong and the drone
won't crash the script, it'll just fly subtly wrong (too fast/slow, or
react to attitude backwards) -- see the accompanying chat message for why
each one is currently a best-guess.

Usage:
    python rl_px4_bridge.py --model /path/to/hover_champion.zip

Assumes PX4 SITL + Gazebo is already running and reachable at
udpin://0.0.0.0:14540 (same as your working test_flight.py).
"""

import argparse
import asyncio
import time

import numpy as np
from mavsdk import System
from mavsdk.offboard import Offboard, OffboardError, VelocityNedYaw
from stable_baselines3 import PPO


# ---------------------------------------------------------------------------
# Config -- values that came straight from your code/docs
# ---------------------------------------------------------------------------

SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"
TARGET_ALTITUDE_M = 1.0          # matches HoverTaskConfig.target_position z (0,0,1.0)
EPISODE_DURATION_S = 30.0        # how long to run RL control before landing
CONTROL_RATE_HZ = 30.0           # confirmed: HoverAviary.CTRL_FREQ == 30

# Confirmed via HoverAviary.SPEED_LIMIT (CF2X drone model, PYB physics):
#   SPEED_LIMIT: 0.25 m/s
#   CTRL_FREQ:   30 Hz
# (previously a 1.0 m/s / 20 Hz placeholder -- the 4x-too-high speed limit
# was almost certainly why the SITL flight bounced around 0.1-0.35m instead
# of converging like the sim eval's 0.015-0.025m: a policy trained to output
# gentle 0.25 m/s-max corrections, given 4x the authority, overshoots every
# correction it makes.)
MAX_SPEED_MPS = 0.25


# ---------------------------------------------------------------------------
# Frame conversion
#
# VERIFY #3 (partially): assumes the PyBullet world frame your model was
# trained in maps as x->North, y->East, z(up)->-Down (PX4's NED). This is
# the most common convention but was never explicitly pinned down in the
# docs you shared -- confirm by eye once in --gui/SITL side by side
# (command "forward" and check both sims agree on which way that pushes
# the drone) before trusting numbers from a longer flight.
# ---------------------------------------------------------------------------

def ned_position_to_train_frame(north, east, down):
    """PX4 local NED position (relative to home) -> training's (x, y, z)."""
    return np.array([north, east, -down], dtype=np.float32)


def ned_velocity_to_train_frame(vn, ve, vd):
    return np.array([vn, ve, -vd], dtype=np.float32)


def train_direction_to_ned(direction_xyz):
    """Inverse of the position mapping above, for sending velocity commands."""
    x, y, z = direction_xyz
    return x, y, -z  # north, east, down


# ---------------------------------------------------------------------------
# Telemetry cache -- MAVSDK streams are async generators; the RL loop needs
# a synchronous "give me the latest state" read, not a blocking wait on the
# next message (which may arrive at a different rate than CONTROL_RATE_HZ).
# ---------------------------------------------------------------------------

class TelemetryCache:
    def __init__(self):
        self.position_ned = None       # (north, east, down)
        self.velocity_ned = None       # (vn, ve, vd)
        self.attitude_rpy_rad = None   # (roll, pitch, yaw)
        self.ready = asyncio.Event()

    def _maybe_set_ready(self):
        if self.position_ned is not None and self.attitude_rpy_rad is not None:
            self.ready.set()

    async def track_position_velocity(self, drone):
        async for pv in drone.telemetry.position_velocity_ned():
            self.position_ned = (pv.position.north_m, pv.position.east_m, pv.position.down_m)
            self.velocity_ned = (pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s)
            self._maybe_set_ready()

    async def track_attitude(self, drone):
        async for att in drone.telemetry.attitude_euler():
            self.attitude_rpy_rad = (
                np.radians(att.roll_deg),
                np.radians(att.pitch_deg),
                np.radians(att.yaw_deg),
            )
            self._maybe_set_ready()


# ---------------------------------------------------------------------------
# Observation construction
#
# VERIFY #2: order/sign here is the most-likely reading of status.md's
# "pos_error, velocity, roll/pitch/yaw_error" (9-dim), i.e.:
#   pos_error = target_position - current_position
#   velocity  = current velocity, raw, world frame
#   rpy_error = target_rpy - current_rpy, target assumed (0, 0, 0)
#     (yaw target=0 even though the VEL action can't actuate yaw --
#     hover's PID just holds whatever yaw it has, so this term may not
#     matter much to the policy, but get the SIGN right regardless: some
#     wrappers do current - target instead of target - current)
# Confirm the exact formula against hover_gym_wrapper.py's obs code before
# trusting this for anything beyond a first smoke test.
# ---------------------------------------------------------------------------

def build_observation(cache: TelemetryCache, target_xyz: np.ndarray) -> np.ndarray:
    pos = ned_position_to_train_frame(*cache.position_ned)
    vel = ned_velocity_to_train_frame(*cache.velocity_ned)
    roll, pitch, yaw = cache.attitude_rpy_rad

    pos_error = target_xyz - pos
    rpy_error = np.array([0.0 - roll, 0.0 - pitch, 0.0 - yaw], dtype=np.float32)

    return np.concatenate([pos_error, vel, rpy_error]).astype(np.float32)


def action_to_ned_velocity(raw_action: np.ndarray, current_yaw_deg: float):
    """Mirrors src/actions/velocity_action.py's normalize_action(), then
    converts the resulting direction+speed into a MAVSDK VelocityNedYaw.
    """
    raw_action = np.asarray(raw_action, dtype=np.float32)
    direction = raw_action[0:3]
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-6 else np.zeros(3, dtype=np.float32)
    speed = float(np.clip(raw_action[3], 0.0, 1.0)) * MAX_SPEED_MPS

    vx, vy, vz = direction * speed
    vn, ve, vd = train_direction_to_ned((vx, vy, vz))

    # No yaw control in this action type -- hold current yaw rather than
    # letting PX4 pick its own, matching HoverAviary's PID-holds-yaw
    # behavior.
    return VelocityNedYaw(vn, ve, vd, current_yaw_deg)


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

async def run(model_path: str):
    print(f"Loading policy from {model_path} ...")
    model = PPO.load(model_path, device="cpu")

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

    # Give it time to reach altitude and settle before handing control to
    # the policy -- takeoff/land stay scripted, matching your project's
    # own "out of scope for the RL push" decision on those phases.
    await asyncio.sleep(10)

    # Capture the post-takeoff position as the hover target, rather than a
    # fixed (0,0,-alt) NED point -- matches how the model was trained
    # (small jitter around a target, not "travel to an absolute point").
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

            obs = build_observation(cache, target_xyz)
            raw_action, _ = model.predict(obs, deterministic=True)

            current_yaw_deg = np.degrees(cache.attitude_rpy_rad[2])
            setpoint = action_to_ned_velocity(raw_action, current_yaw_deg)
            await offboard.set_velocity_ned(setpoint)

            step_count += 1
            if step_count % int(CONTROL_RATE_HZ) == 0:
                pos = ned_position_to_train_frame(*cache.position_ned)
                err = np.linalg.norm(target_xyz - pos)
                print(f"  t={time.monotonic()-start:5.1f}s  pos_error={err:.3f} m")

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to hover_champion.zip")
    args = parser.parse_args()
    asyncio.run(run(args.model))


if __name__ == "__main__":
    main()
