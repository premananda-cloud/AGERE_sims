"""
rl_px4_bridge_numpy.py

Same bridge as rl_px4_bridge.py, but runs inference with a plain numpy
forward pass against an exported .npz weight file instead of loading the
full stable-baselines3/torch checkpoint. No sb3/torch/gymnasium
dependency at all -- only mavsdk + numpy, meant for a Raspberry Pi (or
any target where installing the full training stack isn't practical).

Produce the .npz with weight_manager/export_npz.py:
    python export_npz.py --model hover_champion.zip --out hover_policy.npz

Everything about the flight logic (observation construction, frame
conversion, offboard control loop, speed/rate constants) is IDENTICAL to
rl_px4_bridge.py -- this file only swaps out how the action is computed.
If you tune MAX_SPEED_MPS/CONTROL_RATE_HZ/obs formula in one, mirror the
change in the other, or better: once this is confirmed working
end-to-end, treat this file as the canonical one and retire the sb3
version, since this is what you'll actually fly.

Usage:
    python rl_px4_bridge_numpy.py --model hover_policy.npz
    python rl_px4_bridge_numpy.py --model hover_policy.npz --log-dir ../logs/run_1

With --log-dir, writes two files per run for later analysis/paper use:
    <log-dir>/steps.csv    -- one row per control step (30 Hz)
    <log-dir>/summary.json -- run metadata + summary stats
"""

import argparse
import asyncio
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw


# ---------------------------------------------------------------------------
# Config -- confirmed against the real HoverAviary instance, see the
# previous script's history for how these were derived.
# ---------------------------------------------------------------------------

SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"
TARGET_ALTITUDE_M = 1.0
EPISODE_DURATION_S = 30.0
CONTROL_RATE_HZ = 30.0            # confirmed: HoverAviary.CTRL_FREQ == 30
MAX_SPEED_MPS = 0.25              # confirmed: HoverAviary.SPEED_LIMIT == 0.25


# ---------------------------------------------------------------------------
# Policy: manual forward pass through the exported weights.
#
# Architecture (confirmed from hover_policy.npz's shapes + extract_weight.py's
# state_dict keys): SB3 default MlpPolicy, net_arch=[64,64], Tanh activations,
# Gaussian action head -- we only need the mean (deterministic action), no
# log-std, since inference here is always deterministic=True equivalent.
#   obs(9,) -> Linear(W1,b1) -> tanh -> Linear(W2,b2) -> tanh
#            -> Linear(Wa,ba) -> raw_action(4,)
# ---------------------------------------------------------------------------

class NumpyPolicy:
    def __init__(self, npz_path: str):
        d = np.load(npz_path)
        self.W1, self.b1 = d["W1"], d["b1"]
        self.W2, self.b2 = d["W2"], d["b2"]
        self.Wa, self.ba = d["Wa"], d["ba"]

    def predict(self, obs: np.ndarray) -> np.ndarray:
        h1 = np.tanh(self.W1 @ obs + self.b1)
        h2 = np.tanh(self.W2 @ h1 + self.b2)
        action = self.Wa @ h2 + self.ba
        return action.astype(np.float32)


# ---------------------------------------------------------------------------
# Frame conversion (unchanged from rl_px4_bridge.py)
# ---------------------------------------------------------------------------

def ned_position_to_train_frame(north, east, down):
    return np.array([north, east, -down], dtype=np.float32)


def ned_velocity_to_train_frame(vn, ve, vd):
    return np.array([vn, ve, -vd], dtype=np.float32)


def train_direction_to_ned(direction_xyz):
    x, y, z = direction_xyz
    return x, y, -z


# ---------------------------------------------------------------------------
# Telemetry cache (unchanged)
# ---------------------------------------------------------------------------

class TelemetryCache:
    def __init__(self):
        self.position_ned = None
        self.velocity_ned = None
        self.attitude_rpy_rad = None
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
# Observation construction (unchanged)
# ---------------------------------------------------------------------------

def build_observation(cache: TelemetryCache, target_xyz: np.ndarray) -> np.ndarray:
    pos = ned_position_to_train_frame(*cache.position_ned)
    vel = ned_velocity_to_train_frame(*cache.velocity_ned)
    roll, pitch, yaw = cache.attitude_rpy_rad

    pos_error = target_xyz - pos
    rpy_error = np.array([0.0 - roll, 0.0 - pitch, 0.0 - yaw], dtype=np.float32)

    return np.concatenate([pos_error, vel, rpy_error]).astype(np.float32)


def action_to_ned_velocity(raw_action: np.ndarray, current_yaw_deg: float):
    raw_action = np.asarray(raw_action, dtype=np.float32)
    direction = raw_action[0:3]
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-6 else np.zeros(3, dtype=np.float32)
    speed = float(np.clip(raw_action[3], 0.0, 1.0)) * MAX_SPEED_MPS

    vx, vy, vz = direction * speed
    vn, ve, vd = train_direction_to_ned((vx, vy, vz))

    return VelocityNedYaw(vn, ve, vd, current_yaw_deg)


# ---------------------------------------------------------------------------
# Main flow (unchanged except model loading)
# ---------------------------------------------------------------------------

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


async def run(npz_path: str, log_dir: str | None):
    print(f"Loading policy weights from {npz_path} ...")
    policy = NumpyPolicy(npz_path)

    step_rows = []
    csv_path = None
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, "steps.csv")

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

            if log_dir is not None:
                step_rows.append({
                    "step": step_count,
                    "t_s": round(t_s, 4),
                    "north_m": cache.position_ned[0],
                    "east_m": cache.position_ned[1],
                    "down_m": cache.position_ned[2],
                    "vn_m_s": cache.velocity_ned[0],
                    "ve_m_s": cache.velocity_ned[1],
                    "vd_m_s": cache.velocity_ned[2],
                    "roll_rad": cache.attitude_rpy_rad[0],
                    "pitch_rad": cache.attitude_rpy_rad[1],
                    "yaw_rad": cache.attitude_rpy_rad[2],
                    "pos_error_m": pos_error,
                    "raw_action_0": float(raw_action[0]),
                    "raw_action_1": float(raw_action[1]),
                    "raw_action_2": float(raw_action[2]),
                    "raw_action_3": float(raw_action[3]),
                    "cmd_vn_m_s": setpoint.north_m_s,
                    "cmd_ve_m_s": setpoint.east_m_s,
                    "cmd_vd_m_s": setpoint.down_m_s,
                })

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

        if log_dir is not None and step_rows:
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(step_rows[0].keys()))
                writer.writeheader()
                writer.writerows(step_rows)
            print(f"Wrote {len(step_rows)} step rows to {csv_path}")

            errs = np.array([r["pos_error_m"] for r in step_rows])
            second_half = errs[len(errs) // 2:]
            summary = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "loader": "numpy",
                "model_path": npz_path,
                "model_sha256": sha256_of_file(npz_path),
                "control_rate_hz": CONTROL_RATE_HZ,
                "max_speed_mps": MAX_SPEED_MPS,
                "episode_duration_s": EPISODE_DURATION_S,
                "target_altitude_m": TARGET_ALTITUDE_M,
                "target_xyz_train_frame": target_xyz.tolist(),
                "n_steps": len(step_rows),
                "pos_error_mean_m": float(errs.mean()),
                "pos_error_std_m": float(errs.std()),
                "pos_error_min_m": float(errs.min()),
                "pos_error_max_m": float(errs.max()),
                "pos_error_rmse_m": float(np.sqrt(np.mean(errs ** 2))),
                "steady_state_mean_m": float(second_half.mean()),
                "steady_state_std_m": float(second_half.std()),
            }
            with open(os.path.join(log_dir, "summary.json"), "w") as f:
                json.dump(summary, f, indent=2)
            print(f"Wrote summary to {os.path.join(log_dir, 'summary.json')}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to hover_policy.npz")
    parser.add_argument("--log-dir", default=None, help="If set, write steps.csv + summary.json here")
    args = parser.parse_args()
    asyncio.run(run(args.model, args.log_dir))


if __name__ == "__main__":
    main()
