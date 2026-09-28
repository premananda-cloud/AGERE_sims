"""
rl_bridge_common.py

Shared core for the PX4 + numpy-policy bridge: policy loading, frame
conversion, telemetry caching, observation construction, action decoding,
and step/summary logging.

Both rl_px4_bridge_sim.py and rl_px4_bridge_real.py import from here.
Keeping this in one place is the whole point -- the original script's
docstring warned about exactly the drift that happens when "identical"
logic lives in two files. Flight-critical constants (control rate, speed
limit, observation formula) live here ONCE.
"""

import asyncio
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
from mavsdk.offboard import VelocityNedYaw


# ---------------------------------------------------------------------------
# Constants shared by every profile. If you retrain the policy with a
# different CTRL_FREQ / SPEED_LIMIT / obs formula, change it here ONCE.
# ---------------------------------------------------------------------------

CONTROL_RATE_HZ = 30.0    # confirmed: HoverAviary.CTRL_FREQ == 30
MAX_SPEED_MPS = 0.25       # confirmed: HoverAviary.SPEED_LIMIT == 0.25


# ---------------------------------------------------------------------------
# Policy: manual forward pass through exported weights (unchanged logic).
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
# Frame conversion (unchanged).
# ---------------------------------------------------------------------------

def ned_position_to_train_frame(north, east, down):
    return np.array([north, east, -down], dtype=np.float32)


def ned_velocity_to_train_frame(vn, ve, vd):
    return np.array([vn, ve, -vd], dtype=np.float32)


def train_direction_to_ned(direction_xyz):
    x, y, z = direction_xyz
    return x, y, -z


# ---------------------------------------------------------------------------
# Telemetry cache -- now tracks a monotonic timestamp of the last update
# so callers can detect a stale/dead telemetry link (added for real flight,
# but harmless to have in sim too).
# ---------------------------------------------------------------------------

class TelemetryCache:
    def __init__(self):
        self.position_ned = None
        self.velocity_ned = None
        self.attitude_rpy_rad = None
        self.last_update_monotonic = None
        self.ready = asyncio.Event()

    def _touch(self):
        self.last_update_monotonic = time.monotonic()
        if self.position_ned is not None and self.attitude_rpy_rad is not None:
            self.ready.set()

    def age_s(self) -> float:
        """Seconds since the last telemetry sample. inf if never received."""
        if self.last_update_monotonic is None:
            return float("inf")
        return time.monotonic() - self.last_update_monotonic

    async def track_position_velocity(self, drone):
        async for pv in drone.telemetry.position_velocity_ned():
            self.position_ned = (pv.position.north_m, pv.position.east_m, pv.position.down_m)
            self.velocity_ned = (pv.velocity.north_m_s, pv.velocity.east_m_s, pv.velocity.down_m_s)
            self._touch()

    async def track_attitude(self, drone):
        async for att in drone.telemetry.attitude_euler():
            self.attitude_rpy_rad = (
                np.radians(att.roll_deg),
                np.radians(att.pitch_deg),
                np.radians(att.yaw_deg),
            )
            self._touch()


# ---------------------------------------------------------------------------
# Observation / action (unchanged formula, kept in one place).
# ---------------------------------------------------------------------------

def build_observation(cache: TelemetryCache, target_xyz: np.ndarray) -> np.ndarray:
    pos = ned_position_to_train_frame(*cache.position_ned)
    vel = ned_velocity_to_train_frame(*cache.velocity_ned)
    roll, pitch, yaw = cache.attitude_rpy_rad

    pos_error = target_xyz - pos
    rpy_error = np.array([0.0 - roll, 0.0 - pitch, 0.0 - yaw], dtype=np.float32)

    return np.concatenate([pos_error, vel, rpy_error]).astype(np.float32)


def action_to_ned_velocity(raw_action: np.ndarray, current_yaw_deg: float,
                            speed_scale: float = 1.0) -> VelocityNedYaw:
    """
    speed_scale multiplies the policy's speed command AFTER the trained
    MAX_SPEED_MPS clip, so the policy's internal calibration (what "0.25
    m/s of thrust in this direction" means to the network) is untouched --
    it just damps the real-world magnitude. Use < 1.0 for early real
    flights.
    """
    raw_action = np.asarray(raw_action, dtype=np.float32)
    direction = raw_action[0:3]
    norm = np.linalg.norm(direction)
    direction = direction / norm if norm > 1e-6 else np.zeros(3, dtype=np.float32)
    speed = float(np.clip(raw_action[3], 0.0, 1.0)) * MAX_SPEED_MPS * speed_scale

    vx, vy, vz = direction * speed
    vn, ve, vd = train_direction_to_ned((vx, vy, vz))

    return VelocityNedYaw(vn, ve, vd, current_yaw_deg)


# ---------------------------------------------------------------------------
# Logging helpers (unchanged behavior, factored out).
# ---------------------------------------------------------------------------

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class StepLogger:
    """Collects per-step rows and writes steps.csv + summary.json on close()."""

    def __init__(self, log_dir: str | None):
        self.log_dir = log_dir
        self.rows = []
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)

    def log_step(self, *, step, t_s, cache: TelemetryCache, pos_error,
                 raw_action, setpoint: VelocityNedYaw):
        if self.log_dir is None:
            return
        self.rows.append({
            "step": step,
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

    def close(self, *, loader: str, model_path: str, target_xyz: np.ndarray,
              extra_meta: dict | None = None):
        if self.log_dir is None or not self.rows:
            return None

        csv_path = os.path.join(self.log_dir, "steps.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"Wrote {len(self.rows)} step rows to {csv_path}")

        errs = np.array([r["pos_error_m"] for r in self.rows])
        second_half = errs[len(errs) // 2:]
        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "loader": loader,
            "model_path": model_path,
            "model_sha256": sha256_of_file(model_path),
            "control_rate_hz": CONTROL_RATE_HZ,
            "max_speed_mps": MAX_SPEED_MPS,
            "target_xyz_train_frame": target_xyz.tolist(),
            "n_steps": len(self.rows),
            "pos_error_mean_m": float(errs.mean()),
            "pos_error_std_m": float(errs.std()),
            "pos_error_min_m": float(errs.min()),
            "pos_error_max_m": float(errs.max()),
            "pos_error_rmse_m": float(np.sqrt(np.mean(errs ** 2))),
            "steady_state_mean_m": float(second_half.mean()),
            "steady_state_std_m": float(second_half.std()),
        }
        if extra_meta:
            summary.update(extra_meta)

        summary_path = os.path.join(self.log_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Wrote summary to {summary_path}")
        return summary_path
