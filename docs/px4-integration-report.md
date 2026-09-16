# AGERE_sims: RL Hover Policy → PX4 SITL Integration

Status: **SITL validated. Not yet flown on real hardware.**
Last updated: 2026-09-16

## What this repo is

AGERE_sims is the deployment/integration half of the AGERE project.
Training, reward design, and evaluation live in `AGERE` (the sim-only
PyBullet + Gymnasium codebase). This repo is where a *trained* checkpoint
gets connected to an actual flight stack — PX4 running against Gazebo
today, real hardware next — and where it's proven the model can fly
something outside the training simulator before it's trusted to fly
something outside any simulator at all.

The honest scope of this document: this is a single-pass integration
report, not a benchmark. Numbers below are from the runs that were
actually done while building this, kept because they're useful evidence
the pipeline works — not because n=4 constitutes a proper evaluation.
When this goes toward a paper, it gets redone properly: multiple seeds,
multiple episodes per condition, statistical treatment. That's future
work, flagged here so nobody mistakes this for it.

## Stack

- **Flight controller:** PX4 (SITL), running locally
- **Physics/visual sim:** Gazebo
- **Link:** MAVSDK (Python), UDP, `udpin://0.0.0.0:14540`
- **Policy:** PPO, trained in `AGERE` on `gym-pybullet-drones`'
  `HoverAviary` (`DroneModel.CF2X` — Crazyflie 2.X, ~27g, T/W 2.25),
  `ActionType.VEL`, `ObservationType.KIN`
- **Checkpoint used:** `hover_champion.zip` (hover-only; see AGERE's own
  weight_manager registry for lineage/eval history — not reproduced here)

## What was built, in order

### 1. `test_flight.py` (pre-existing) — MAVSDK connectivity check
Arm, takeoff, print altitude, land, using PX4's own high-level `action`
plugin. No RL involved. Confirmed the PX4↔Gazebo↔MAVSDK link works at
all before building anything on top of it. **Result: works.**

### 2. `rl_px4_bridge.py` — RL policy in the loop, via stable-baselines3
Loads `hover_champion.zip` directly with `PPO.load()`, runs
`model.predict()` against live PX4 telemetry each control step, sends
the result as a MAVSDK `Offboard` velocity setpoint.

Two calibration values had to be pulled from the actual
`gym_pybullet_drones.HoverAviary` instance rather than assumed, because
getting either wrong produces a policy that *runs* but degrades subtly
rather than erroring:

| Constant | Wrong initial guess | Confirmed value | Effect of being wrong |
|---|---|---|---|
| `SPEED_LIMIT` (max commanded m/s) | 1.0 | **0.25** | 4x too much authority per action → overshoot, oscillation |
| `CTRL_FREQ` (control loop rate) | 20 Hz | **30 Hz** | timing mismatch vs. what the policy was trained against |

**First attempt (wrong constants):** flew, but oscillated, pos error
0.1–0.36m, not converging.
**After correction:** pos error settled into the 0.02–0.1m range and
stayed there for the full 30s hold. This is the first empirical result
below.

### 3. `weight_manager/export_npz.py` (in AGERE) — checkpoint → plain numpy
Pulls `mlp_extractor.policy_net.{0,2}` and `action_net` weights out of
the SB3 `.state_dict()` and saves them as a flat `.npz`
(`W1,b1,W2,b2,Wa,ba`). Generalized from a one-off `extract_weight.py`
script into a proper `weight_manager/` tool (`--model`/`--out` args)
since "package a checkpoint for deployment" is the same kind of
artifact-lifecycle concern the rest of `weight_manager/` already handles.

Network confirmed as SB3's default `MlpPolicy` shape: `net_arch=[64,64]`,
Tanh activations, Gaussian action head (mean only needed —
deterministic inference doesn't need the log-std).

### 4. `rl_px4_bridge_numpy.py` — same bridge, no torch/sb3/gymnasium
Identical flight logic to `rl_px4_bridge.py` (same observation
construction, same frame conversion, same `SPEED_LIMIT`/`CTRL_FREQ`) —
only the policy forward pass changed, from `PPO.predict()` to three
manual matmuls + two `tanh`s against the exported `.npz`. Purpose: cut
the dependency footprint and compute cost for running on a Raspberry Pi,
where installing torch/sb3/gymnasium isn't practical.

**Result: numerically consistent with the sb3 version** (see numbers
below) — the export is a faithful copy of the policy, not an
approximation.

## Known assumptions, not yet independently verified

Flagged honestly rather than buried, since these matter more on real
hardware than they did in SITL:

- **Observation formula.** Built as `[pos_error(3), velocity(3),
  rpy_error(3)]` (target_rpy assumed `(0,0,0)`), inferred from AGERE's
  `status.md` description rather than read directly out of
  `hover_gym_wrapper.py`. The fact that pos_error converges and stays
  bounded is decent indirect evidence this is right, but it hasn't been
  diffed against the actual training-side obs code.
- **Axis mapping.** PyBullet world frame → PX4 NED assumed as
  `x→North, y→East, z(up)→−Down`. Same status: works empirically, not
  independently confirmed by a dedicated axis test.
- **Airframe mismatch is the biggest open risk, not the software.** The
  policy was trained on a Crazyflie-class (~27g) dynamics model. Whatever
  the real airframe turns out to be, if it's not close to that mass/T:W
  class, the policy's corrective outputs are scaled for the wrong
  vehicle — independent of anything in this bridge being correct.

## Empirical results (SITL, this integration pass only)

4 runs, 30s each at 30Hz, hover-hold at ~0.8–1.0m altitude, position
error measured every 1s against the post-takeoff captured target. Two
runs on the sb3-loaded policy, two on the numpy-ported one, all with
corrected `SPEED_LIMIT`/`CTRL_FREQ`.

| Run | Loader | n | mean pos err (m) | std (m) | min (m) | max (m) |
|---|---|---|---|---|---|---|
| 1 | sb3 (`PPO.load`) | 29 | 0.061 | 0.021 | 0.017 | 0.109 |
| 2 | sb3 (`PPO.load`) | 29 | 0.058 | 0.020 | 0.023 | 0.098 |
| 3 | numpy (`.npz`) | 29 | 0.059 | 0.025 | 0.007 | 0.121 |
| 4 | numpy (`.npz`) | 29 | 0.052 | 0.017 | 0.019 | 0.091 |
| **sb3 pooled** | | 58 | **0.060** | 0.020 | | |
| **numpy pooled** | | 58 | **0.055** | 0.022 | | |
| **all runs** | | 116 | **0.058** | 0.021 | 0.007 | 0.121 |

No crashes, no failed arms, no offboard rejections, no manual
interventions, across all 4 runs. All landed cleanly on script-driven
`action.land()`.

**For context, not a fair comparison:** AGERE's own PyBullet eval
reports tighter convergence (roughly 0.015–0.025m). The SITL numbers
being ~2–4x looser is expected, not concerning — Gazebo's physics/motor
model isn't PyBullet's, and this bridge adds a real MAVSDK
telemetry→policy→offboard round-trip that the training sim never had.
Closing that gap is a training-side problem (domain randomization,
latency-aware training), not something to chase by further tuning this
bridge.

One incidental data point worth recording: attempting to load the
`.npz` file with the sb3 script (`rl_px4_bridge.py --model
hover_policy.npz`) fails immediately with `AssertionError: No data found
in the saved file` — correct behavior, not a bug. The two scripts expect
different file formats and aren't interchangeable; the failure mode is
loud and immediate rather than silent.

## What "done" means here, and what's next

Done: closed-loop RL control of a PX4/Gazebo-simulated drone, in both a
full-framework and a framework-free form, holding position within a
tight, repeatable, safe band, with no manual recovery needed across 4
runs.

Not done, and explicitly out of scope for this pass:
- Any real hardware flight
- A safety watchdog in the bridge scripts (no auto-land-on-divergence,
  no geofence in code — RC override + FC-level geofence/altitude limits
  are the current safety plan, and need confirming before real flight)
- Independent verification of the observation formula and axis mapping
  above
- Any statistically rigorous multi-seed, multi-condition evaluation —
  reserved for if/when this becomes a paper
