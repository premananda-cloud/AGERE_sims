"""
run_experiments.py

Runs rl_px4_bridge_numpy.py N times back-to-back against a live PX4 SITL
instance, each run writing its own steps.csv + summary.json (via the
bridge script's --log-dir option), then aggregates all runs' summaries
into one CSV table.

Requires PX4 SITL + Gazebo already running (same as running the bridge
script by hand). Runs sequentially, not in parallel -- each run does its
own arm/takeoff/hover/land cycle, so the sim needs to be back in a
"ready to arm again" state between runs, which landing already gives you.

Usage:
    python run_experiments.py --model ../model/hover_policy.npz --runs 3
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time


# Maps known bridge script filenames to a short tag used in output naming.
# Falls back to the script's filename (minus .py) for anything else.
KNOWN_LOADER_TAGS = {
    "rl_px4_bridge.py": "sb3",
    "rl_px4_bridge_numpy.py": "numpy",
}


def infer_tag(bridge_script: str) -> str:
    basename = os.path.basename(bridge_script)
    return KNOWN_LOADER_TAGS.get(basename, os.path.splitext(basename)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to hover_champion.zip or hover_policy.npz")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    parser.add_argument("--out-dir", default="../logs", help="Where to store per-run logs + aggregate")
    parser.add_argument("--pause-between-s", type=float, default=15.0,
                         help="Rest between runs, lets the sim settle after landing")
    parser.add_argument("--bridge-script", default="rl_px4_bridge_numpy.py",
                         help="Path to the bridge script to invoke")
    parser.add_argument("--tag", default=None,
                         help="Label for this batch (default: inferred from --bridge-script, "
                              "e.g. 'sb3' or 'numpy')")
    args = parser.parse_args()

    tag = args.tag or infer_tag(args.bridge_script)
    model_stem = os.path.splitext(os.path.basename(args.model))[0]
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    batch_name = f"batch_{tag}_{model_stem}_{run_stamp}"
    batch_dir = os.path.join(args.out_dir, batch_name)
    os.makedirs(batch_dir, exist_ok=True)
    print(f"Batch tag: {tag}  |  model: {model_stem}  |  output: {batch_dir}")

    summaries = []

    for i in range(1, args.runs + 1):
        run_dir = os.path.join(batch_dir, f"run_{i}")
        print(f"\n=== Run {i}/{args.runs} -> {run_dir} ===")

        stdout_path = os.path.join(batch_dir, f"run_{i}_stdout.log")
        cmd = [
            sys.executable, args.bridge_script,
            "--model", args.model,
            "--log-dir", run_dir,
        ]

        with open(stdout_path, "w") as stdout_file:
            result = subprocess.run(cmd, stdout=stdout_file, stderr=subprocess.STDOUT)

        if result.returncode != 0:
            print(f"  Run {i} exited with code {result.returncode} -- see {stdout_path}. "
                  f"Skipping from aggregate; check PX4/Gazebo state before continuing.")
            continue

        summary_path = os.path.join(run_dir, "summary.json")
        if not os.path.exists(summary_path):
            print(f"  Run {i} produced no summary.json (likely errored before landing) -- "
                  f"see {stdout_path}. Skipping from aggregate.")
            continue

        with open(summary_path) as f:
            summary = json.load(f)
        summary["run_index"] = i
        summaries.append(summary)
        print(f"  Run {i} done: mean pos error {summary['pos_error_mean_m']:.4f} m "
              f"(steady-state {summary['steady_state_mean_m']:.4f} m)")

        if i < args.runs:
            print(f"  Pausing {args.pause_between_s}s before next run...")
            time.sleep(args.pause_between_s)

    if not summaries:
        print("\nNo successful runs to aggregate.")
        return

    agg_path = os.path.join(batch_dir, f"aggregate_summary_{tag}_{model_stem}.csv")
    fieldnames = list(summaries[0].keys())
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)

    means = [s["pos_error_mean_m"] for s in summaries]
    steady = [s["steady_state_mean_m"] for s in summaries]
    print(f"\n=== {len(summaries)}/{args.runs} runs succeeded ({tag}, {model_stem}) ===")
    print(f"Per-run mean pos error (m): {[round(m, 4) for m in means]}")
    print(f"Per-run steady-state mean (m): {[round(s, 4) for s in steady]}")
    print(f"Aggregate table written to: {agg_path}")
    print(f"Full per-step traces + per-run summaries under: {batch_dir}")


if __name__ == "__main__":
    main()
