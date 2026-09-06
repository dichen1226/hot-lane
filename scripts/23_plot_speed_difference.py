"""Step 23 - Speed difference between two scenarios.

Compares a baseline panel against a modified one (e.g. a control strategy, or a
re-derived dataset) and plots the mean HOT / GP speed difference by time of
day. A positive value means the modified scenario is faster.

    python scripts/23_plot_speed_difference.py \\
        --baseline data/processed/model_table_WB_2018-10.csv \\
        --scenario data/processed/modified_data_flow_included.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from hotlane.config import PATHS
from hotlane.viz.timeseries import plot_speed_difference

KEYS = ["time_index2", "day", "hour", "minute", "gantry"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", help="baseline panel CSV")
    parser.add_argument("--scenario", required=True, help="modified panel CSV")
    parser.add_argument("--direction", default="WB", choices=["WB", "EB"])
    args = parser.parse_args()

    baseline_path = Path(args.baseline) if args.baseline else PATHS.model_table(args.direction)
    baseline = pd.read_csv(baseline_path)
    scenario = pd.read_csv(args.scenario)

    merged = pd.merge(baseline, scenario, on=KEYS, how="left", suffixes=("_base", "_scenario"))
    merged["speed_diff_HOT"] = merged["speed_HOT_scenario"] - merged["speed_HOT_base"]
    merged["speed_diff_GP"] = merged["speed_GP_scenario"] - merged["speed_GP_base"]

    merged["time"] = (
        merged["hour"].astype(int).astype(str).str.zfill(2)
        + ":"
        + merged["minute"].astype(int).astype(str).str.zfill(2)
    )

    grouped = merged.groupby("time").mean(numeric_only=True).reset_index()
    # The final interval is only partially observed on the last study day.
    grouped = grouped.iloc[:-1]

    plot_speed_difference(
        grouped, PATHS.figure("control_comparison", "speed_difference.jpg")
    )
    print(f"Figure -> {PATHS.figures / 'control_comparison' / 'speed_difference.jpg'}")
    print(
        f"Mean HOT difference: {grouped['speed_diff_HOT'].mean():+.2f} mph\n"
        f"Mean GP  difference: {grouped['speed_diff_GP'].mean():+.2f} mph"
    )


if __name__ == "__main__":
    main()
