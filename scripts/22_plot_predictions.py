"""Step 22 - Predicted vs observed speed, per gantry.

Joins the prediction table from step 21 back onto the observed panel and draws
one figure per gantry plus a single grid covering every gantry and both lanes.

    python scripts/22_plot_predictions.py --day 1
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from hotlane.config import PATHS
from hotlane.modeling.predict import TARGET_COLUMNS, merge_observed_and_predicted
from hotlane.viz.timeseries import (
    add_time_column,
    plot_prediction_figure,
    plot_prediction_vs_actual,
)

LANE_TITLES = {"speed_HOT": "HOT lane", "speed_GP": "GP lane"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", default="WB", choices=["WB", "EB"])
    parser.add_argument("--predictions", help="prediction CSV from step 21")
    parser.add_argument("--day", type=int, help="restrict the figures to one day")
    args = parser.parse_args()

    observed = pd.read_csv(PATHS.model_table(args.direction))
    pred_path = args.predictions or (PATHS.processed / f"pred_data_{args.direction}.csv")
    predicted = pd.read_csv(pred_path)

    merged = merge_observed_and_predicted(observed, predicted)

    if args.day is not None:
        merged = merged[merged["day"] == args.day].reset_index(drop=True)
        if merged.empty:
            raise SystemExit(f"No rows for day {args.day}")

    merged = add_time_column(merged)
    gantries = sorted(merged["gantry"].unique())

    for gantry in gantries:
        for target in TARGET_COLUMNS:
            lane = target.split("_")[-1]
            plot_prediction_figure(
                merged,
                gantry,
                f"preds_{lane}",
                target,
                PATHS.figure("predictions", f"gantry{gantry}_{lane}.jpg"),
                LANE_TITLES[target],
            )

    # One grid: rows of gantries, two columns (HOT | GP).
    fig, axs = plt.subplots(len(gantries), 2, figsize=(14, 3 * len(gantries)), squeeze=False)
    for row, gantry in enumerate(gantries):
        for col, target in enumerate(TARGET_COLUMNS):
            lane = target.split("_")[-1]
            plot_prediction_vs_actual(
                axs[row][col], merged, gantry, f"preds_{lane}", target, LANE_TITLES[target]
            )
    plt.tight_layout()
    plt.savefig(PATHS.figure("predictions", "all_gantries.jpg"), dpi=300)
    plt.close(fig)

    print(f"Figures written under {PATHS.figures / 'predictions'}")


if __name__ == "__main__":
    main()
