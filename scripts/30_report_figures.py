"""Step 30 - Regenerate Figures 5-14 of the project report.

Writes to ``outputs/figures/report_regenerated/``. The published versions,
extracted from the report PDF, sit alongside in ``outputs/figures/report/``;
``outputs/README.md`` maps figure numbers to both.

Figures 11-14 compare speeds before and after the proposed (LQR) tolling
strategy. Only the "before" panels come from this repository — the controlled
scenario is produced by separate code that is not part of it. Pass
``--scenario <panel.csv>`` if you have that output and the "after" panels will
be drawn too.

    python scripts/30_report_figures.py                     # figures 5-8, 10-14 (before)
    python scripts/30_report_figures.py --with-training     # adds figure 9
    python scripts/30_report_figures.py --scenario data/processed/lqr_panel.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hotlane.config import N_GANTRIES, PATHS
from hotlane.modeling.dataset import (
    DEFAULT_LOOKBACK,
    chronological_split,
    fit_scaler,
    make_sequences,
    select_features,
)
from hotlane.modeling.predict import TARGET_COLUMNS, merge_observed_and_predicted
from hotlane.modeling.train import TrainConfig, train_model
from hotlane.viz.correlation import hourly_correlation, plot_correlation_profile
from hotlane.viz.heatmaps import add_clock_time, plot_speed_heatmap
from hotlane.viz.report import plot_hourly_day_grid, plot_loss_curves, plot_prediction_grid
from hotlane.viz.timeseries import add_time_column

OUT = "report_regenerated"

#: Figure 6 / 7 profile the entry gantry at the eastern end of the corridor.
PROFILE_GANTRY = 1


def figure_5(data: pd.DataFrame, out: Path) -> None:
    """Space-time heatmap for the HOT lane (left) and adjacent GP lane (right)."""
    averaged = data.groupby(["time", "gantry"]).mean(numeric_only=True).reset_index()
    for lane in ("HOT", "GP"):
        plot_speed_heatmap(
            averaged, f"speed_{lane}", out / f"fig05_{lane.lower()}_month_average.jpg",
            title=f"speed_{lane}",
        )


def figures_6_and_7(data: pd.DataFrame, out: Path) -> None:
    """SOV origin flow (6) and toll rate (7) over days, at each hour of day."""
    plot_hourly_day_grid(
        data, "num_O", out / "fig06_sov_origin_flow_over_days.jpg",
        aggregate="sum", ylabel="SOV origin flow", color="green",
        ylim=(0, 200), gantry=PROFILE_GANTRY,
    )
    plot_hourly_day_grid(
        data, "toll_rate", out / "fig07_toll_rate_over_days.jpg",
        aggregate="mean", ylabel="Toll rate ($)", color="blue",
        ylim=(0.4, 2.3), gantry=PROFILE_GANTRY,
    )


def figure_8(data: pd.DataFrame, out: Path) -> None:
    """Correlation between toll rate and origin volume over time of day."""
    gantries = list(range(1, N_GANTRIES["WB"] + 1))
    hours = np.arange(data["hour"].min(), data["hour"].max(), 1)
    plot_correlation_profile(
        hours,
        {
            "Pearson": [hourly_correlation(data, h, gantries, "pearson") for h in hours],
            "Spearman": [hourly_correlation(data, h, gantries, "spearman") for h in hours],
        },
        "Correlation between toll rate and SOV origin flow",
        out / "fig08_corr_toll_rate_origin_flow.jpg",
        colors={"Pearson": "green", "Spearman": "blue"},
    )


def figure_9(panel: pd.DataFrame, out: Path, epochs: int, lookback: int) -> None:
    """Training and evaluation loss for HOT-lane and GP-lane speed prediction."""
    histories = {}
    for target_col in TARGET_COLUMNS:
        print(f"  training {target_col} for {epochs} epochs")
        features = select_features(panel, target_col)
        scaler = fit_scaler(features)
        train, test = chronological_split(features)
        X_train, y_train = make_sequences(train, scaler, target_col, lookback)
        X_test, y_test = make_sequences(test, scaler, target_col, lookback)
        # eval_every=1 so the curve has a point at every epoch, as published.
        _, histories[target_col] = train_model(
            X_train, y_train, X_test, y_test,
            TrainConfig(n_epochs=epochs, eval_every=1),
        )

    plot_loss_curves(
        histories,
        {
            "speed_HOT": out / "fig09_loss_hot.jpg",
            "speed_GP": out / "fig09_loss_gp.jpg",
        },
    )


def figures_10a_10b(panel: pd.DataFrame, predictions: pd.DataFrame, out: Path,
                    day: int, lookback: int) -> None:
    """Testing predictions of traffic speed, gantries 1-8 (A) and 9-17 (B)."""
    merged = merge_observed_and_predicted(panel, predictions)
    merged = merged[merged["day"] == day]
    # The first lookback-1 slots have no prediction and keep observed values.
    merged = merged[merged["time_index2"] >= merged["time_index2"].min() + lookback - 1]
    merged = add_time_column(merged)

    plot_prediction_grid(merged, list(range(1, 9)), out / "fig10A_predictions_gantry_1_8.jpg")
    plot_prediction_grid(merged, list(range(9, 18)), out / "fig10B_predictions_gantry_9_17.jpg")


def figures_11_to_14(
    data: pd.DataFrame, out: Path, day: int, scenario: pd.DataFrame | None
) -> None:
    """Average speed before / after the proposed tolling strategy."""
    specs = [
        ("fig11", "HOT", f"day {day}", data[data["day"] == day]),
        ("fig12", "GP", f"day {day}", data[data["day"] == day]),
        ("fig13", "HOT", "one month", None),
        ("fig14", "GP", "one month", None),
    ]

    for tag, lane, scope, subset in specs:
        frame = subset if subset is not None else (
            data.groupby(["time", "gantry"]).mean(numeric_only=True).reset_index()
        )
        plot_speed_heatmap(
            frame, f"speed_{lane}",
            out / f"{tag}_{lane.lower()}_{scope.replace(' ', '_')}_before.jpg",
            title=f"speed_{lane}",
        )

        if scenario is None:
            continue
        after = scenario[scenario["day"] == day] if subset is not None else (
            scenario.groupby(["time", "gantry"]).mean(numeric_only=True).reset_index()
        )
        plot_speed_heatmap(
            after, f"speed_{lane}",
            out / f"{tag}_{lane.lower()}_{scope.replace(' ', '_')}_after.jpg",
            title=f"speed_{lane}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", default="WB", choices=["WB", "EB"])
    parser.add_argument("--day", type=int, default=1, help="day shown in figures 10-12")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--predictions", help="prediction CSV from step 21")
    parser.add_argument("--scenario", help="controlled-scenario panel for figures 11-14")
    parser.add_argument(
        "--with-training",
        action="store_true",
        help="retrain both models to produce figure 9 (slow)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = PATHS.figure_dir(OUT)

    panel = pd.read_csv(PATHS.model_table(args.direction))
    with_time = add_clock_time(panel)

    print("Figure 5  - space-time heatmaps")
    figure_5(with_time, out)

    print("Figures 6, 7 - flow and toll rate over days")
    figures_6_and_7(panel, out)

    print("Figure 8  - toll rate vs origin flow correlation")
    figure_8(panel, out)

    if args.with_training:
        print("Figure 9  - training and validation loss")
        figure_9(panel, out, args.epochs, args.lookback)
    else:
        print("Figure 9  - skipped (pass --with-training)")

    pred_path = Path(args.predictions) if args.predictions else (
        PATHS.processed / f"pred_data_{args.direction}.csv"
    )
    if pred_path.exists():
        print("Figures 10A, 10B - predicted vs observed speed")
        figures_10a_10b(panel, pd.read_csv(pred_path), out, args.day, args.lookback)
    else:
        print(f"Figures 10A, 10B - skipped ({pred_path} not found; run step 21)")

    scenario = None
    if args.scenario:
        scenario = add_clock_time(pd.read_csv(args.scenario))
    print(
        "Figures 11-14 - before/after speed heatmaps"
        + ("" if scenario is not None else " (before panels only)")
    )
    figures_11_to_14(with_time, out, args.day, scenario)

    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
