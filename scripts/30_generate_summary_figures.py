"""Step 30 - Generate figures that summarize the main analysis results.

Writes composite heatmaps, demand profiles, correlations, prediction plots,
and optional training curves to ``outputs/figures/summary/``. Pass a modified
panel with ``--scenario`` to add comparisons against a baseline.

    python scripts/30_generate_summary_figures.py
    python scripts/30_generate_summary_figures.py --with-training
    python scripts/30_generate_summary_figures.py --scenario data/processed/scenario.csv
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
from hotlane.viz.summary import plot_hourly_day_grid, plot_loss_curves, plot_prediction_grid
from hotlane.viz.timeseries import add_time_column

OUT = "summary"

#: Demand and toll profiles use the eastern entry gantry.
PROFILE_GANTRY = 1


def plot_month_average_speeds(data: pd.DataFrame, out: Path) -> None:
    """Space-time heatmap for the HOT lane (left) and adjacent GP lane (right)."""
    averaged = data.groupby(["time", "gantry"]).mean(numeric_only=True).reset_index()
    for lane in ("HOT", "GP"):
        plot_speed_heatmap(
            averaged, f"speed_{lane}", out / f"{lane.lower()}_month_average_speed.jpg",
            title=f"speed_{lane}",
        )


def plot_demand_and_toll_profiles(data: pd.DataFrame, out: Path) -> None:
    """Plot SOV origin flow and toll rate over days at each hour."""
    plot_hourly_day_grid(
        data, "num_O", out / "sov_origin_flow_by_day.jpg",
        aggregate="sum", ylabel="SOV origin flow", color="green",
        ylim=(0, 200), gantry=PROFILE_GANTRY,
    )
    plot_hourly_day_grid(
        data, "toll_rate", out / "toll_rate_by_day.jpg",
        aggregate="mean", ylabel="Toll rate ($)", color="blue",
        ylim=(0.4, 2.3), gantry=PROFILE_GANTRY,
    )


def plot_toll_demand_correlation(data: pd.DataFrame, out: Path) -> None:
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
        out / "toll_rate_origin_flow_correlation.jpg",
        colors={"Pearson": "green", "Spearman": "blue"},
    )


def plot_training_loss(panel: pd.DataFrame, out: Path, epochs: int, lookback: int) -> None:
    """Training and evaluation loss for HOT-lane and GP-lane speed prediction."""
    histories = {}
    for target_col in TARGET_COLUMNS:
        print(f"  training {target_col} for {epochs} epochs")
        features = select_features(panel, target_col)
        scaler = fit_scaler(features)
        train, test = chronological_split(features)
        X_train, y_train = make_sequences(train, scaler, target_col, lookback)
        X_test, y_test = make_sequences(test, scaler, target_col, lookback)
        # eval_every=1 so the curve has a point at every epoch.
        _, histories[target_col] = train_model(
            X_train, y_train, X_test, y_test,
            TrainConfig(n_epochs=epochs, eval_every=1),
        )

    plot_loss_curves(
        histories,
        {
            "speed_HOT": out / "training_loss_hot.jpg",
            "speed_GP": out / "training_loss_gp.jpg",
        },
    )


def plot_speed_predictions(panel: pd.DataFrame, predictions: pd.DataFrame, out: Path,
                           day: int, lookback: int) -> None:
    """Plot predicted and observed traffic speeds for all gantries."""
    merged = merge_observed_and_predicted(panel, predictions)
    merged = merged[merged["day"] == day]
    # The first lookback-1 slots have no prediction and keep observed values.
    merged = merged[merged["time_index2"] >= merged["time_index2"].min() + lookback - 1]
    merged = add_time_column(merged)

    plot_prediction_grid(
        merged, list(range(1, 9)), out / "speed_predictions_gantries_01_08.jpg"
    )
    plot_prediction_grid(
        merged, list(range(9, 18)), out / "speed_predictions_gantries_09_17.jpg"
    )


def plot_baseline_and_scenario_speeds(
    data: pd.DataFrame, out: Path, day: int, scenario: pd.DataFrame | None
) -> None:
    """Plot average speeds for the baseline and an optional scenario."""
    specs = [
        ("HOT", f"day_{day:02d}", data[data["day"] == day]),
        ("GP", f"day_{day:02d}", data[data["day"] == day]),
        ("HOT", "month", None),
        ("GP", "month", None),
    ]

    for lane, scope, subset in specs:
        frame = subset if subset is not None else (
            data.groupby(["time", "gantry"]).mean(numeric_only=True).reset_index()
        )
        plot_speed_heatmap(
            frame, f"speed_{lane}",
            out / f"{lane.lower()}_speed_{scope}_baseline.jpg",
            title=f"speed_{lane}",
        )

        if scenario is None:
            continue
        after = scenario[scenario["day"] == day] if subset is not None else (
            scenario.groupby(["time", "gantry"]).mean(numeric_only=True).reset_index()
        )
        plot_speed_heatmap(
            after, f"speed_{lane}",
            out / f"{lane.lower()}_speed_{scope}_scenario.jpg",
            title=f"speed_{lane}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", default="WB", choices=["WB", "EB"])
    parser.add_argument("--day", type=int, default=1, help="day shown in daily plots")
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--predictions", help="prediction CSV from step 21")
    parser.add_argument("--scenario", help="optional scenario panel to compare with baseline")
    parser.add_argument(
        "--with-training",
        action="store_true",
        help="retrain both models and plot their loss curves (slow)",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    out = PATHS.figure_dir(OUT)

    panel = pd.read_csv(PATHS.model_table(args.direction))
    with_time = add_clock_time(panel)

    print("Month-average space-time heatmaps")
    plot_month_average_speeds(with_time, out)

    print("Flow and toll rate over days")
    plot_demand_and_toll_profiles(panel, out)

    print("Toll rate vs origin flow correlation")
    plot_toll_demand_correlation(panel, out)

    if args.with_training:
        print("Training and validation loss")
        plot_training_loss(panel, out, args.epochs, args.lookback)
    else:
        print("Training loss - skipped (pass --with-training)")

    pred_path = Path(args.predictions) if args.predictions else (
        PATHS.processed / f"pred_data_{args.direction}.csv"
    )
    if pred_path.exists():
        print("Predicted vs observed speed")
        plot_speed_predictions(panel, pd.read_csv(pred_path), out, args.day, args.lookback)
    else:
        print(f"Prediction plots - skipped ({pred_path} not found; run step 21)")

    scenario = None
    if args.scenario:
        scenario = add_clock_time(pd.read_csv(args.scenario))
    print(
        "Baseline/scenario speed heatmaps"
        + ("" if scenario is not None else " (baseline panels only)")
    )
    plot_baseline_and_scenario_speeds(with_time, out, args.day, scenario)

    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
