"""Step 10 - Correlation figures.

Covers four questions from the EDA:
  1. how correlated are gantries with each other, per variable;
  2. how correlated are HOT and general-purpose lanes;
  3. does the posted toll rate track SOV entry volume, hour by hour;
  4. does what a trip paid track how long it took.

    python scripts/10_plot_correlations.py --method spearman
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from hotlane.config import N_GANTRIES, PATHS
from hotlane.viz.correlation import (
    hourly_correlation,
    hourly_travel_time_toll_correlation,
    plot_correlation_profile,
    plot_gantry_correlations,
    plot_pair_correlation,
)

GANTRY_VARIABLES = ["flow_HOT", "flow_GP", "speed_HOT", "speed_GP", "toll_rate"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--direction", default="WB", choices=["WB", "EB"])
    args = parser.parse_args()

    method = args.method
    data = pd.read_csv(PATHS.model_table(args.direction))
    data = data.sort_values(["time_index2", "gantry"]).reset_index(drop=True)

    # 1. gantry-to-gantry -------------------------------------------------- #
    plot_gantry_correlations(
        data,
        GANTRY_VARIABLES,
        PATHS.figure("correlation", f"correlation_gantry_{method}.jpg"),
        method,
    )
    plot_gantry_correlations(
        data,
        GANTRY_VARIABLES,
        PATHS.figure("correlation", f"correlation_gantry_shared_scale_{method}.jpg"),
        method,
        shared_scale=True,
    )

    # 2. HOT vs GP --------------------------------------------------------- #
    for label, columns in (
        ("speed", ["speed_HOT", "speed_GP"]),
        ("flow", ["flow_HOT", "flow_GP"]),
    ):
        corr = plot_pair_correlation(
            data, columns, PATHS.figure("correlation", f"corr_HOT_GP_{label}_{method}.jpg"), method
        )
        print(f"HOT vs GP {label} ({method}):\n{corr}\n")

    # 3. toll rate vs SOV entries, by hour --------------------------------- #
    gantries = list(range(1, N_GANTRIES[args.direction] + 1))
    hours = np.arange(data["hour"].min(), data["hour"].max(), 1)
    profiles = {
        "Pearson": [hourly_correlation(data, h, gantries, "pearson") for h in hours],
        "Spearman": [hourly_correlation(data, h, gantries, "spearman") for h in hours],
    }
    plot_correlation_profile(
        hours,
        profiles,
        "Correlation between toll rate and SOV origin flow",
        PATHS.figure("correlation", "corr_toll_rate_origin_flow_all_gantries.jpg"),
        colors={"Pearson": "green", "Spearman": "blue"},
    )

    # 4. trip travel time vs toll paid ------------------------------------- #
    trips_path = PATHS.trip_travel_times("SOV", args.direction)
    if not trips_path.exists():
        print(f"Skipping travel-time correlations: {trips_path} not found "
              "(run scripts/05_build_trip_travel_times.py first)")
        return

    trips = pd.read_csv(trips_path)
    for label, subset in (
        ("all_gantries", trips),
        ("origin_W1_W2", trips[trips["Origin"].isin([1, 2])]),
    ):
        trip_hours = np.arange(subset["hour"].min(), subset["hour"].max() + 1, 1)
        values = [hourly_travel_time_toll_correlation(subset, h, method) for h in trip_hours]
        plot_correlation_profile(
            trip_hours,
            {"Correlation": values},
            "Correlation between travel time and tolls paid",
            PATHS.figure("correlation", f"corr_traveltime_toll_{label}_{method}.jpg"),
            colors={"Correlation": "blue"},
        )

    print(f"Figures written under {PATHS.figures / 'correlation'}")


if __name__ == "__main__":
    main()
