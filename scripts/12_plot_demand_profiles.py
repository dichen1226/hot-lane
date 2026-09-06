"""Step 12 - Toll rate and SOV demand profiles.

Two views of the same pair of series:
  * day-to-day, one figure per hour (how stable is a given hour across the month);
  * diurnal, averaged over days (how toll and demand move together within a day).

    python scripts/12_plot_demand_profiles.py --gantry 1
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from hotlane.config import PATHS
from hotlane.viz.timeseries import plot_daily_profile, plot_diurnal_profile


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gantry", type=int, default=1, help="origin gantry to profile")
    parser.add_argument("--direction", default="WB", choices=["WB", "EB"])
    args = parser.parse_args()

    data = pd.read_csv(PATHS.model_table(args.direction))

    save_dir = PATHS.figure_dir("demand_profiles")
    for hour in np.arange(data["hour"].min(), data["hour"].max(), 1):
        plot_daily_profile(data, int(hour), args.gantry, save_dir)
    print(f"Day-to-day profiles for gantry {args.gantry} written")

    plot_diurnal_profile(
        data,
        args.gantry,
        PATHS.figure("demand_profiles", f"diurnal_toll_and_flow_W{args.gantry}.jpg"),
    )
    print(f"Figures written under {save_dir}")


if __name__ == "__main__":
    main()
