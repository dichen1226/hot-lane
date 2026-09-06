"""Step 11 - Time-space speed heatmaps.

One heatmap per day plus a month-average, for both the HOT lane and the
general-purpose lanes. Congestion shows up as a red band travelling backwards
across the gantries.

    python scripts/11_plot_speed_heatmaps.py --days 1 2 --average
"""
from __future__ import annotations

import argparse

import pandas as pd

from hotlane.config import PATHS, STUDY_DAYS
from hotlane.viz.heatmaps import add_clock_time, plot_speed_heatmap

VARIABLES = ["speed_HOT", "speed_GP"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, nargs="*", default=STUDY_DAYS)
    parser.add_argument("--direction", default="WB", choices=["WB", "EB"])
    parser.add_argument("--average", action="store_true", help="also plot the month average")
    parser.add_argument("--vmin", type=float, default=30)
    parser.add_argument("--vmax", type=float, default=90)
    args = parser.parse_args()

    data = add_clock_time(pd.read_csv(PATHS.model_table(args.direction)))

    for day in args.days:
        day_data = data[data["day"] == day].reset_index(drop=True)
        if day_data.empty:
            print(f"Day {day}: no rows, skipping")
            continue
        for variable in VARIABLES:
            plot_speed_heatmap(
                day_data,
                variable,
                PATHS.figure("heatmap", f"heatmap_{variable}_day{day}.jpg"),
                vmin=args.vmin,
                vmax=args.vmax,
                title=f"{variable} - day {day}",
            )
        print(f"Day {day}: heatmaps written")

    if args.average:
        averaged = data.groupby(["time", "gantry"]).mean(numeric_only=True).reset_index()
        for variable in VARIABLES:
            plot_speed_heatmap(
                averaged,
                variable,
                PATHS.figure("heatmap", f"heatmap_{variable}_month_average.jpg"),
                vmin=args.vmin,
                vmax=args.vmax,
                title=f"{variable} - month average",
            )
        print("Month-average heatmaps written")

    print(f"Figures written under {PATHS.figures / 'heatmap'}")


if __name__ == "__main__":
    main()
