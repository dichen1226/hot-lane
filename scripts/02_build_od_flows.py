"""Step 02 - Aggregate trip OD records into 15-minute gantry entry/exit counts.

Produces, for each day and vehicle class, two long tables: entries (origins)
and exits (destinations) per gantry per 15-minute interval.

    python scripts/02_build_od_flows.py --payment-class SOV
    python scripts/02_build_od_flows.py --payment-class HOV2 HOV3 VIO
"""
from __future__ import annotations

import argparse

from hotlane.config import PATHS, PAYMENT_CLASSES, STUDY_DAYS
from hotlane.data.flows import od_flow_matrices, stack_matrix
from hotlane.data.transactions import load_trip_od

#: (output side, time column, OD column)
SIDES = [("o", "Start_Time", "Origin"), ("d", "End_Time", "Destination")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, nargs="*", default=STUDY_DAYS)
    parser.add_argument(
        "--payment-class", nargs="*", default=["SOV"], choices=list(PAYMENT_CLASSES)
    )
    parser.add_argument("--direction", default="WB", choices=["WB", "EB"])
    parser.add_argument(
        "--no-impute-destination",
        action="store_true",
        help="drop single-gantry trips instead of imputing destination = origin + 1",
    )
    args = parser.parse_args()

    for payment_class in args.payment_class:
        for day in args.days:
            trip_od = load_trip_od(day, args.direction)

            for side, time_col, od_col in SIDES:
                west, east = od_flow_matrices(
                    trip_od,
                    time_col,
                    od_col,
                    payment_class,
                    impute_missing_destination=not args.no_impute_destination,
                )
                matrix = west if args.direction == "WB" else east
                flows = stack_matrix(matrix, args.direction)
                flows["day"] = day

                out_path = PATHS.od_flows(day, payment_class, side, args.direction)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                flows.to_csv(out_path, index=False)

            print(f"{payment_class} day {day}: OD flows written")


if __name__ == "__main__":
    main()
