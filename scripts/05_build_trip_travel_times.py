"""Step 05 - Build the trip-level travel time and toll-paid table.

Used by the EDA that correlates what a driver paid with how long the trip took.
The original analysis sampled 1,000 trips per day because expanding every trip
was too slow; keep ``--sample-size`` set for comparability with those results.

    python scripts/05_build_trip_travel_times.py --payment-class SOV
"""
from __future__ import annotations

import argparse

import pandas as pd

from hotlane.config import DIRECTION_CODES, PATHS, PAYMENT_CLASSES, STUDY_DAYS
from hotlane.data.transactions import build_trip_od, read_day_transactions
from hotlane.features import trip_travel_times


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, nargs="*", default=STUDY_DAYS)
    parser.add_argument("--payment-class", default="SOV", choices=list(PAYMENT_CLASSES))
    parser.add_argument("--direction", default="WB", choices=["WB", "EB"])
    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
        help="trips sampled per day; pass 0 to use every trip",
    )
    parser.add_argument("--random-state", type=int, default=123)
    args = parser.parse_args()

    payment_values = PAYMENT_CLASSES[args.payment_class]
    direction_codes = DIRECTION_CODES[args.direction]

    daily = []
    for day in args.days:
        print(f"Day {day}")
        transactions = read_day_transactions(day)
        trips = build_trip_od(
            transactions,
            include_fare=True,
            drop_single_gantry_trips=True,
            sample_size=args.sample_size or None,
            random_state=args.random_state,
        )

        trips = trips[
            trips["direction"].isin(direction_codes)
            & trips["Payment Type"].isin(payment_values)
        ].reset_index(drop=True)

        trips = trip_travel_times(trips)
        trips["day"] = day
        daily.append(trips)

    out = pd.concat(daily).reset_index(drop=True)
    out = out.sort_values(["day", "hour", "minute", "second"]).reset_index(drop=True)

    out_path = PATHS.trip_travel_times(args.payment_class, args.direction)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"\n{len(out):,} trips -> {out_path}")


if __name__ == "__main__":
    main()
