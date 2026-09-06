"""Step 01 - Extract per-trip origin/destination records from raw transactions.

Reads the vendor trip-transaction CSVs for each study day and collapses the
per-gantry rows into one row per trip (first gantry = origin, last = destination).

    python scripts/01_extract_trip_od.py --days 1 2 3
"""
from __future__ import annotations

import argparse

from hotlane.config import PATHS, STUDY_DAYS
from hotlane.data.transactions import build_trip_od, read_day_transactions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, nargs="*", default=STUDY_DAYS)
    parser.add_argument("--direction", default="WB", choices=["WB", "EB"])
    parser.add_argument(
        "--overwrite", action="store_true", help="rebuild days that already exist"
    )
    args = parser.parse_args()

    for day in args.days:
        out_path = PATHS.trip_od(day, args.direction)
        if out_path.exists() and not args.overwrite:
            print(f"Day {day}: already built, skipping")
            continue

        print(f"Day {day}: reading transactions")
        transactions = read_day_transactions(day)
        trip_od = build_trip_od(transactions)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        trip_od.to_csv(out_path, index=False)
        print(f"Day {day}: {len(trip_od):,} trips -> {out_path}")


if __name__ == "__main__":
    main()
