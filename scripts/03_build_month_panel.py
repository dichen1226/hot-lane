"""Step 03 - Join OD flows, posted tolls and detector traffic into a month panel.

This is the table the whole study is built on: one row per
(15-minute interval, gantry) for the study month.

Toll rates and detector traffic only make sense for the paying (SOV) class, so
by default they are joined only there; for HOV/violation classes pass
``--no-toll --no-traffic`` and the panel carries entry/exit counts alone
(step 04 merges those onto the SOV panel).

    python scripts/03_build_month_panel.py --payment-class SOV
    python scripts/03_build_month_panel.py --payment-class HOV2 --no-toll --no-traffic
"""
from __future__ import annotations

import argparse

import pandas as pd

from hotlane.config import N_GANTRIES, PATHS, PAYMENT_CLASSES, STUDY_DAYS
from hotlane.data.panel import (
    assign_month_time_index,
    audit_panel,
    expected_slot_count,
    fill_missing_gantries,
    merge_daily_sources,
)
from hotlane.data.tolls import toll_rates_for_day
from hotlane.data.traffic import traffic_for_day


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, nargs="*", default=STUDY_DAYS)
    parser.add_argument("--payment-class", default="SOV", choices=list(PAYMENT_CLASSES))
    parser.add_argument("--direction", default="WB", choices=["WB", "EB"])
    parser.add_argument("--no-toll", action="store_true", help="skip the toll-rate join")
    parser.add_argument("--no-traffic", action="store_true", help="skip the detector join")
    args = parser.parse_args()

    daily_panels = []
    for day in args.days:
        origins = pd.read_csv(PATHS.od_flows(day, args.payment_class, "o", args.direction))
        destinations = pd.read_csv(PATHS.od_flows(day, args.payment_class, "d", args.direction))

        tolls = None if args.no_toll else toll_rates_for_day(day)
        traffic = None if args.no_traffic else traffic_for_day(day, args.direction)

        panel = merge_daily_sources(origins, destinations, tolls, traffic)
        out_path = PATHS.daily_panel(day, args.payment_class, args.direction)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_csv(out_path, index=False)

        daily_panels.append(panel)
        print(f"Day {day}: {len(panel):,} rows")

    month = pd.concat(daily_panels).reset_index(drop=True)
    month = assign_month_time_index(month, args.days)

    gaps = audit_panel(month, N_GANTRIES[args.direction])
    if len(gaps):
        print(f"\n{len(gaps)} time slots are missing at least one gantry:")
        print(gaps.to_string(index=False))

    month = fill_missing_gantries(month, N_GANTRIES[args.direction])

    out_path = PATHS.month_panel(args.payment_class, args.direction)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    month.to_csv(out_path, index=False)

    observed = month["time_index2"].nunique()
    print(
        f"\nWrote {out_path}\n"
        f"  rows        : {len(month):,}\n"
        f"  time slots  : {observed:,} of {expected_slot_count(args.days):,} expected"
    )


if __name__ == "__main__":
    main()
