"""Step 04 - Merge vehicle classes and emit the analysis-ready model table.

Attaches HOV2 / HOV3+ / violation entry-exit counts to the SOV panel, then
renames columns to the canonical modelling schema:

    volume_o -> num_O      volume_HOT     -> flow_HOT
    volume_d -> num_D      volume_GP_avg  -> flow_GP
                           speed_GP_avg   -> speed_GP

    python scripts/04_build_model_table.py
    python scripts/04_build_model_table.py --sov-only   # skip the HOV/VIO join
"""
from __future__ import annotations

import argparse

import pandas as pd

from hotlane.config import PATHS
from hotlane.data.panel import merge_vehicle_classes, to_model_table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", default="WB", choices=["WB", "EB"])
    parser.add_argument(
        "--sov-only",
        action="store_true",
        help="build the model table from the SOV panel without the HOV/VIO counts",
    )
    parser.add_argument(
        "--class-counts-out",
        help="optional path for the full vehicle-class breakdown table",
    )
    args = parser.parse_args()

    sov = pd.read_csv(PATHS.month_panel("SOV", args.direction))

    if args.sov_only:
        panel = sov
    else:
        classes = {
            name: pd.read_csv(PATHS.month_panel(name, args.direction))
            for name in ("HOV2", "HOV3", "VIO")
        }
        panel = merge_vehicle_classes(sov, classes["HOV2"], classes["HOV3"], classes["VIO"])

        breakdown_path = args.class_counts_out or (
            PATHS.processed / f"vehicle_class_counts_{args.direction}.csv"
        )
        panel.to_csv(breakdown_path, index=False)
        print(f"Vehicle-class breakdown -> {breakdown_path}")

        # merge_vehicle_classes already applies the num_O / num_D rename.
        panel = panel.rename(columns={"num_O": "volume_o", "num_D": "volume_d"})

    model_table = to_model_table(panel)
    out_path = PATHS.model_table(args.direction)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model_table.to_csv(out_path, index=False)

    print(f"Model table -> {out_path}  ({len(model_table):,} rows)")
    print(model_table.head().to_string(index=False))


if __name__ == "__main__":
    main()
