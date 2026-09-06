"""Step 13 - Fundamental diagram: density, free-flow speed and wave speed.

Adds hourly flow and density to the model table, plots flow against density,
and reads the free-flow speed ``u_f`` and backward wave speed ``w`` off the two
branches of the diagram.

    python scripts/13_fundamental_diagram.py --critical-density 30
"""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from hotlane.config import PATHS
from hotlane.features import add_density, fundamental_diagram_parameters


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", default="WB", choices=["WB", "EB"])
    parser.add_argument(
        "--critical-density",
        type=float,
        default=30,
        help="drawn as a vertical reference line (veh/mile)",
    )
    parser.add_argument("--free-flow-range", type=float, nargs=2, default=(1, 20))
    parser.add_argument("--congested-range", type=float, nargs=2, default=(40, 64))
    parser.add_argument("--save-table", action="store_true", help="write the density table")
    args = parser.parse_args()

    data = add_density(pd.read_csv(PATHS.model_table(args.direction)))

    plt.figure(figsize=(6, 5))
    plt.scatter(data["density_HOT"], data["flow_HOT_hr"], s=4)
    plt.axvline(x=args.critical_density, color="red", label="Critical density")
    plt.xlabel("Density (veh/mile)")
    plt.ylabel("Flow (veh/hour)")
    plt.title("HOT lane fundamental diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PATHS.figure("fundamental_diagram", "fd_HOT.jpg"), dpi=300)
    plt.close()

    params = fundamental_diagram_parameters(
        data,
        free_flow_range=tuple(args.free_flow_range),
        congested_range=tuple(args.congested_range),
    )
    print(f"Free-flow speed u_f = {params['u_f']:.2f} mph")
    print(f"Wave speed      w   = {params['w']:.2f} mph")

    if args.save_table:
        out_path = PATHS.processed / f"model_table_with_density_{args.direction}.csv"
        data.to_csv(out_path, index=False)
        print(f"Density table -> {out_path}")


if __name__ == "__main__":
    main()
