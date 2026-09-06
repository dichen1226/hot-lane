"""Step 21 - Roll the trained models across the panel to produce predictions.

Walks the month one 15-minute step at a time, feeding the previous ``lookback``
intervals to both models and writing the predicted speeds back into a copy of
the panel. Requires the checkpoints from step 20.

    python scripts/21_predict_lstm.py
"""
from __future__ import annotations

import argparse

import pandas as pd

from hotlane.config import PATHS
from hotlane.modeling.dataset import DEFAULT_LOOKBACK
from hotlane.modeling.predict import TARGET_COLUMNS, model_path, predict_panel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direction", default="WB", choices=["WB", "EB"])
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--out", help="output CSV (default: data/processed/pred_data.csv)")
    args = parser.parse_args()

    missing = [t for t in TARGET_COLUMNS if not model_path(t).exists()]
    if missing:
        raise SystemExit(
            f"Missing checkpoints for {missing}. Run scripts/20_train_lstm.py "
            f"--target {' '.join(missing)} first."
        )

    panel = pd.read_csv(PATHS.model_table(args.direction))
    predicted = predict_panel(panel, args.lookback, hidden_size=args.hidden_size,
                              num_layers=args.num_layers)

    out_path = args.out or (PATHS.processed / f"pred_data_{args.direction}.csv")
    predicted.to_csv(out_path, index=False)
    print(f"Predictions -> {out_path}  ({len(predicted):,} rows)")


if __name__ == "__main__":
    main()
