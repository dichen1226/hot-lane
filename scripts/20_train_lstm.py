"""Step 20 - Train the LSTM speed predictor.

Trains one model per target lane. Run it twice (or pass both targets) to get
the pair that step 21 needs.

    python scripts/20_train_lstm.py --target speed_HOT speed_GP
    python scripts/20_train_lstm.py --target speed_HOT --epochs 500 --hidden-size 256
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch

from hotlane.config import PATHS
from hotlane.modeling.dataset import (
    DEFAULT_LOOKBACK,
    chronological_split,
    fit_scaler,
    make_sequences,
    save_scaler,
    select_features,
)
from hotlane.modeling.predict import model_path, scaler_path
from hotlane.modeling.train import TrainConfig, train_model
from hotlane.viz.timeseries import plot_rmse_curves


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target", nargs="*", default=["speed_HOT"], choices=["speed_HOT", "speed_GP"]
    )
    parser.add_argument("--direction", default="WB", choices=["WB", "EB"])
    parser.add_argument("--lookback", type=int, default=DEFAULT_LOOKBACK)
    parser.add_argument("--train-fraction", type=float, default=0.67)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    panel = pd.read_csv(PATHS.model_table(args.direction))

    for target_col in args.target:
        print(f"\n=== {target_col} ===")
        features = select_features(panel, target_col)

        # NOTE: the scaler is fitted on the full series, matching the original
        # experiments. See README, "Known issues".
        scaler = fit_scaler(features)
        save_scaler(scaler, scaler_path(target_col))

        train, test = chronological_split(features, args.train_fraction)
        X_train, y_train = make_sequences(train, scaler, target_col, args.lookback)
        X_test, y_test = make_sequences(test, scaler, target_col, args.lookback)
        print(f"train windows: {len(X_train):,}   test windows: {len(X_test):,}")

        model, history = train_model(
            X_train,
            y_train,
            X_test,
            y_test,
            TrainConfig(
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                n_epochs=args.epochs,
            ),
        )

        out_path = model_path(target_col)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path)
        plot_rmse_curves(history, PATHS.figure("training", f"rmse_{target_col}.jpg"))
        print(f"model  -> {out_path}")
        print(f"scaler -> {scaler_path(target_col)}")


if __name__ == "__main__":
    main()
