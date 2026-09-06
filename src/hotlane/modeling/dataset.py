"""Turn the gantry panel into LSTM sequences, and persist the feature scaler."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

#: Features fed to the model in addition to the target speed series.
BASIC_FEATURES = ["day", "weekday", "hour", "minute", "gantry", "toll_rate"]

#: Number of 15-minute steps of history the model sees.
DEFAULT_LOOKBACK = 4


def select_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Subset the panel to the model's input columns, target last."""
    return df[BASIC_FEATURES + [target_col]]


def fit_scaler(df: pd.DataFrame) -> RobustScaler:
    """Fit a RobustScaler on the full feature frame.

    Note that this is fitted before the train/test split, matching the original
    experiments; see README, "Known issues".
    """
    return RobustScaler().fit(df)


def save_scaler(scaler: RobustScaler, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(scaler, fh)


def load_scaler(path: Path) -> RobustScaler:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def make_sequences(
    df: pd.DataFrame,
    scaler: RobustScaler,
    target_col: str,
    lookback: int = DEFAULT_LOOKBACK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cut ``(lookback, n_features)`` windows per gantry with a one-step target.

    Sequences never straddle two gantries: each detector's series is windowed
    independently and the windows are then pooled.

    Returns:
        ``X`` of shape ``(n_windows, lookback, n_features)`` (scaled) and ``y``
        of shape ``(n_windows, 1)`` in the target's original units (mph).
    """
    inputs, targets = [], []
    for gantry in df["gantry"].unique():
        block = df[df["gantry"] == gantry]
        scaled = scaler.transform(block)
        for i in range(block.shape[0] - lookback - 1):
            inputs.append(scaled[i : i + lookback, :])
            targets.append([block[target_col].iloc[i + lookback]])

    return (
        torch.tensor(np.array(inputs)).float(),
        torch.tensor(np.array(targets)).float(),
    )


def make_prediction_windows(
    df: pd.DataFrame,
    scaler: RobustScaler,
    lookback: int = DEFAULT_LOOKBACK,
) -> torch.Tensor:
    """Like :func:`make_sequences` but for inference: every window, no targets."""
    inputs = []
    for gantry in df["gantry"].unique():
        block = df[df["gantry"] == gantry]
        scaled = scaler.transform(block)
        for i in range(block.shape[0] - lookback + 1):
            inputs.append(scaled[i : i + lookback, :])
    return torch.tensor(np.array(inputs)).float()


def chronological_split(df: pd.DataFrame, train_fraction: float = 0.67):
    """Split a panel in time order (no shuffling across the train/test cut)."""
    cut = int(len(df) * train_fraction)
    return df[:cut], df[cut:]
