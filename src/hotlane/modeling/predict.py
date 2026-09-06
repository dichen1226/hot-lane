"""Rolling one-step-ahead speed prediction with the trained LSTMs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hotlane.config import PATHS
from hotlane.modeling.dataset import (
    DEFAULT_LOOKBACK,
    load_scaler,
    make_prediction_windows,
    select_features,
)
from hotlane.modeling.model import HOTSpeedLSTM, get_device

#: One model per lane type; both are applied to produce the prediction table.
TARGET_COLUMNS = ("speed_HOT", "speed_GP")

#: Columns that identify a panel row, used when rejoining predictions.
PANEL_KEYS = ["time_index2", "day", "hour", "minute", "gantry"]


def merge_observed_and_predicted(
    observed: pd.DataFrame, predicted: pd.DataFrame
) -> pd.DataFrame:
    """Join a prediction table back onto the observed panel.

    :func:`predict_panel` overwrites the speed columns in place, so the observed
    values have to come from the original panel. The result carries both:
    ``speed_HOT`` / ``speed_GP`` (observed) and ``preds_HOT`` / ``preds_GP``.
    """
    merged = observed.merge(
        predicted[PANEL_KEYS + list(TARGET_COLUMNS)], on=PANEL_KEYS, suffixes=("", "_pred")
    )
    return merged.rename(
        columns={f"{t}_pred": f"preds_{t.split('_')[-1]}" for t in TARGET_COLUMNS}
    )


def model_path(target_col: str, models_dir: Path | None = None) -> Path:
    return (models_dir or PATHS.models) / f"trained_model_{target_col}.pth"


def scaler_path(target_col: str, models_dir: Path | None = None) -> Path:
    return (models_dir or PATHS.models) / f"scaler_{target_col}.pkl"


def load_trained_model(
    target_col: str,
    input_size: int,
    hidden_size: int = 32,
    num_layers: int = 1,
    models_dir: Path | None = None,
) -> HOTSpeedLSTM:
    """Instantiate the architecture and load its saved weights.

    ``hidden_size`` / ``num_layers`` must match whatever was used at training
    time; the checkpoint stores weights only.
    """
    device = get_device()
    model = HOTSpeedLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_output=1,
    ).to(device)
    model.load_state_dict(torch.load(model_path(target_col, models_dir), map_location=device))
    model.eval()
    return model


class SpeedPredictor:
    """Both lane models plus their scalers, loaded once and reused.

    Checkpoints are read on construction rather than per window; rolling the
    month otherwise reloads them a few thousand times.
    """

    def __init__(
        self,
        lookback: int = DEFAULT_LOOKBACK,
        models_dir: Path | None = None,
        hidden_size: int = 32,
        num_layers: int = 1,
        n_features: int | None = None,
    ) -> None:
        from hotlane.modeling.dataset import BASIC_FEATURES

        self.lookback = lookback
        self.device = get_device()
        n_features = n_features or len(BASIC_FEATURES) + 1

        self.scalers = {t: load_scaler(scaler_path(t, models_dir)) for t in TARGET_COLUMNS}
        self.models = {
            t: load_trained_model(t, n_features, hidden_size, num_layers, models_dir)
            for t in TARGET_COLUMNS
        }

    def predict_window(self, window: pd.DataFrame) -> np.ndarray:
        """Predict HOT and GP speed for every gantry in one ``lookback`` window.

        Returns an array of shape ``(n_gantries, 2)``, columns ordered as
        :data:`TARGET_COLUMNS`.
        """
        per_target = []
        for target_col in TARGET_COLUMNS:
            features = select_features(window, target_col)
            inputs = make_prediction_windows(
                features, self.scalers[target_col], self.lookback
            ).to(self.device)
            with torch.no_grad():
                per_target.append(self.models[target_col](inputs).cpu().numpy())
        return np.concatenate(per_target, axis=1)


def predict_panel(
    panel: pd.DataFrame,
    lookback: int = DEFAULT_LOOKBACK,
    models_dir: Path | None = None,
    hidden_size: int = 32,
    num_layers: int = 1,
    progress_every: int = 100,
) -> pd.DataFrame:
    """Roll the models across the whole panel, one 15-minute step at a time.

    The first ``lookback - 1`` time slots have no history and keep their
    observed speeds; every later slot's ``speed_HOT`` / ``speed_GP`` is
    replaced by the model's prediction.
    """
    n_gantries = panel["gantry"].nunique()
    last_index = int(panel["time_index2"].max())
    predictor = SpeedPredictor(lookback, models_dir, hidden_size, num_layers)

    blocks = []
    for step, index in enumerate(range(lookback - 1, last_index + 1)):
        window_indices = range(index - lookback + 1, index + 1)
        window = panel[panel["time_index2"].isin(window_indices)]
        blocks.append(predictor.predict_window(window))
        if progress_every and step % progress_every == 0:
            print(f"  time index {index} / {last_index}")

    predictions = np.vstack(blocks)
    out = panel.copy()
    # Detector speeds can arrive as integers; predictions are not.
    out[list(TARGET_COLUMNS)] = out[list(TARGET_COLUMNS)].astype(float)
    out.loc[n_gantries * (lookback - 1) :, list(TARGET_COLUMNS)] = predictions
    return out
