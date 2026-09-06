"""Figure layouts specific to the project report.

The figures published in the report are composites — grids over hours or over
gantries — that the exploratory scripts produced one panel at a time and were
then assembled by hand. These helpers build the composites directly.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hotlane.viz.timeseries import plot_prediction_vs_actual


def plot_hourly_day_grid(
    data: pd.DataFrame,
    value_col: str,
    save_path: Path,
    *,
    aggregate: str = "mean",
    ylabel: str = "",
    color: str = "blue",
    ylim: tuple[float, float] | None = None,
    gantry: int = 1,
    n_cols: int = 4,
) -> None:
    """One panel per hour of day; within a panel, the value against day.

    Each panel aggregates the hour's four 15-minute intervals (``mean`` for a
    rate such as the toll, ``sum`` for a count such as entries) and draws the
    across-day average as a dotted reference line.
    """
    block = data[data["gantry"] == gantry]
    hours = sorted(int(h) for h in block["hour"].unique())
    # The last hour of the window is only partially covered.
    hours = [h for h in hours if h < max(hours)]

    n_rows = int(np.ceil(len(hours) / n_cols))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.2 * n_rows))
    axs = np.atleast_1d(axs).flatten()

    for ax, hour in zip(axs, hours, strict=False):
        window = block[
            (block["hour"] == hour) | ((block["hour"] == hour + 1) & (block["minute"] == 15))
        ]
        grouped = getattr(window.groupby("day"), aggregate)(numeric_only=True).reset_index()

        ax.plot(grouped["day"], grouped[value_col], marker="o", color=color)
        ax.axhline(grouped[value_col].mean(), color="gray", linestyle="dotted")
        ax.set_xlabel("Day")
        ax.set_ylabel(ylabel or value_col)
        ax.set_title(f"{hour}:00 - {hour}:59")
        if ylim:
            ax.set_ylim(*ylim)

    for ax in axs[len(hours) :]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_prediction_grid(
    data: pd.DataFrame,
    gantries: list[int],
    save_path: Path,
    *,
    n_cols: int = 4,
    ylim: tuple[float, float] = (35, 85),
) -> None:
    """Prediction-vs-observed panels laid out as (gantry, HOT) then (gantry, GP).

    With the default four columns each row holds two gantries, which is the
    layout used in the report.
    """
    panels = [(g, lane) for g in gantries for lane in ("HOT", "GP")]
    n_rows = int(np.ceil(len(panels) / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 3.0 * n_rows))
    axs = np.atleast_1d(axs).flatten()

    for ax, (gantry, lane) in zip(axs, panels, strict=False):
        plot_prediction_vs_actual(
            ax, data, gantry, f"preds_{lane}", f"speed_{lane}", f"{lane} lane", ylim=ylim
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    for ax in axs[len(panels) :]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_loss_curves(history, save_paths: dict[str, Path], metric: str = "mse") -> None:
    """Training and validation loss against epoch, one figure per target.

    ``history`` maps target column -> :class:`hotlane.modeling.train.TrainHistory`.
    """
    label = metric.upper()
    for target_col, path in save_paths.items():
        h = history[target_col]
        train = getattr(h, f"train_{metric}")
        valid = getattr(h, f"test_{metric}")

        plt.figure(figsize=(6, 4))
        plt.plot(h.epochs, valid, color="tab:blue", label=f"Valid {label}")
        plt.plot(h.epochs, train, color="orange", label=f"Train {label}")
        plt.xlabel("Epochs")
        plt.ylabel(label)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=300)
        plt.close()
