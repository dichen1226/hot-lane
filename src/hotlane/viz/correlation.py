"""Correlation figures: across gantries, between lanes, and against tolls."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hotlane.config import N_GANTRIES

DEFAULT_METHOD = "spearman"


def variable_by_gantry(
    data: pd.DataFrame, variable: str, n_gantries: int | None = None
) -> pd.DataFrame:
    """Reshape one variable into a (time x gantry) frame for correlation."""
    n_gantries = n_gantries or N_GANTRIES["WB"]
    return pd.DataFrame(
        {
            str(g): data.loc[data["gantry"] == g, variable].reset_index(drop=True)
            for g in range(1, n_gantries + 1)
        }
    )


def plot_gantry_correlations(
    data: pd.DataFrame,
    variables: list[str],
    save_path: Path,
    method: str = DEFAULT_METHOD,
    shared_scale: bool = False,
) -> None:
    """Grid of gantry-to-gantry correlation matrices, one panel per variable.

    ``shared_scale`` pins every panel to [0, 1] with a single colourbar so the
    variables can be compared directly.
    """
    n_cols = 2
    n_rows = (len(variables) + 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(9, 16 if shared_scale else 12))
    axs = axs.flatten()

    image = None
    for i, variable in enumerate(variables):
        frame = variable_by_gantry(data, variable)
        kwargs = {"vmin": 0, "vmax": 1} if shared_scale else {}
        image = axs[i].matshow(frame.corr(method=method), cmap="viridis", **kwargs)

        ticks = range(frame.shape[1])
        axs[i].set_xticks(ticks)
        axs[i].set_yticks(ticks)
        axs[i].set_xticklabels(frame.columns, fontsize=10)
        axs[i].set_yticklabels(frame.columns, fontsize=10)
        axs[i].set_title(f"Variable: {variable}")

        if not shared_scale:
            fig.colorbar(image, ax=axs[i]).ax.tick_params(labelsize=12)

    if len(variables) % n_cols != 0:
        fig.delaxes(axs[-1])

    plt.tight_layout()

    if shared_scale and image is not None:
        # Added after tight_layout so the manual placement is not overridden.
        cax = fig.add_axes([0.55, 0.04, 0.04, 0.25])
        fig.colorbar(image, cax=cax).ax.tick_params(labelsize=10)

    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_pair_correlation(
    data: pd.DataFrame, columns: list[str], save_path: Path, method: str = DEFAULT_METHOD
) -> pd.DataFrame:
    """Small correlation matrix for a pair of columns (e.g. HOT vs GP speed)."""
    pair = data[columns]
    corr = pair.corr(method=method)

    fig = plt.figure(figsize=(4, 4))
    plt.matshow(corr, fignum=fig.number)
    plt.xticks(range(len(columns)), columns, fontsize=10)
    plt.yticks(range(len(columns)), columns, fontsize=10)
    plt.colorbar().ax.tick_params(labelsize=14)
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    return corr


def hourly_correlation(
    data: pd.DataFrame,
    hour: int,
    gantries: list[int],
    method: str = DEFAULT_METHOD,
) -> float:
    """Correlate the hour's average toll rate with its total SOV entry volume.

    The hour is taken as ``HH:15``-``HH+1:00``, matching the interval-ending
    convention of the panel.
    """
    block = data[data["gantry"].isin(gantries)]
    block = block[(block["hour"] == hour) | ((block["hour"] == hour + 1) & (block["minute"] == 0))]

    volume = block[["day", "hour", "minute", "num_O"]].groupby(["day", "hour", "minute"]).sum()
    toll = block[["day", "hour", "minute", "toll_rate"]].groupby(["day", "hour", "minute"]).mean()

    merged = pd.merge(toll.reset_index(), volume.reset_index(), on=["day", "hour", "minute"])
    return merged[["toll_rate", "num_O"]].corr(method=method).iloc[0, 1]


def hourly_travel_time_toll_correlation(
    trips: pd.DataFrame, hour: int, method: str = DEFAULT_METHOD
) -> float:
    """Correlate what a trip paid with how long it took, within one hour."""
    block = trips[trips["hour"] == hour]
    return block[["Payment", "travel_time_sec"]].corr(method=method).iloc[0, 1]


def plot_correlation_profile(
    hours: np.ndarray,
    series: dict[str, list[float]],
    title: str,
    save_path: Path,
    colors: dict[str, str] | None = None,
) -> None:
    """Line chart of a correlation measure against hour of day."""
    colors = colors or {}
    plt.figure(figsize=(6, 4))
    for label, values in series.items():
        plt.plot(hours, values, marker="o", label=label, color=colors.get(label))
    plt.xlabel("Hour")
    plt.ylabel("Correlation")
    plt.title(title)
    if len(series) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
