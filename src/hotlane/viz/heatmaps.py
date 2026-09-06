"""Time-space speed heatmaps (gantry on the y-axis, time of day on the x-axis)."""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from hotlane.config import DAY_START, INTERVAL_MINUTES


def add_clock_time(data: pd.DataFrame) -> pd.DataFrame:
    """Add a ``time`` (HH:MM) column derived from the month-wide time index.

    ``time_index2`` counts intervals across the whole month; this maps it back
    onto the repeating within-day clock so days can be overlaid or averaged.
    """
    out = data.copy()
    intervals_per_day = int(out.loc[out["day"] == out["day"].min(), "time_index2"].max()) + 1
    out["time_index_rep"] = out["time_index2"] - (out["day"] - 1) * intervals_per_day

    start = datetime.combine(datetime(1900, 1, 1), DAY_START)
    labels = [
        (start + timedelta(minutes=INTERVAL_MINUTES * i)).strftime("%H:%M")
        for i in range(intervals_per_day)
    ]
    lookup = pd.DataFrame(
        {"time_index_rep": np.arange(intervals_per_day), "time": labels}
    )
    return out.merge(lookup, on="time_index_rep", how="left")


def plot_speed_heatmap(
    data: pd.DataFrame,
    variable: str,
    save_path: Path | None = None,
    *,
    index: str = "gantry",
    columns: str = "time",
    vmin: float = 30,
    vmax: float = 90,
    title: str | None = None,
) -> None:
    """Heatmap of one speed variable over the time-space plane."""
    pivot = data.pivot(index=index, columns=columns, values=variable)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title(title or variable, fontsize=16)
    sns.heatmap(
        pivot,
        ax=ax,
        cmap="RdYlGn",
        cbar_kws={"label": "Speed (mph)"},
        vmin=vmin,
        vmax=vmax,
    )
    ax.invert_yaxis()
    ax.set_ylabel("Gantry")
    ax.set_xlabel("Time")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=90, ha="right")

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)
