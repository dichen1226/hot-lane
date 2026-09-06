"""Time-series figures: day-to-day profiles, diurnal profiles, predictions."""
from __future__ import annotations

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


def plot_daily_profile(
    data: pd.DataFrame,
    hour: int,
    gantry: int,
    save_dir: Path,
    *,
    toll_ylim: tuple[float, float] = (0.4, 2.3),
    volume_ylim: tuple[float, float] = (0, 200),
) -> None:
    """For one hour and gantry, plot toll rate and SOV entries against day.

    The dotted line is the across-day mean, so weekday-to-weekday variation is
    easy to read off.
    """
    block = data[data["gantry"] == gantry]
    block = block[(block["hour"] == hour) | ((block["hour"] == hour + 1) & (block["minute"] == 15))]

    title = f"{hour}:00 - {hour}:59"

    hourly_mean = block.groupby("day").mean(numeric_only=True).reset_index()
    plt.figure(figsize=(5, 4))
    plt.plot(hourly_mean["day"], hourly_mean["toll_rate"], marker="o", color="blue")
    plt.axhline(hourly_mean["toll_rate"].mean(), color="gray", linestyle="dotted")
    plt.xlabel("Day")
    plt.ylabel("Toll rate ($)")
    plt.ylim(*toll_ylim)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_dir / f"toll_rate_W{gantry}_hour_{hour}.jpg", dpi=300)
    plt.close()

    hourly_sum = block.groupby("day").sum(numeric_only=True).reset_index()
    plt.figure(figsize=(5, 4))
    plt.plot(hourly_sum["day"], hourly_sum["num_O"], marker="o", color="green")
    plt.axhline(hourly_sum["num_O"].mean(), color="gray", linestyle="dotted")
    plt.xlabel("Day")
    plt.ylabel("SOV volume")
    plt.ylim(*volume_ylim)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_dir / f"SOV_trips_W{gantry}_hour_{hour}.jpg", dpi=300)
    plt.close()


def plot_diurnal_profile(data: pd.DataFrame, gantry: int, save_path: Path) -> None:
    """Average toll rate and total SOV entries by hour, on twin y-axes."""
    block = data[data["gantry"] == gantry]
    block = block.groupby("time_index2").mean(numeric_only=True).reset_index()

    # Four consecutive 15-minute intervals make one hour.
    groups = (block["time_index2"] // 4).astype(int)
    hourly_mean = block.groupby(groups).mean(numeric_only=True).reset_index(drop=True)
    hourly_sum = block.groupby(groups).sum(numeric_only=True).reset_index(drop=True)
    hourly_mean["hour"] = hourly_mean["hour"].round()
    hourly_sum["hour"] = hourly_mean["hour"]

    fig, ax1 = plt.subplots(figsize=(5, 4))
    ax1.plot(
        hourly_mean["hour"], hourly_mean["toll_rate"], marker="o", color="blue", label="Toll rate"
    )
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Toll rate ($)", color="blue")

    ax2 = ax1.twinx()
    ax2.plot(hourly_sum["hour"], hourly_sum["num_O"], marker="o", color="green", label="SOV flow")
    ax2.set_ylabel("SOV flow", color="green")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2)
    plt.xlim(4, 20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def add_time_column(data: pd.DataFrame) -> pd.DataFrame:
    """Combine ``hour`` and ``minute`` into a plottable datetime column."""
    out = data.copy()
    out["time"] = pd.to_datetime(
        out["hour"].astype(str) + " " + out["minute"].astype(str), format="%H %M"
    )
    return out


def plot_prediction_vs_actual(
    ax,
    data: pd.DataFrame,
    gantry: int,
    pred_col: str,
    actual_col: str,
    title: str,
    ylim: tuple[float, float] = (25, 90),
) -> None:
    """Draw observed and predicted speed for one gantry onto an existing axis."""
    block = data[data["gantry"] == gantry].reset_index(drop=True)

    ax.plot(block["time"], block[actual_col], label="Actual", marker="o", markersize=2)
    ax.plot(
        block["time"], block[pred_col], label="Prediction", marker="o", markersize=2, color="red"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Speed (mph)")
    ax.set_title(f"Gantry {gantry} - {title}")
    ax.legend(loc="lower right")
    ax.set_ylim(*ylim)
    ax.grid(True)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))


def plot_prediction_figure(
    data: pd.DataFrame,
    gantry: int,
    pred_col: str,
    actual_col: str,
    save_path: Path,
    title: str = "",
) -> None:
    """Standalone prediction-vs-actual figure for a single gantry."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_prediction_vs_actual(ax, data, gantry, pred_col, actual_col, title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_speed_difference(
    grouped: pd.DataFrame,
    save_path: Path,
    columns: tuple[str, str] = ("speed_diff_HOT", "speed_diff_GP"),
    titles: tuple[str, str] = ("HOT Speed Difference", "GP Speed Difference"),
) -> None:
    """Stacked panels of predicted-minus-observed speed over the day."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 10))
    for ax, column, title in zip(axes, columns, titles, strict=True):
        ax.plot(grouped["time"], grouped[column], marker="o")
        ax.axhline(y=0, color="red", linestyle="-")
        ax.set_xlabel("Time", fontsize=14)
        ax.set_ylabel("Speed difference (mph)", fontsize=14)
        ax.tick_params(axis="x", rotation=90, labelsize=9)
        ax.set_title(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_rmse_curves(history, save_path: Path) -> None:
    """Train / test RMSE against epoch."""
    plt.figure()
    plt.plot(history.epochs, history.train_rmse, color="blue", label="Train")
    plt.plot(history.epochs, history.test_rmse, color="red", label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE (mph)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
