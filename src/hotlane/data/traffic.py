"""Loop-detector traffic data: per-lane speeds and volumes at each gantry."""
from __future__ import annotations

from datetime import time
from pathlib import Path

import pandas as pd

from hotlane.config import MONTH, PATHS, YEAR, gantry_ids

#: Rows 20..79 of a gantry sheet are the 05:15-20:00 window.
_WINDOW = slice(20, 80)

#: Lane 1 is the HOT lane westbound; lanes 1-2 are HOT eastbound.
_WB_GP_SPEED = ["Spd_Ln2", "Spd_Ln3", "Spd_Ln4", "Spd_Ln5", "Spd_Ln6"]
_WB_GP_VOLUME = ["Vol_Ln2", "Vol_Ln3", "Vol_Ln4", "Vol_Ln5", "Vol_Ln6"]
_EB_GP_SPEED = ["Spd_Ln3", "Spd_Ln4", "Spd_Ln5", "Spd_Ln6", "Spd_Ln7"]
_EB_GP_VOLUME = ["Vol_Ln3", "Vol_Ln4", "Vol_Ln5", "Vol_Ln6", "Vol_Ln7"]


def _spread(frame: pd.DataFrame, source: pd.DataFrame, cols: list[str], prefix: str) -> None:
    """Copy the lanes that exist into ``prefix1..prefixN`` plus a ``_avg`` column."""
    present = [c for c in cols if c in source.columns]
    for i, col in enumerate(present, start=1):
        frame[f"{prefix}{i}"] = source[col]
    frame[f"{prefix}_avg"] = source[present].mean(axis=1, skipna=True)


def traffic_volume_and_speed(
    date: str, traffic_data: pd.DataFrame, direction: str = "WB"
) -> pd.DataFrame:
    """Extract HOT / general-purpose speeds and volumes for one gantry-day.

    Args:
        date: as formatted in the workbook, e.g. ``'10/1/2018'``.
        traffic_data: one gantry's sheet.
        direction: ``'WB'`` (one HOT lane) or ``'EB'`` (two HOT lanes).
    """
    src = traffic_data[traffic_data["Date"] == date].iloc[_WINDOW].reset_index(drop=True)
    out = src[["Time_Ending", "Plaza_Id", "Date"]].copy()

    if direction == "WB":
        out["speed_HOT"] = src["Spd_Ln1"]
        _spread(out, src, _WB_GP_SPEED, "speed_GP")
        out["volume_HOT"] = src["Vol_Ln1"]
        _spread(out, src, _WB_GP_VOLUME, "volume_GP")
    else:
        out["speed_HOT1"] = src["Spd_Ln1"]
        out["speed_HOT2"] = src["Spd_Ln2"]
        out["speed_HOT_avg"] = src[["Spd_Ln1", "Spd_Ln2"]].mean(axis=1)
        _spread(out, src, _EB_GP_SPEED, "speed_GP")
        out["volume_HOT1"] = src["Vol_Ln1"]
        out["volume_HOT2"] = src["Vol_Ln2"]
        out["volume_HOT_avg"] = src[["Vol_Ln1", "Vol_Ln2"]].mean(axis=1)
        _spread(out, src, _EB_GP_VOLUME, "volume_GP")

    # Monday == 0 ... Sunday == 6
    out["weekday"] = pd.to_datetime(out["Date"]).dt.weekday
    return out


def round_to_hour(t: time) -> time:
    """Snap ``HH:59`` timestamps up to ``HH+1:00`` so they join with toll data."""
    if t.minute == 59:
        return t.replace(second=0, microsecond=0, minute=0, hour=t.hour + 1)
    return t


def traffic_for_day(
    day: int, direction: str = "WB", workbook: Path | None = None
) -> pd.DataFrame:
    """Stack every gantry's traffic record for one day, ready to merge."""
    xls = pd.ExcelFile(workbook or PATHS.traffic_workbook(direction))
    date = f"{MONTH}/{day}/{YEAR}"

    frames = [
        traffic_volume_and_speed(date, pd.read_excel(xls, gantry), direction)
        for gantry in gantry_ids(direction)
    ]
    out = pd.concat(frames).reset_index(drop=True)
    out["Time_Ending"] = [round_to_hour(t) for t in out["Time_Ending"]]
    return out.rename(columns={"Time_Ending": "Timestamp", "Plaza_Id": "gantry"})
