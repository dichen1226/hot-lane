"""Derived traffic-flow quantities: density and fundamental-diagram parameters."""
from __future__ import annotations

import numpy as np
import pandas as pd

#: 15-minute counts scaled to an hourly rate.
INTERVALS_PER_HOUR = 4


def add_density(
    df: pd.DataFrame, flow_col: str = "flow_HOT", speed_col: str = "speed_HOT"
) -> pd.DataFrame:
    """Add hourly flow (vph) and density (veh/mile) from a 15-minute count.

    Density follows the identity ``k = q / v``.
    """
    out = df.copy()
    hourly = f"{flow_col}_hr"
    out[hourly] = out[flow_col] * INTERVALS_PER_HOUR
    out[f"density_{flow_col.split('_')[-1]}"] = out[hourly] / out[speed_col]
    return out


def fit_line_through_extremes(
    df: pd.DataFrame,
    density_col: str,
    flow_col: str,
    density_begin: float,
    density_end: float,
) -> float:
    """Slope of the line joining the max-flow points at two density levels.

    Used to read the free-flow speed ``u_f`` off the uncongested branch of the
    fundamental diagram (low densities) and the backward wave speed ``w`` off
    the congested branch (high densities).
    """
    flow_begin = df.loc[df[density_col] == density_begin, flow_col].max()
    flow_end = df.loc[df[density_col] == density_end, flow_col].max()
    return (flow_end - flow_begin) / (density_end - density_begin)


def fundamental_diagram_parameters(
    df: pd.DataFrame,
    density_col: str = "density_HOT",
    flow_col: str = "flow_HOT_hr",
    free_flow_range: tuple[float, float] = (1, 20),
    congested_range: tuple[float, float] = (40, 64),
) -> dict[str, float]:
    """Estimate ``u_f`` (free-flow speed) and ``w`` (wave speed), both in mph."""
    return {
        "u_f": fit_line_through_extremes(df, density_col, flow_col, *free_flow_range),
        "w": fit_line_through_extremes(df, density_col, flow_col, *congested_range),
    }


def trip_travel_times(trips: pd.DataFrame) -> pd.DataFrame:
    """Add clock parts and a travel time (seconds) to a trip-level table.

    Trips are stamped by their **exit** time, which is what the toll they paid
    corresponds to.
    """
    out = trips.copy()
    start = pd.to_datetime(out["Start_Time"])
    end = pd.to_datetime(out["End_Time"])

    out["hour"] = end.dt.hour
    out["minute"] = end.dt.minute
    out["second"] = end.dt.second

    def to_seconds(ts: pd.Series) -> pd.Series:
        return ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second

    out["Start_Time_sec"] = to_seconds(start)
    out["End_Time_sec"] = to_seconds(end)
    out["travel_time_sec"] = out["End_Time_sec"] - out["Start_Time_sec"]
    return out


def round_minute_to_interval(df: pd.DataFrame, interval: int = 15) -> pd.DataFrame:
    """Snap ``minute`` up to the end of its 15-minute bin, rolling the hour over.

    Lets trip-level records join the interval-level panel.
    """
    out = df.copy()
    edges = np.arange(interval, 60 + interval, interval)
    for upper in edges:
        lower = upper - interval
        mask = (out["minute"] > lower) & (out["minute"] <= upper)
        out.loc[mask, "minute"] = upper
    # minute == 0 belongs to the first bin of its own hour
    out.loc[out["minute"] == 0, "minute"] = interval

    rollover = out["minute"] == 60
    out.loc[rollover, "hour"] = out.loc[rollover, "hour"] + 1
    out.loc[rollover, "minute"] = 0
    return out
