"""Aggregate trip origins/destinations into 15-minute gantry entry/exit counts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from hotlane.config import (
    DAY_START,
    DIRECTION_CODES,
    INTERVAL_MINUTES,
    INTERVALS_PER_DAY,
    N_GANTRIES,
    PAYMENT_CLASSES,
    interval_end_times,
)

#: Trips ceiling to exactly 05:00 fall outside the analysis window and are
#: dropped (the first retained interval ends at 05:15).
_EXCLUDED_CEIL = pd.Timestamp("1900-01-01 05:00:00").time()


def od_flow_matrices(
    trip_od: pd.DataFrame,
    time_col: str,
    od_col: str,
    payment_class: str = "SOV",
    *,
    impute_missing_destination: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Count trips per (15-min interval, gantry) for one vehicle class.

    Args:
        trip_od: output of :func:`hotlane.data.transactions.build_trip_od`.
        time_col: ``'Start_Time'`` (pair with ``od_col='Origin'``) or
            ``'End_Time'`` (pair with ``od_col='Destination'``).
        od_col: ``'Origin'`` or ``'Destination'``.
        payment_class: key of :data:`hotlane.config.PAYMENT_CLASSES`.
        impute_missing_destination: single-gantry trips have no destination.
            When True (the behaviour of the final pipeline run) they are given
            destination ``origin + 1`` at their start time; when False they are
            skipped entirely.

    Returns:
        ``(westbound, eastbound)`` matrices of shape
        ``(INTERVALS_PER_DAY, n_gantries)``; rows are interval index, columns
        are gantry index (0-based).
    """
    payment_values = PAYMENT_CLASSES[payment_class]
    times = pd.DataFrame(interval_end_times(), columns=["time_inv"])

    west = np.zeros((INTERVALS_PER_DAY, N_GANTRIES["WB"]))
    east = np.zeros((INTERVALS_PER_DAY, N_GANTRIES["EB"]))

    trips = trip_od.copy()
    if impute_missing_destination:
        trips["Destination"] = trips["Destination"].fillna(trips["Origin"] + 1)
        trips["End_Time"] = trips["End_Time"].fillna(trips["Start_Time"])
        # A trip imputed past the last gantry is clipped to it.
        wb = trips["direction"].isin(DIRECTION_CODES["WB"])
        eb = trips["direction"].isin(DIRECTION_CODES["EB"])
        trips.loc[wb & (trips["Destination"] > N_GANTRIES["WB"]), "Destination"] = N_GANTRIES["WB"]
        trips.loc[eb & (trips["Destination"] > N_GANTRIES["EB"]), "Destination"] = N_GANTRIES["EB"]

    time_lookup = {t: i for i, t in enumerate(times["time_inv"])}

    for _, row in trips.iterrows():
        gantry = row[od_col]
        if pd.isna(gantry):
            continue
        if row["Payment Type"] not in payment_values:
            continue

        stamp = pd.Timestamp(row[time_col]).ceil(f"{INTERVAL_MINUTES}min").time()
        if stamp == _EXCLUDED_CEIL or stamp not in time_lookup:
            continue

        t = time_lookup[stamp]
        g = int(gantry) - 1
        if row["direction"] in DIRECTION_CODES["WB"] and g < N_GANTRIES["WB"]:
            west[t, g] += 1
        elif row["direction"] in DIRECTION_CODES["EB"] and g < N_GANTRIES["EB"]:
            east[t, g] += 1

    return west, east


def stack_matrix(matrix: np.ndarray, direction: str = "WB") -> pd.DataFrame:
    """Flatten an (interval x gantry) matrix into a long frame.

    Columns: ``time`` (interval index), ``gantry`` (e.g. ``'W01'``),
    ``volume``, ``Timestamp`` (interval-ending clock time).
    """
    df = pd.DataFrame(matrix).stack().reset_index(drop=False)
    df = df.rename(columns={"level_0": "time", "level_1": "gantry", 0: "volume"})

    start = pd.Timestamp.combine(pd.Timestamp("1900-01-01"), DAY_START)
    interval = pd.Timedelta(minutes=INTERVAL_MINUTES)
    df["Timestamp"] = [(start + i * interval).time() for i in df["time"]]

    prefix = "W" if direction == "WB" else "E"
    df["gantry"] = prefix + (df["gantry"] + 1).astype(str).str.zfill(2)
    return df
