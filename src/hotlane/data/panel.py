"""Build the gantry x 15-minute panel that everything downstream consumes."""
from __future__ import annotations

import numpy as np
import pandas as pd

from hotlane.config import INTERVALS_PER_DAY, N_GANTRIES, STUDY_DAYS

#: Column order of a daily panel once toll and traffic data are joined.
DAILY_PANEL_COLUMNS = [
    "time_index", "day", "weekday", "hour", "minute", "gantry",
    "volume_o", "volume_d", "toll_rate",
    "speed_HOT", "speed_GP1", "speed_GP2", "speed_GP3", "speed_GP4", "speed_GP5",
    "speed_GP_avg",
    "volume_HOT", "volume_GP1", "volume_GP2", "volume_GP3", "volume_GP4", "volume_GP5",
    "volume_GP_avg",
]

#: Rename applied when the panel is handed to the models, so that column names
#: describe quantities rather than their source table.
MODEL_COLUMN_RENAME = {
    "volume_o": "num_O",
    "volume_d": "num_D",
    "volume_HOT": "flow_HOT",
    "volume_GP_avg": "flow_GP",
    "speed_GP_avg": "speed_GP",
}

MODEL_TABLE_COLUMNS = [
    "time_index2", "day", "weekday", "hour", "minute", "gantry",
    "num_O", "num_D", "flow_HOT", "flow_GP", "toll_rate", "speed_HOT", "speed_GP",
]


def merge_daily_sources(
    origin_flows: pd.DataFrame,
    destination_flows: pd.DataFrame,
    toll_rates: pd.DataFrame | None = None,
    traffic: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join one day's O/D counts with posted tolls and detector traffic.

    ``toll_rates`` and ``traffic`` may be omitted for the non-SOV vehicle
    classes, where only the O/D counts are needed (they are merged onto the SOV
    panel later by :func:`merge_vehicle_classes`).
    """
    df = pd.merge(origin_flows, destination_flows, on=["gantry", "Timestamp"])
    df = df.rename(
        columns={
            "volume_x": "volume_o",
            "volume_y": "volume_d",
            "time_x": "time_index",
            "day_x": "day",
        }
    )
    df = df.drop(columns=[c for c in ("time_y", "day_y") if c in df.columns])

    if toll_rates is not None:
        df = pd.merge(df, toll_rates, on=["gantry", "Timestamp"])
    if traffic is not None:
        df = pd.merge(df, traffic, on=["gantry", "Timestamp"])

    dropped = ("index_x", "index_y", "index", "Section", "Date")
    df = df.drop(columns=[c for c in dropped if c in df.columns])
    df = df.rename(columns={"Generated Toll/ Message": "toll_rate"})

    # 'W07' -> 7
    df["gantry"] = df["gantry"].str.extract(r"(\d+)").astype(int)

    timestamps = pd.to_datetime(df["Timestamp"].astype(str))
    df["hour"] = timestamps.dt.hour
    df["minute"] = timestamps.dt.minute
    df = df.drop(columns=["Timestamp"])

    # Detector volumes are lane averages and are reported as integers.
    for column in [c for c in df.columns if "volume_GP" in c]:
        df[column] = df[column].round()

    ordered = [c for c in DAILY_PANEL_COLUMNS if c in df.columns]
    return df[ordered + [c for c in df.columns if c not in ordered]]


def assign_month_time_index(df: pd.DataFrame, days: list[int] | None = None) -> pd.DataFrame:
    """Number every observed (day, hour, minute) slot consecutively.

    The result is ``time_index2``: a single monotone clock across the whole
    study month, which is what the LSTM sequences are cut along. Slots with no
    data at all are skipped rather than numbered, so the index stays gap-free.
    """
    days = days or STUDY_DAYS
    out = df.sort_values(by=["day", "hour", "minute"]).copy()

    slots = (
        out[["day", "hour", "minute"]]
        .drop_duplicates()
        .sort_values(["day", "hour", "minute"])
        .reset_index(drop=True)
    )
    slots["time_index2"] = np.arange(len(slots))
    return out.merge(slots, on=["day", "hour", "minute"], how="left")


def fill_missing_gantries(df: pd.DataFrame, n_gantries: int | None = None) -> pd.DataFrame:
    """Pad each time slot so that every gantry has a row.

    Detector outages leave holes in the panel; the LSTM input builder assumes a
    rectangular (time x gantry) grid. Padded rows carry only ``time_index2``
    and ``gantry``; every measurement is NaN.
    """
    n_gantries = n_gantries or N_GANTRIES["WB"]
    counts = df.groupby("time_index2")["gantry"].nunique()
    incomplete = counts[counts < n_gantries].index
    if len(incomplete) == 0:
        return df

    padding = []
    for time_index in incomplete:
        present = set(df.loc[df["time_index2"] == time_index, "gantry"].unique())
        missing = [g for g in range(1, n_gantries + 1) if g not in present]
        padding.append(
            pd.DataFrame(
                {
                    "time_index2": time_index,
                    "gantry": missing,
                    "time_index": np.nan,
                    "day": np.nan,
                }
            )
        )
    return pd.concat([df, *padding], ignore_index=True)


def merge_vehicle_classes(
    sov: pd.DataFrame,
    hov2: pd.DataFrame,
    hov3: pd.DataFrame,
    violations: pd.DataFrame,
) -> pd.DataFrame:
    """Attach HOV2 / HOV3+ / violation entry-exit counts to the SOV panel.

    Adds ``total_o`` / ``total_d`` / ``total_veh`` so the class split can be
    sanity-checked against the detector volumes.
    """
    out = sov.rename(columns={"volume_o": "num_O", "volume_d": "num_D"}).copy()

    for other, suffix in ((hov2, "HOV2"), (hov3, "HOV3"), (violations, "VIO")):
        cols = ["time_index2", "gantry", "volume_o", "volume_d"]
        block = other[cols].rename(
            columns={"volume_o": f"num_o_{suffix}", "volume_d": f"num_d_{suffix}"}
        )
        out = pd.merge(out, block, how="left", on=["time_index2", "gantry"])

    out["total_o"] = out[["num_O", "num_o_HOV2", "num_o_HOV3", "num_o_VIO"]].sum(axis=1)
    out["total_d"] = out[["num_D", "num_d_HOV2", "num_d_HOV3", "num_d_VIO"]].sum(axis=1)
    out["total_veh"] = out["total_o"] + out["total_d"]
    return out


def to_model_table(df: pd.DataFrame) -> pd.DataFrame:
    """Rename and subset a panel to the canonical LSTM training schema."""
    out = df.rename(columns=MODEL_COLUMN_RENAME)
    missing = [c for c in MODEL_TABLE_COLUMNS if c not in out.columns]
    if missing:
        raise KeyError(f"Panel is missing model columns: {missing}")
    return out[MODEL_TABLE_COLUMNS].sort_values(["time_index2", "gantry"]).reset_index(drop=True)


def audit_panel(df: pd.DataFrame, n_gantries: int | None = None) -> pd.DataFrame:
    """Report which gantries are absent from each incomplete time slot."""
    n_gantries = n_gantries or N_GANTRIES["WB"]
    all_gantries = set(range(1, n_gantries + 1))
    counts = df.groupby("time_index2")["gantry"].nunique()

    rows = []
    for time_index in counts[counts < n_gantries].index:
        present = set(df.loc[df["time_index2"] == time_index, "gantry"].unique())
        rows.append([time_index, len(present), sorted(all_gantries - present)])
    return pd.DataFrame(rows, columns=["time_index2", "num_gantries", "missing_gantries"])


def expected_slot_count(days: list[int] | None = None) -> int:
    """How many (day, interval) slots a complete month should contain."""
    return len(days or STUDY_DAYS) * INTERVALS_PER_DAY
