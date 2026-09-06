"""Reading and cleaning the raw HOT-lane toll transaction files."""
from __future__ import annotations

import glob
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from hotlane.config import PATHS

#: The vendor CSVs carry a 9-row banner before the real header row.
_HEADER_ROWS = 9

_ID_COLUMNS = ["Anonymized Trip Tag ID", "Anonymized Trip Plate ID"]


def clean_transaction_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Turn one raw transaction CSV into a tidy frame.

    Steps: strip the banner, promote the header row, drop the anonymised
    vehicle identifiers, parse ``Assigned Fare`` to a number, and decode the
    ``Lane`` string (e.g. ``W15E03-21``) into ``gantry`` and ``direction``.
    """
    data = df.tail(-_HEADER_ROWS).reset_index(drop=True)
    data.columns = data.iloc[0]
    data = data.tail(-1).reset_index(drop=True)
    data = data.drop(columns=_ID_COLUMNS)

    # "$1.25" -> 1.25
    data["Assigned Fare"] = pd.to_numeric(
        data["Assigned Fare"].str.replace("$", "", regex=False), errors="coerce"
    )

    data = data.reset_index(drop=False)

    # Common form: <WB letters><WB number><EB letters><EB number>-<direction>
    data[["WB", "WB_gantry", "EB", "EB_gantry", "direction"]] = data["Lane"].str.extract(
        r"(\D+)(\d+)(\D+)(\d+)-(\d+)"
    )
    data = data.drop(columns=["WB", "EB"])

    # Which of the two gantry numbers applies depends on the direction code.
    data["gantry"] = np.nan
    data.loc[data["direction"] == "21", "gantry"] = data["WB_gantry"]
    data.loc[data["direction"] == "01", "gantry"] = data["EB_gantry"]
    data.loc[data["direction"] == "02", "gantry"] = data["EB_gantry"]
    data = data.drop(columns=["WB_gantry", "EB_gantry"])

    # Short form, e.g. "W15-21".
    data[["bound", "num", "direction2"]] = data["Lane"].str.extract(r"(\D+)(\d+)-(\d+)")
    data.loc[data["direction"].isna(), "direction"] = data.loc[
        data["direction"].isna(), "direction2"
    ]
    data.loc[data["gantry"].isna(), "gantry"] = data.loc[data["gantry"].isna(), "num"]
    data = data.drop(columns=["bound", "num", "direction2"])

    return data.astype({"gantry": "int", "direction": "int"})


def read_day_transactions(day: int, transactions_dir: Path | None = None) -> pd.DataFrame:
    """Concatenate and clean every transaction CSV for one calendar day."""
    folder = transactions_dir or PATHS.transactions_dir(day)
    files = sorted(glob.glob(str(Path(folder) / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No transaction CSVs under {folder}")

    frames = [clean_transaction_frame(pd.read_csv(f, encoding="windows-1252")) for f in files]
    return pd.concat(frames, ignore_index=True)


def build_trip_od(
    transactions: pd.DataFrame,
    *,
    include_fare: bool = False,
    drop_single_gantry_trips: bool = False,
    sample_size: int | None = None,
    random_state: int = 123,
) -> pd.DataFrame:
    """Collapse per-gantry transactions into one row per trip.

    A trip's *origin* is the first gantry it was seen at and its *destination*
    the last one. Trips observed at a single gantry get a null destination
    unless ``drop_single_gantry_trips`` is set.

    Args:
        include_fare: also carry the trip's ``Assigned Fare`` through as
            ``Payment`` (used by the trip-level travel-time analysis).
        sample_size: if given, sample this many trip IDs instead of using all
            of them. The original travel-time analysis sampled 1000 trips/day
            because the full expansion was too slow.
    """
    trip_ids = transactions["Trip ID"].unique()
    if sample_size is not None and sample_size < len(trip_ids):
        rng = np.random.default_rng(random_state)
        trip_ids = rng.choice(trip_ids, size=sample_size, replace=False)

    records = []
    for tid in trip_ids:
        d = transactions[transactions["Trip ID"] == tid].reset_index(drop=True)
        single = len(d) == 1
        if single and drop_single_gantry_trips:
            continue

        start = datetime.strptime(d["Time"][0], "%H:%M:%S")
        end = np.nan if single else datetime.strptime(d["Time"].iloc[-1], "%H:%M:%S")
        record = [
            d["Trip ID"][0], d["Date"][0], start, end, d["Payment Type"][0],
            d["direction"][0], d["gantry"][0], np.nan if single else d["gantry"].iloc[-1],
        ]
        if include_fare:
            record.append(d["Assigned Fare"][0])
        records.append(record)

    columns = [
        "Trip ID", "Date", "Start_Time", "End_Time",
        "Payment Type", "direction", "Origin", "Destination",
    ]
    if include_fare:
        columns.append("Payment")
    return pd.DataFrame(records, columns=columns)


def load_trip_od(day: int, direction: str = "WB") -> pd.DataFrame:
    """Read a cached trip-OD table and restore the datetime columns."""
    df = pd.read_csv(PATHS.trip_od(day, direction))
    df["Start_Time"] = pd.to_datetime(df["Start_Time"])
    df["End_Time"] = pd.to_datetime(df["End_Time"])
    return df
