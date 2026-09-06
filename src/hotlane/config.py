"""Project-wide configuration: study period, network layout and filesystem paths.

Everything that used to be a hard-coded literal scattered across the analysis
scripts lives here, so a new study month / direction only needs edits in one
place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, time
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# Study period
# --------------------------------------------------------------------------- #
YEAR = 2018
MONTH = 10

#: Weekdays of October 2018 that are used throughout the analysis
#: (weekends and the 2018-10-08 / 10-09 style gaps are excluded by construction).
STUDY_DAYS: list[int] = (
    list(range(1, 6))
    + list(range(8, 13))
    + list(range(15, 20))
    + list(range(22, 27))
    + list(range(29, 32))
)

#: Analysis window within a day. Intervals are labelled by their *ending* time,
#: so the first interval 05:15 covers 05:00-05:15.
DAY_START = time(5, 15)
DAY_END = time(20, 0)
INTERVAL_MINUTES = 15
#: Number of 15-minute intervals per day (05:15 ... 20:00 inclusive).
INTERVALS_PER_DAY = 60

# --------------------------------------------------------------------------- #
# Network layout (I-580 express lanes)
# --------------------------------------------------------------------------- #
#: Direction codes as they appear in the "Lane" field of the transaction files.
DIRECTION_CODES = {"WB": (21,), "EB": (1, 2)}

N_GANTRIES = {"WB": 17, "EB": 15}

#: Tolling section -> gantries covered by that section (westbound).
WB_SECTION_GANTRIES: dict[str, list[str]] = {
    "GREEW": ["W01", "W02", "W03"],
    "NFIRW": ["W04", "W05", "W06"],
    "LIVEW": ["W07", "W08", "W09"],
    "ISABW": ["W10"],
    "AIRWW": ["W11", "W12"],
    "FALLW": ["W13", "W14"],
    "SANRW": ["W15"],
    "HOPYW": ["W16", "W17"],
}

#: Eastbound section -> number of gantries. Kept for reference; the eastbound
#: pipeline was never run end-to-end (see README, "Scope").
EB_SECTION_SIZES = {
    "HACIE": 2, "FALLE": 2, "AIRWE": 1, "LIVEE": 3,
    "NFIRE": 2, "VASCE": 2, "GREEE": 3,
}

#: Distance between consecutive gantries, in miles (used for travel times).
GANTRY_SPACING_MILES = 0.82

#: "Payment Type" values in the transaction files, grouped into vehicle classes.
PAYMENT_CLASSES: dict[str, tuple[str, ...]] = {
    "SOV": ("SOV",),
    "HOV2": ("HOV2",),
    "HOV3": ("HOV3+",),
    "VIO": ("VIOLATION", "INVALID TAG"),
}

#: Arbitrary but fixed reference date used when parsing bare clock times.
BASE_DATE = "2018-01-01"


def gantry_ids(direction: str = "WB") -> list[str]:
    """['W01', 'W02', ...] for the requested direction."""
    prefix = "W" if direction == "WB" else "E"
    return [f"{prefix}{i:02d}" for i in range(1, N_GANTRIES[direction] + 1)]


def interval_end_times() -> list[time]:
    """The 60 interval-ending clock times, 05:15 ... 20:00."""
    start = datetime.combine(datetime.strptime(BASE_DATE, "%Y-%m-%d"), DAY_START)
    end = datetime.combine(datetime.strptime(BASE_DATE, "%Y-%m-%d"), DAY_END)
    return [ts.time() for ts in pd.date_range(start, end, freq=f"{INTERVAL_MINUTES}min")]


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    """Filesystem layout.

    Override the data root with the ``HOTLANE_DATA_ROOT`` environment variable
    if the (large, non-public) inputs live outside the repository.
    """

    root: Path = REPO_ROOT

    @property
    def data(self) -> Path:
        env = os.environ.get("HOTLANE_DATA_ROOT")
        return Path(env) if env else self.root / "data"

    # raw vendor deliverables ------------------------------------------------ #
    @property
    def raw(self) -> Path:
        return self.data / "raw"

    def transactions_dir(self, day: int) -> Path:
        return self.raw / "TripTransactions" / f"{YEAR}-{MONTH:02d}-{day}_TripTransactions"

    @property
    def toll_workbook(self) -> Path:
        return self.raw / "PostedTolls" / f"{YEAR}-{MONTH:02d}_Toll_Rate_report.xlsx"

    def traffic_workbook(self, direction: str = "WB") -> Path:
        return self.raw / "Traffic" / f"{direction}_580RawTraffic_{MONTH:02d}{YEAR}.xlsx"

    # intermediate artefacts ------------------------------------------------- #
    @property
    def interim(self) -> Path:
        return self.data / "interim"

    def trip_od(self, day: int, direction: str = "WB") -> Path:
        return self.interim / "trip_od" / direction / f"trip_od_day_{day}.csv"

    def od_flows(self, day: int, payment_class: str, side: str, direction: str = "WB") -> Path:
        """``side`` is 'o' (origin counts) or 'd' (destination counts)."""
        return (
            self.interim / "od_flows" / direction / payment_class
            / f"flow_{side}_day_{day}.csv"
        )

    def daily_panel(self, day: int, payment_class: str, direction: str = "WB") -> Path:
        return self.interim / "daily_panel" / direction / payment_class / f"panel_day_{day}.csv"

    # analysis-ready tables -------------------------------------------------- #
    @property
    def processed(self) -> Path:
        return self.data / "processed"

    def month_panel(self, payment_class: str, direction: str = "WB") -> Path:
        return self.processed / f"panel_{direction}_{YEAR}-{MONTH:02d}_{payment_class}.csv"

    def model_table(self, direction: str = "WB") -> Path:
        """The canonical LSTM training table (all vehicle classes merged)."""
        return self.processed / f"model_table_{direction}_{YEAR}-{MONTH:02d}.csv"

    def trip_travel_times(self, payment_class: str, direction: str = "WB") -> Path:
        return self.processed / f"trip_travel_time_{payment_class}_{direction}.csv"

    # outputs ---------------------------------------------------------------- #
    @property
    def models(self) -> Path:
        return self.root / "models"

    @property
    def figures(self) -> Path:
        return self.root / "outputs" / "figures"

    def figure(self, *parts: str) -> Path:
        """Path to a figure file, creating its directory."""
        path = self.figures.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def figure_dir(self, *parts: str) -> Path:
        """Path to a figure sub-directory, creating it."""
        path = self.figures.joinpath(*parts)
        path.mkdir(parents=True, exist_ok=True)
        return path


PATHS = Paths()
