"""Posted toll rates: read the monthly workbook and expand sections to gantries."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from hotlane.config import BASE_DATE, INTERVAL_MINUTES, PATHS, WB_SECTION_GANTRIES

#: The toll workbook has a 12-row banner ahead of the header row.
_HEADER_ROWS = 12

#: Rows 21..80 of the per-section, per-interval averages correspond to the
#: 05:15-20:00 analysis window.
_WINDOW = slice(21, 81)


def read_toll_day(day: int, workbook: Path | None = None) -> pd.DataFrame:
    """Read one day's sheet from the monthly posted-toll workbook."""
    xls = pd.ExcelFile(workbook or PATHS.toll_workbook)
    toll = pd.read_excel(xls, str(day))

    toll = toll.tail(-_HEADER_ROWS).reset_index(drop=True)
    toll.columns = toll.iloc[0]
    toll = toll.tail(-1).reset_index(drop=True)

    # 'Section' is only stamped on the first row of each block.
    toll["Section"] = toll["Section"].ffill()
    # Non-numeric entries mean "lane open to all" and become NaN.
    toll["Generated Toll/ Message"] = pd.to_numeric(
        toll["Generated Toll/ Message"], errors="coerce"
    )
    return toll


def toll_by_section(
    toll_data: pd.DataFrame,
    section_gantries: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Average the posted rate of each section over 15-minute intervals.

    Only the within-section rate (``580-<SEC>_TO_<SEC>``) is used, which is the
    rate a vehicle entering that section sees.
    """
    sections = section_gantries or WB_SECTION_GANTRIES
    frames = []
    for section in sections:
        rows = toll_data[toll_data["Section"] == f"580-{section}_TO_{section}"].reset_index(
            drop=True
        )
        rows["Rate Time_ceil"] = [
            pd.to_datetime(f"{BASE_DATE} {t}").ceil(f"{INTERVAL_MINUTES}min").time()
            for t in rows["Rate Time"]
        ]
        grouped = (
            rows.groupby("Rate Time_ceil")
            .mean(numeric_only=True)
            .reset_index(drop=False)
            .iloc[_WINDOW]
            .reset_index(drop=True)
        )
        grouped["Section"] = section
        frames.append(grouped)
    return pd.concat(frames).reset_index(drop=True)


def expand_toll_to_gantries(
    section_tolls: pd.DataFrame,
    section_gantries: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Broadcast each section's rate onto every gantry inside that section."""
    sections = section_gantries or WB_SECTION_GANTRIES
    frames = []
    for section, gantries in sections.items():
        block = section_tolls[section_tolls["Section"] == section]
        for gantry in gantries:
            gantry_block = block.copy()
            gantry_block["gantry"] = gantry
            frames.append(gantry_block)
    out = pd.concat(frames).reset_index(drop=True)
    return out.rename(columns={"Rate Time_ceil": "Timestamp"})


def toll_rates_for_day(day: int, workbook: Path | None = None) -> pd.DataFrame:
    """Convenience wrapper: workbook sheet -> per-gantry 15-minute toll rates."""
    return expand_toll_to_gantries(toll_by_section(read_toll_day(day, workbook)))
