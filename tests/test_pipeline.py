"""Smoke tests for the pure-pandas transforms.

These build tiny synthetic frames rather than touching the real (non-public)
data, so they run anywhere and catch the API-level breakage that made the
original scripts stop working.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hotlane.config import INTERVALS_PER_DAY, N_GANTRIES, interval_end_times
from hotlane.data.flows import od_flow_matrices, stack_matrix
from hotlane.data.panel import (
    assign_month_time_index,
    audit_panel,
    fill_missing_gantries,
    merge_vehicle_classes,
    to_model_table,
)
from hotlane.features import add_density, round_minute_to_interval


def test_interval_grid_covers_the_analysis_window():
    times = interval_end_times()
    assert len(times) == INTERVALS_PER_DAY
    assert str(times[0]) == "05:15:00"
    assert str(times[-1]) == "20:00:00"


def _trip(origin, destination, start, end, payment="SOV", direction=21):
    return {
        "Trip ID": f"{origin}-{destination}-{start}",
        "Start_Time": pd.Timestamp(f"1900-01-01 {start}"),
        "End_Time": pd.Timestamp(f"1900-01-01 {end}"),
        "Payment Type": payment,
        "direction": direction,
        "Origin": origin,
        "Destination": destination,
    }


def test_od_flow_matrices_counts_by_interval_and_gantry():
    trips = pd.DataFrame(
        [
            _trip(1, 3, "05:10:00", "05:20:00"),
            _trip(1, 3, "05:12:00", "05:25:00"),
            _trip(2, 4, "06:01:00", "06:14:00"),
            _trip(1, 3, "05:10:00", "05:20:00", payment="HOV2"),
        ]
    )

    west, _ = od_flow_matrices(trips, "Start_Time", "Origin", "SOV")

    # Two SOV trips enter gantry 1 in the interval ending 05:15 (index 0).
    assert west[0, 0] == 2
    # The HOV2 trip is excluded from the SOV class.
    assert west.sum() == 3
    assert west.shape == (INTERVALS_PER_DAY, N_GANTRIES["WB"])


def test_od_flow_matrices_imputes_missing_destination():
    trips = pd.DataFrame([_trip(5, np.nan, "07:05:00", "07:05:00")])
    trips["End_Time"] = pd.NaT

    west, _ = od_flow_matrices(trips, "End_Time", "Destination", "SOV")

    # destination := origin + 1 == gantry 6 (index 5), at the start time.
    assert west[:, 5].sum() == 1


def test_od_flow_matrices_can_skip_imputation():
    trips = pd.DataFrame([_trip(5, np.nan, "07:05:00", "07:05:00")])
    trips["End_Time"] = pd.NaT

    west, _ = od_flow_matrices(
        trips, "End_Time", "Destination", "SOV", impute_missing_destination=False
    )
    assert west.sum() == 0


def test_stack_matrix_labels_gantries_and_timestamps():
    matrix = np.zeros((INTERVALS_PER_DAY, N_GANTRIES["WB"]))
    matrix[0, 0] = 7

    stacked = stack_matrix(matrix)

    first = stacked.iloc[0]
    assert first["gantry"] == "W01"
    assert str(first["Timestamp"]) == "05:15:00"
    assert first["volume"] == 7
    assert len(stacked) == INTERVALS_PER_DAY * N_GANTRIES["WB"]


def _panel(days=(1, 2), gantries=(1, 2, 3)):
    rows = []
    for day in days:
        for hour, minute in ((5, 15), (5, 30)):
            for gantry in gantries:
                rows.append(
                    {
                        "day": day,
                        "hour": hour,
                        "minute": minute,
                        "gantry": gantry,
                        "weekday": 0,
                        "time_index": 0,
                        "volume_o": 1.0,
                        "volume_d": 2.0,
                        "toll_rate": 0.5,
                        "speed_HOT": 65.0,
                        "speed_GP_avg": 55.0,
                        "volume_HOT": 100.0,
                        "volume_GP_avg": 200.0,
                    }
                )
    return pd.DataFrame(rows)


def test_assign_month_time_index_is_consecutive_and_shared_across_gantries():
    indexed = assign_month_time_index(_panel(), days=[1, 2])

    # 2 days x 2 intervals = 4 distinct slots, numbered 0..3.
    assert sorted(indexed["time_index2"].unique().tolist()) == [0, 1, 2, 3]
    # Every gantry in a slot shares its index.
    assert indexed.groupby("time_index2")["gantry"].nunique().eq(3).all()


def test_fill_missing_gantries_pads_incomplete_slots():
    panel = assign_month_time_index(_panel(gantries=(1, 2, 3)), days=[1, 2])
    panel = panel[~((panel["time_index2"] == 0) & (panel["gantry"] == 2))]

    gaps = audit_panel(panel, n_gantries=3)
    assert gaps.loc[0, "missing_gantries"] == [2]

    filled = fill_missing_gantries(panel, n_gantries=3)
    assert filled.groupby("time_index2")["gantry"].nunique().eq(3).all()
    # The padded row carries no measurements.
    padded = filled[(filled["time_index2"] == 0) & (filled["gantry"] == 2)]
    assert padded["speed_HOT"].isna().all()


def test_to_model_table_applies_the_canonical_schema():
    panel = assign_month_time_index(_panel(), days=[1, 2])
    table = to_model_table(panel)

    assert list(table.columns) == [
        "time_index2", "day", "weekday", "hour", "minute", "gantry",
        "num_O", "num_D", "flow_HOT", "flow_GP", "toll_rate", "speed_HOT", "speed_GP",
    ]
    assert (table["flow_GP"] == 200.0).all()


def test_to_model_table_rejects_an_incomplete_panel():
    panel = assign_month_time_index(_panel(), days=[1, 2]).drop(columns=["toll_rate"])
    with pytest.raises(KeyError, match="toll_rate"):
        to_model_table(panel)


def test_merge_vehicle_classes_totals_every_class():
    sov = assign_month_time_index(_panel(), days=[1, 2])
    other = sov[["time_index2", "gantry"]].copy()
    other["volume_o"] = 3.0
    other["volume_d"] = 4.0

    merged = merge_vehicle_classes(sov, other, other, other)

    # SOV 1 + three classes at 3 each.
    assert (merged["total_o"] == 10.0).all()
    assert (merged["total_d"] == 14.0).all()
    assert (merged["total_veh"] == 24.0).all()


def test_add_density_uses_the_hourly_flow_rate():
    df = pd.DataFrame({"flow_HOT": [100.0], "speed_HOT": [50.0]})
    out = add_density(df)

    assert out["flow_HOT_hr"].iloc[0] == 400.0
    assert out["density_HOT"].iloc[0] == 8.0


@pytest.mark.parametrize(
    ("hour", "minute", "expected"),
    [(7, 0, (7, 15)), (7, 1, (7, 15)), (7, 15, (7, 15)), (7, 16, (7, 30)),
     (7, 45, (7, 45)), (7, 46, (8, 0)), (7, 59, (8, 0))],
)
def test_round_minute_to_interval(hour, minute, expected):
    out = round_minute_to_interval(pd.DataFrame({"hour": [hour], "minute": [minute]}))
    assert (int(out["hour"].iloc[0]), int(out["minute"].iloc[0])) == expected
