# Data directory

Nothing in `raw/`, `interim/` or `processed/` is committed — the raw inputs
contain vehicle-level toll transactions. Point `HOTLANE_DATA_ROOT` at wherever
the data actually lives if you keep it outside the repository:

```bash
export HOTLANE_DATA_ROOT=/Volumes/research/hot-lane/data
```

## `raw/` — vendor deliverables (inputs, never modified)

```
raw/
├── TripTransactions/
│   ├── 2018-10-1_TripTransactions/*.csv     one folder per calendar day
│   ├── 2018-10-2_TripTransactions/*.csv
│   └── ...
├── PostedTolls/
│   └── 2018-10_Toll_Rate_report.xlsx        one sheet per day, named "1".."31"
└── Traffic/
    └── WB_580RawTraffic_102018.xlsx         one sheet per gantry, "W01".."W17"
```

**Trip transactions** (`windows-1252` encoded, 9 banner rows before the header)
— one row per gantry read. Key fields: `Trip ID`, `Date`, `Time`, `Lane`
(e.g. `W15E03-21`, decoded into gantry number and direction code),
`Payment Type` (`SOV`, `HOV2`, `HOV3+`, `VIOLATION`, `INVALID TAG`),
`Assigned Fare`.

**Posted tolls** (12 banner rows) — `Section` (e.g. `580-GREEW_TO_GREEW`),
`Rate Time`, `Generated Toll/ Message`. Non-numeric rates mean the lane was
open to all.

**Traffic** — loop detector aggregates per gantry: `Date`, `Time_Ending`,
`Plaza_Id`, `Spd_Ln1..Spd_Ln7`, `Vol_Ln1..Vol_Ln7`. Westbound lane 1 is the HOT
lane; eastbound lanes 1–2 are.

## `interim/` — per-day intermediates (regenerable)

```
interim/
├── trip_od/WB/trip_od_day_{d}.csv               one row per trip (step 01)
├── od_flows/WB/{class}/flow_{o,d}_day_{d}.csv   15-min gantry counts (step 02)
└── daily_panel/WB/{class}/panel_day_{d}.csv     joined day panels (step 03)
```

## `processed/` — analysis-ready tables

| File | Produced by | Contents |
| --- | --- | --- |
| `panel_WB_2018-10_{class}.csv` | step 03 | month panel for one vehicle class |
| `vehicle_class_counts_WB.csv` | step 04 | SOV/HOV2/HOV3+/violation entry-exit split |
| `model_table_WB_2018-10.csv` | step 04 | the LSTM training table |
| `trip_travel_time_{class}_WB.csv` | step 05 | trip-level travel time and toll paid |
| `pred_data_WB.csv` | step 21 | panel with model-predicted speeds |

### `model_table` schema

| Column | Units | Meaning |
| --- | --- | --- |
| `time_index2` | – | 15-min interval index across the whole month (0-based) |
| `day` | – | day of October 2018 |
| `weekday` | 0–6 | Monday = 0 |
| `hour`, `minute` | – | interval **ending** time |
| `gantry` | 1–17 | gantry number, 1 = easternmost westbound entry |
| `num_O` | trips | SOV trips **entering** at this gantry this interval |
| `num_D` | trips | SOV trips **exiting** at this gantry this interval |
| `flow_HOT` | veh/15min | detector volume, HOT lane |
| `flow_GP` | veh/15min | detector volume, mean over general-purpose lanes |
| `toll_rate` | USD | posted rate for the section containing this gantry |
| `speed_HOT` | mph | detector speed, HOT lane |
| `speed_GP` | mph | detector speed, mean over general-purpose lanes |
