# Traffic Flow and Speed Prediction for I-580 HOT Lanes

Data pipeline, exploratory analysis and LSTM speed prediction for the I-580
express (HOT) lanes, October 2018, westbound.

The study combines three sources that arrive on different grids — vehicle-level
toll transactions, section-level posted toll rates, and lane-level loop detector
readings — into a single **gantry × 15-minute panel**, then trains a short-horizon
LSTM to predict HOT and general-purpose lane speeds from that panel.

## Layout

```
src/hotlane/            importable package — all reusable logic
  config.py             study period, network layout, filesystem paths
  data/
    transactions.py     read + clean raw transaction CSVs, build trip ODs
    flows.py            trip ODs -> 15-min gantry entry/exit counts
    tolls.py            posted toll workbook -> per-gantry 15-min rates
    traffic.py          detector workbook -> per-gantry speeds and volumes
    panel.py            joins, month-wide time index, gap filling, QC
  features.py           density, fundamental diagram, trip travel times
  modeling/
    dataset.py          scaling and sequence windowing
    model.py            HOTSpeedLSTM
    train.py            training loop
    predict.py          rolling inference
  viz/                  correlation, heatmap and time-series figures

scripts/                thin CLI entry points, numbered in run order
data/                   inputs and derived tables (gitignored; see data/README.md)
models/                 checkpoints and scalers (gitignored)
outputs/figures/        generated plots (see outputs/figures/README.md)
tests/                  smoke tests over the pure-python transforms
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .            # or: pip install -r requirements.txt
./hooks/install.sh          # guard against committing data (see "Data handling")
```

Put the vendor data under `data/raw/` in the layout described in
[`data/README.md`](data/README.md), or point `HOTLANE_DATA_ROOT` elsewhere.

## Running the pipeline

Each step reads what the previous one wrote; all of them take `--help`.

```bash
# 1. raw transactions -> one row per trip
python scripts/01_extract_trip_od.py

# 2. trip ODs -> 15-minute gantry entry/exit counts, per vehicle class
python scripts/02_build_od_flows.py --payment-class SOV HOV2 HOV3 VIO

# 3. join counts with posted tolls and detector traffic -> month panel
python scripts/03_build_month_panel.py --payment-class SOV
python scripts/03_build_month_panel.py --payment-class HOV2 --no-toll --no-traffic
python scripts/03_build_month_panel.py --payment-class HOV3 --no-toll --no-traffic
python scripts/03_build_month_panel.py --payment-class VIO  --no-toll --no-traffic

# 4. merge vehicle classes -> the modelling table
python scripts/04_build_model_table.py

# 5. trip-level travel times and tolls paid (EDA input)
python scripts/05_build_trip_travel_times.py
```

Steps 01 and 05 are the slow ones — they expand every transaction into trips.
Step 01 caches per day and skips days already built unless `--overwrite`.

### Exploratory analysis

```bash
python scripts/10_plot_correlations.py --method spearman
python scripts/11_plot_speed_heatmaps.py --days 1 2 --average
python scripts/12_plot_demand_profiles.py --gantry 1
python scripts/13_fundamental_diagram.py
```

### Model

```bash
python scripts/20_train_lstm.py --target speed_HOT speed_GP
python scripts/21_predict_lstm.py
python scripts/22_plot_predictions.py --day 1
python scripts/23_plot_speed_difference.py --scenario data/processed/some_scenario.csv
```

### Script reference

The numbered files in `scripts/` are command-line entry points for the data
pipeline, exploratory analysis, modeling, and figure generation:

| Script | Purpose |
| --- | --- |
| `01_extract_trip_od.py` | Collapse raw gantry transactions into one origin–destination record per trip. |
| `02_build_od_flows.py` | Aggregate trip records into 15-minute gantry entry and exit counts. |
| `03_build_month_panel.py` | Join flows, toll rates, and detector measurements into a month-long panel. |
| `04_build_model_table.py` | Merge vehicle classes and create the analysis-ready modeling table. |
| `05_build_trip_travel_times.py` | Calculate trip travel times and tolls paid for exploratory analysis. |
| `10_plot_correlations.py` | Plot correlations among gantries, lanes, tolls, demand, and travel time. |
| `11_plot_speed_heatmaps.py` | Create daily and month-average time–space speed heatmaps. |
| `12_plot_demand_profiles.py` | Plot daily and diurnal toll-rate and SOV-demand profiles. |
| `13_fundamental_diagram.py` | Estimate and plot density, free-flow speed, and backward wave speed. |
| `20_train_lstm.py` | Train separate LSTM speed predictors for the HOT and GP lanes. |
| `21_predict_lstm.py` | Apply the trained models across the panel to generate speed predictions. |
| `22_plot_predictions.py` | Compare predicted and observed speeds by gantry and lane. |
| `23_plot_speed_difference.py` | Compare average HOT and GP speeds between two scenarios. |
| `30_generate_summary_figures.py` | Generate an overview of the main analysis results under `outputs/figures/summary/`. |

## How the panel is built

```
transaction CSVs ──► trip OD table ──► 15-min gantry entry/exit counts ─┐
                                                                        │
posted toll workbook ──► section rates ──► per-gantry rates ────────────┼──► daily panel
                                                                        │
detector workbook ──► per-gantry lane speeds & volumes ─────────────────┘
                                                                             │
                                            month-wide time index + gap fill │
                                                                             ▼
                              SOV panel ◄── HOV2 / HOV3+ / violation counts ──► model table
```

Conventions worth knowing:

- **Intervals are labelled by their ending time.** The first interval of a day
  is `05:15` and covers 05:00–05:15; the last is `20:00`. 60 intervals per day.
- **`time_index2` is the month-wide clock.** It counts observed 15-minute slots
  consecutively across all 22 study days, and is what LSTM windows are cut
  along. `time_index` (per-day, 0–59) is kept for traceability.
- **A trip's origin is its first gantry, destination its last.** Trips seen at
  only one gantry get destination `origin + 1` at their start time (see
  "Known issues").
- **Study days are the 22 weekdays of October 2018.**

## Data handling

**The inputs are internal and can not reach a public repository.** Raw trip
transactions are vehicle-level records; every table derived from them
(`data/interim`, `data/processed`) inherits that restriction, as do the model
checkpoints and scalers, which are fitted on them.

Three layers guard this:

1. `.gitignore` excludes all of `data/` and `models/` — only `data/README.md` is
   tracked — plus every data-shaped extension (`.csv`, `.xlsx`, `.pkl`, `.pth`,
   …) anywhere in the tree. `outputs/` is deliberately **not** excluded: the
   figures there are aggregated across trips and are safe to publish.
2. A pre-commit hook rejects any staged file under `data/` or `models/`, or with
   a data-shaped extension. Install it once per clone:

   ```bash
   ./hooks/install.sh
   ```

   Use `git commit --no-verify` only if you are certain the file is safe.
3. Keeping the data out of the working tree entirely is the strongest option —
   put it anywhere on disk and point the code at it:

   ```bash
   export HOTLANE_DATA_ROOT=/path/outside/the/repo/hot-lane-data
   ```

Everything under `data/` and `models/` is regenerable from the raw inputs plus
this code, so nothing is lost by never committing it. Figures are the exception
and are tracked — but take the same care there: heatmaps and profiles are
aggregated and fine, anything plotted at trip level is not.
