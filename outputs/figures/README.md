# Generated figures

This directory contains plots produced by the analysis scripts. The images are
derived from aggregated traffic data and are safe to include in the repository.

## `summary/`

`scripts/30_generate_summary_figures.py` collects the main analysis results in
one directory. Existing images can be regenerated with:

```bash
python scripts/30_generate_summary_figures.py
python scripts/30_generate_summary_figures.py --with-training
python scripts/30_generate_summary_figures.py --scenario path/to/scenario_panel.csv
```

| Output | Description |
| --- | --- |
| `hot_month_average_speed.jpg` | Month-average HOT-lane speed heatmap. |
| `gp_month_average_speed.jpg` | Month-average general-purpose-lane speed heatmap. |
| `sov_origin_flow_by_day.jpg` | SOV entry flow by day and hour. |
| `toll_rate_by_day.jpg` | Posted toll rate by day and hour. |
| `toll_rate_origin_flow_correlation.jpg` | Hourly relationship between toll rates and SOV entry flow. |
| `speed_predictions_gantries_01_08.jpg` | Predicted and observed speeds for gantries 1–8. |
| `speed_predictions_gantries_09_17.jpg` | Predicted and observed speeds for gantries 9–17. |
| `hot_speed_day_01_baseline.jpg` | Baseline HOT-lane speeds for the selected day. |
| `gp_speed_day_01_baseline.jpg` | Baseline GP-lane speeds for the selected day. |
| `hot_speed_month_baseline.jpg` | Baseline HOT-lane speeds averaged over the month. |
| `gp_speed_month_baseline.jpg` | Baseline GP-lane speeds averaged over the month. |

Passing `--with-training` also creates `training_loss_hot.jpg` and
`training_loss_gp.jpg`. Passing `--scenario` creates matching files ending in
`_scenario.jpg` for comparison with the baseline.

## Other generated directories

The exploratory scripts (`10`–`13`, `22`, and `23`) write their plots to other
subdirectories of `outputs/figures/`. Those outputs are reproducible and can be
regenerated as needed.

Do not add figures containing trip-level or otherwise sensitive records.
