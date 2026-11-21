# Reference results

Reference outputs (scores and metrics), one folder per
backbone:

| File | Content |
|---|---|
| `val_test_scores.npz` | Value-token NLL anomaly score for each of the 22,602 pooled validation+test windows, plus `end_ts`, `actual_sev`, `forecast_sev`, `has_label`, `rrp_last`. |
| `metrics_actual_lor.json` | All metrics with Actual LOR as ground truth (ROC-AUC, AP, recall/precision/F1 at q90/q95/q99, per-severity recall, Forecast LOR baseline). |
| `metrics_forecast_lor.json` | Same metrics with Forecast LOR as ground truth. |

Recompute the metrics and plots from these scores (CPU only):

```bash
uv run python scripts/reproduce.py --mode evaluate
```

Rebuild the summary figures and tables from this folder:

```bash
quarto render report/results.qmd
```
