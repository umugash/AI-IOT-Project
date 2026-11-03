# Chennai Forecasting Baseline Report

## Result and interpretation

This is a **historical backtest**, not a forecast of current Chennai conditions. The source observations end in 2018 and their Day/Night units, weighting, and aggregation are unresolved. Errors below are in the dataset's **source-value units**, not verified dB(A).

One-step-ahead expanding-origin backtesting covers January 2017–December 2018. Every test month predicts all ten stations from strictly earlier rows; the observed month is only available to predict the next origin. Each metric/method therefore has 240 test predictions (10 stations × 24 months). No random time split is used.

## Chronological partitions

| Metric | Partition | Years | Rows | Stations |
|---|---|---:|---:|---:|
| Day and Night | Initial training history | 2012–2015 | 235 each | 5 |
| Day and Night | Validation period | 2016 | 120 each | 10 |
| Day and Night | Test/backtest period | 2017–2018 | 240 each | 10 |

Rows before a station has 12 months of history remain in the modelling panel but are not forecast-eligible. At each test origin, all prior eligible months are used; consequently the later test predictions incorporate observations that would have become available month by month. The 2016 block is reported as a chronological validation partition; no hyperparameter search or model selection was performed from it.

## Aggregate historical test metrics

| Metric | Method | MAE | RMSE |
|---|---|---:|---:|
| Day | Persistence | **3.138** | 5.781 |
| Day | Seasonal naive (12-month) | 5.008 | 7.763 |
| Day | Moving average (3-month) | 3.408 | 5.815 |
| Day | Pooled Ridge | 3.281 | 5.473 |
| Day | Station-specific Ridge | 4.113 | 6.466 |
| Day | Pooled Random Forest | 3.284 | **5.523** |
| Night | Persistence | **3.483** | 5.718 |
| Night | Seasonal naive (12-month) | 6.167 | 8.804 |
| Night | Moving average (3-month) | 3.799 | 5.875 |
| Night | Pooled Ridge | 3.618 | 5.620 |
| Night | Station-specific Ridge | 4.835 | 7.044 |
| Night | Pooled Random Forest | 3.535 | **5.530** |

Persistence has the best MAE for both targets. Random Forest has the lowest RMSE, narrowly ahead of Ridge/persistence, but neither learned method improves MAE on persistence. This is not evidence that a complex model is generally superior; it supports retaining persistence as the primary baseline. The 12-month seasonal-naive baseline performs poorly on this test period. Results are source-specific and based on a short, small panel.

Per-station, per-method MAE/RMSE are in [chennai_walk_forward_metrics_by_station.csv](./outputs/india/models/chennai_walk_forward_metrics_by_station.csv). Full predictions are in [chennai_walk_forward_predictions.csv](./outputs/india/models/chennai_walk_forward_predictions.csv), with actual-versus-predicted figures in the model output directory.

## Algorithm suitability

- **Persistence:** best-supported baseline with the lowest observed MAE; simple and suitable for comparison.
- **Seasonal naive:** valid to evaluate because 12-month lags exist, but weak on this test window.
- **Moving average:** a useful smooth baseline; does not beat persistence on MAE.
- **Linear/Ridge:** implementable in a pooled panel, but did not beat persistence on MAE. Station-specific Ridge performs worse, consistent with low sample counts per station.
- **Random Forest:** a constrained tree baseline was run. Slightly lower RMSE than persistence but worse MAE; not sufficient to justify operational deployment.
- **XGBoost:** not run. It would add another learned model and dependency without enough evidence that a more flexible learner is warranted.
- **LSTM:** not run and not justified. Each station has only 48 or 95 monthly observations overall, before lag loss; many station-months are correlated, and the test set has only 24 points per station.

## Limitations and safeguards

1. Day/Night measurement definitions and numeric units are unresolved. Metric names, error magnitudes, and source limits must not be interpreted as legal noise levels.
2. Data stop in 2018; no result describes 2026/current conditions.
3. The dataset contains ten Chennai station labels and no verified exact sensor coordinates. These are station-level time-series experiments, not geographic hotspot forecasts.
4. Five stations start in 2011; five start in 2015. The initial training period has only five stations. Later pooled fits have more stations but remain a small sample.
5. No XGBoost/LSTM evaluation, uncertainty intervals, external validation, unit calibration, or present-day data validation has been performed.
6. This is a one-month horizon only. Longer-horizon forecasts are not evaluated.

## Reproducibility

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\run_india_analysis.py
```

Outputs include aggregate/per-station scores, chronological split description, test predictions, the modelling feature panel, and the two actual-versus-predicted plots under `outputs/india/models/`. Forecasting source code is in `src/urbannoise/india_forecasting.py`.

## Historical next-period product

The integrated Phase 4 pipeline adds `outputs/india/system/next_month_station_forecasts.csv`. Because the latest source observation is December 2018, its station-level persistence estimates are for January 2019 only. This is a reproducible historical extension of the last available period, **not a forecast of the present or future conditions in 2026**.

For each station and Day/Night target, the output reports:

- Last observed value copied forward by persistence.
- An empirical relative-high frequency over the latest up to 24 observations, using the pooled 75th-percentile threshold. This is a descriptive historical frequency proxy, not a calibrated or learned probability.
- A range based on that station's 10th/90th percentile 2017–2018 persistence residuals (with pooled fallback). It is a heuristic historical error range, not a confidence/prediction interval with a coverage guarantee.

No geolocation or city-map prediction is generated. This product is included to complete the local inference pipeline while preserving the source's temporal and measurement limits.
