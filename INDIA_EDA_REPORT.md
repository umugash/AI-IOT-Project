# India EDA and Forecasting Feasibility — Phase 2 findings

**Phase 2 analysis date: 2026-11-01 | No model training was done during Phase 2**

Phase 3 has since run the authorized historical baseline experiment. See [FORECASTING_BASELINE_REPORT.md](./FORECASTING_BASELINE_REPORT.md) for actual model metrics. The historical data, exact-location, and metric-definition limitations below remain in force.

Phase 4 adds the local event replay, hotspot transition outputs, SQLite store, separate-source reference ledger, and static dashboard via [`scripts/run_india_system.py`](./scripts/run_india_system.py). The newest-source investigation did not obtain additional 2025/2026 or repeated SRMIST raw observations; see the [audit](./INDIA_DATA_AUDIT.md) and [request draft](./DATA_ACCESS_REQUEST_DRAFT.md).

## Findings at a glance

- The CPCB-attributed Kaggle copy contains **5,005 monthly rows and 70 station records** overall. The Chennai subset has **715 rows, 10 stations, eight calendar years, and all 12 month numbers**.
- Chennai coverage is structurally complete within each station's recorded interval: five stations have 95 consecutive months (February 2011–December 2018), and five have 48 consecutive months (January 2015–December 2018). There are no internal missing months, duplicate station-month keys, or missing Chennai Day/Night values.
- Pooled Chennai values average **70.17 (Day)** and **63.84 (Night)** in the source's numeric scale; the mean paired Day-minus-Night difference is **6.33**. The CSV does not state the metric's unit, weighting, or aggregation.
- The station panel is small and old. A one-month-ahead baseline comparison is feasible as an offline research experiment; reliable current city forecasts or campus forecasts are not established.
- Exact sensor coordinates are unresolved for all ten stations. Neighborhood candidates are approximate place markers and cannot support station-level hotspot mapping or interpolation.

## Data definition and reproducibility

The analysis filters `station_month.csv` against `stations.csv` where `City == Chennai`, then retains the source-provided station ID, year, month, day, night, limits, and station type. The year/month pair is used only as a **monthly period index** for coverage, grouping, and chart axes. It is never turned into a daily or exact measurement timestamp. No data are imputed, smoothed for display, or redistributed across missing periods.

Regenerate the CSV summaries and PNG figures from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_india_eda.py
```

Outputs are in [`outputs/india/eda/`](outputs/india/eda/), including the machine-readable [`summary JSON`](outputs/india/eda/chennai_eda_summary.json), [coverage CSV](outputs/india/eda/chennai_station_coverage.csv), and [station descriptive summary CSV](outputs/india/eda/chennai_station_descriptive_summary.csv).

## Record, missingness, and station availability

The complete Kaggle copy spans February 2011–December 2018, has 5,005 monthly rows and 70 station IDs, and has no duplicate full rows or duplicate station/year/month keys. Across that whole file, `Day` has 3 missing values and `Night` has 10. The Chennai subset has **zero missing values** in all seven required fields (`Station`, `Year`, `Month`, `Day`, `Night`, `DayLimit`, `NightLimit`) and no duplicate station-month keys.

| Station | Source type | Available months | Rows | Internal missing months |
|---|---|---:|---:|---:|
| CHE01 Eye Hospital | Silence | 2011-02–2018-12 | 95 | 0 |
| CHE02 Guindy | Industrial | 2011-02–2018-12 | 95 | 0 |
| CHE03 Perambur | Commercial | 2011-02–2018-12 | 95 | 0 |
| CHE04 T. Nagar | Commercial | 2011-02–2018-12 | 95 | 0 |
| CHE05 Triplicane | Residential | 2011-02–2018-12 | 95 | 0 |
| CHE06 Pallikaranai | Commercial | 2015-01–2018-12 | 48 | 0 |
| CHE07 Velachery | Residential | 2015-01–2018-12 | 48 | 0 |
| CHE08 Washermanpet | Commercial | 2015-01–2018-12 | 48 | 0 |
| CHE09 Anna Nagar | Silence | 2015-01–2018-12 | 48 | 0 |
| CHE10 Sowcarpet | Residential | 2015-01–2018-12 | 48 | 0 |

The five later stations are **not missing** observations before 2015: those months are outside their observed span in this release. The coverage figure distinguishes absent station-months from missing values inside rows.

## Day/night numeric distributions

These are descriptive statistics for the source's numeric values; do not attach a dB(A), Leq, or calibration interpretation without a verified data dictionary.

| Period | n | Mean | Median | Standard deviation | Min | Q1 | Q3 | Max | Pooled 1.5×IQR flags |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Day | 715 | 70.17 | 71 | 6.24 | 32 | 67 | 75 | 88 | 17 |
| Night | 715 | 63.84 | 63 | 7.61 | 31 | 58 | 69 | 88 | 3 |

Pooled-IQR flags are review prompts, not evidence that a reading is invalid. The low Eye Hospital values in May/June 2018 (Day 35/32; Night 31/31) and multiple Triplicane low Day values, for example, deserve source-quality checks. Do not automatically delete, cap, or replace them. The 88-valued readings at Pallikaranai/Velachery in December 2016 and Pallikaranai Night in August 2017 are also flagged by the pooled rule.

### Station means and descriptive linear trends

The slope column is a within-station ordinary least-squares line through the monthly values, expressed as source-value units/year. It is an EDA summary—not a forecast, significance test, or evidence of causality. It is sensitive to data quality, abrupt changes, and only 4–8 annual cycles.

| ID | Station | n | Day mean (range) | Night mean (range) | Mean Day−Night | Day slope/year | Night slope/year |
|---|---|---:|---:|---:|---:|---:|---:|
| CHE01 | Eye Hospital | 95 | 63.27 (32–73) | 55.32 (31–68) | 7.96 | −0.28 | +0.93 |
| CHE02 | Guindy | 95 | 75.85 (73–80) | 70.22 (63–75) | 5.63 | +0.05 | −0.17 |
| CHE03 | Perambur | 95 | 70.09 (60–81) | 63.02 (49–83) | 7.07 | +0.40 | +1.81 |
| CHE04 | T. Nagar | 95 | 75.65 (72–81) | 68.51 (60–77) | 7.15 | +0.63 | +1.72 |
| CHE05 | Triplicane | 95 | 66.83 (43–86) | 59.75 (46–81) | 7.08 | −0.11 | +1.31 |
| CHE06 | Pallikaranai | 48 | 74.08 (62–88) | 70.10 (64–88) | 3.98 | −0.70 | +1.49 |
| CHE07 | Velachery | 48 | 70.02 (58–88) | 65.23 (53–84) | 4.79 | −0.07 | +0.78 |
| CHE08 | Washermanpet | 48 | 71.46 (69–77) | 65.33 (60–81) | 6.13 | +0.48 | +0.72 |
| CHE09 | Anna Nagar | 48 | 66.52 (64–73) | 60.75 (54–80) | 5.77 | −0.08 | −0.75 |
| CHE10 | Sowcarpet | 48 | 67.08 (48–77) | 62.50 (56–80) | 4.58 | −1.03 | −0.11 |

Day's station means range from 63.27 at Eye Hospital to 75.85 at Guindy. Night means range from 55.32 at Eye Hospital to 70.22 at Guindy. Day's station mean is higher at all ten stations. These are station-level differences in the recorded numeric scale, not a city-wide exposure estimate or compliance calculation.

### Long-term changes and seasonality

All five stations in the early cohort are present from February 2011. Their pooled arithmetic mean changes from 68.64 Day / 59.51 Night in the partial first year (2011) to 69.80 / 61.35 in 2014, then 73.52 / 67.02 in 2015, 73.25 / 68.22 in 2016, 70.72 / 66.30 in 2017, and 67.48 / 64.03 in 2018. This is not a monotone trend. The 2015 step and later decline may be real or reflect method/reporting changes; the available files do not explain it.

The series is an unbalanced panel: five stations enter in January 2015, changing any pooled Chennai average. Do not call the city-level pooled curve a uniform long-term city trend. Evaluate station-specific history and the fixed five-station cohort separately.

After centering each station by its own mean and giving each station equal weight, the month-of-year profile is modest and variable: Day ranges from about −0.46 (May) to +0.94 (August); Night ranges from about −1.59 (March) to +2.29 (August). Five stations span nearly eight years (February 2011 start, so the first calendar year is partial); the other five contribute four complete calendar years. These descriptive differences do not establish statistically robust seasonal effects. Do not use the figures as a seasonal forecast or infer weather/holiday causes.

## Figures

### Monthly trends

![Monthly Day/Night series by station](outputs/india/eda/monthly_noise_trends.png)

### Station comparisons

![Station-wise means](outputs/india/eda/station_wise_comparison.png)

![Paired Day/Night comparisons](outputs/india/eda/day_night_comparison.png)

### Distribution and anomalies

![Noise value distributions](outputs/india/eda/noise_distributions.png)

### Coverage / missingness

![Station availability and missing values](outputs/india/eda/missing_data_and_coverage.png)

### Per-station time series

![Monthly time series for each Chennai station](outputs/india/eda/station_time_series.png)

### Seasonal description

![Station-balanced month-of-year profiles](outputs/india/eda/descriptive_monthly_seasonality.png)

## Spatial feasibility

Station names are available, but monitor coordinates are not. Eight OpenStreetMap Nominatim results are preserved only as approximate named-neighbourhood/facility representations in [the coordinate mapping CSV](data/external/india/chennai_station_coordinates.csv); the exact monitor latitude/longitude fields are blank for all ten. Two locality matches are additionally ambiguous/unresolved.

**A reliable geographical hotspot map is not feasible from this dataset at present.** A map using the approximate place points would misrepresent monitor locations. There are only ten labelled station areas, not a spatially dense city mesh; there is no basis to interpolate values over intervening neighborhoods. A station-ranking chart may be made without coordinates, but it is not a geographical hotspot model.

## TNPCB and SRMIST comparisons

The TNPCB source contains annual average/minimum/maximum Leq dB(A) summaries for April 2023–March 2024. The CPCB-attributed source does not identify its measurement metric/weighting in the CSV and is monthly. Although station names mostly align, five source classification labels differ and exact sensor-site identity is unverified. The periods do not overlap. **Keep the data separate.**

The SRMIST 48.3 Day / 41.7 Night dB(A) pair from January 2025 is a local reference only. It is neither repeated campus sampling nor a training set. It cannot validate or calibrate the older Chennai stations.

## Forecasting feasibility and algorithm assessment

The seven-to-eight-year record is at the **monthly** level, and the last source month is December 2018. A 715-row panel is not 715 independent observations: each station is repeatedly observed, with serial dependence and shared city-level influences. The effective historical depth per station is only 48 or 95 months, respectively.

The only defensible forecast experiment to propose from these data is **one month ahead**, separately for Day and Night and for each station (or an explicitly station-adjusted pooled panel). A three-month-ahead comparison may be a secondary sensitivity experiment only if rolling-origin evaluation shows skill; twelve-month or longer horizons are not justified by this short, stale, definition-limited dataset. No claims about current 2026 conditions or live alerts can be made from data ending in 2018.

| Method | Feasibility from current sample | Recommendation / risk |
|---|---|---|
| Linear regression | Possible as an interpretable station-specific trend or station-fixed-effect monthly baseline (48–95 rows/station). | Include as a simple comparator only; changing data regimes and non-monotone station curves make extrapolation risky. |
| Random Forest | Technically trainable on lagged monthly features, but each station has at most 95 rows before lag reduction. A pooled model has 715 rows but just ten stations and serial dependence. | Not a first model. Only test with strict rolling-origin splits and station-aware features; high overfitting risk. |
| XGBoost | Same small effective sample limitation as Random Forest, with additional tuning capacity. | Defer unless simple baselines leave a clear nonlinear residual pattern; use conservative depth/tuning and nested time-based validation. |
| Seasonal naive (12-month lag) | Technically feasible; the shortest series has four annual cycles and the longer series eight. | Best first seasonal baseline for one-month-ahead testing, but four cycles are weak evidence of stable annual seasonality. |
| Moving average (3/6/12 months) | Feasible and simple. | Compare as smoothing/persistence baselines; it may lag abrupt changes and is not by itself a seasonal model. |
| LSTM | A 12-month sliding window would leave at most 83 sequences from a 95-month series and 36 from a 48-month series, before validation/test split. Pooling does not remove station dependence or increase the number of sites. | **Not justified.** Severe overfitting risk; do not choose it because the project is labelled AIoT. |

No model was trained or scored during Phase 2. Phase 3 later completed the requested historical backtest; see [FORECASTING_BASELINE_REPORT.md](./FORECASTING_BASELINE_REPORT.md) for the actual expanding-origin splits, scores, and per-station results. Its metrics remain historical source-value comparisons, not evidence of present-day prediction accuracy. TNPCB annual summaries and the SRMIST one-off pair were not included in the training/evaluation series.

## Direct answers and next action

1. **Can we build an actual Chennai noise hotspot model?** Not a defensible geographic hotspot surface yet. We can describe and rank the ten source station IDs; exact coordinates and metric definitions are missing.
2. **Can we generate a reliable geographical map?** No, not from station measurements until exact monitor locations are verified. Approximate OSM locality points are not sensor coordinates.
3. **Can we forecast future noise conditions?** A historical one-month-ahead experiment can be tested at station level, but reliable/current city forecasts are not established. No 2026/live forecast is supportable.
4. **What can we do for Kattankulathur/SRMIST?** Keep the January campus pair as a provenance-labelled reference and report the local time-series gap. Ask SRMIST for repeated, timestamped ambient observations and method/location metadata.
5. **Do we need additional data?** Yes: source metric/data dictionary, exact station coordinates, validated recent station-level readings, written reuse permission for the 60-site data, and repeated SRMIST/Kattankulathur measurements for any local claim.
6. **Which algorithms are most suitable?** Start with moving-average, persistence, and seasonal-naive monthly baselines; compare a low-complexity linear trend after data definitions are confirmed. Trees are optional second-stage comparisons. LSTM is not suitable for this data volume.
7. **What is next?** Review the proposed [India implementation plan](INDIA_IMPLEMENTATION_PLAN.md) and approve/request changes. Obtain data access and definitions before Phase 3 modeling. No training begins until the user approves the next phase.
