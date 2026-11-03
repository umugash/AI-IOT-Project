# India Noise Data Dictionary

## Source file: CPCB-attributed monthly dataset

Files:

- `data/external/india/cpcb_noise_2011_2018/station_month.csv`
- `data/external/india/cpcb_noise_2011_2018/stations.csv`

The downloaded distribution has 5,005 station-month rows and 70 station IDs from February 2011 through December 2018. It contains eight city labels. Kaggle attributes the data to CPCB and lists CC0; keep that attribution. The original CPCB endpoint and source terms have not been independently revalidated, and the Kaggle/TUMI-distributed copy's precise provenance remains a limitation. Do not infer a measurement definition from the field names.

### Observation table (`station_month.csv`)

| Field | Type | Meaning and handling |
|---|---|---|
| `Station` | string | Station ID; join key to station metadata. |
| `Year` | integer | Source-reported calendar year. |
| `Month` | integer 1–12 | Source-reported month. No day or time is provided. |
| `Day` | numeric, nullable | Daytime source value. Unit, weighting, duration, and aggregation are not verified; do not relabel as dB(A), Leq, or Lday. |
| `Night` | numeric, nullable | Nighttime source value with the same definition uncertainty as `Day`. |
| `DayLimit` | numeric, nullable | Limit-like source field; meaning and category compatibility require source documentation. Not treated as a verified regulatory threshold. |
| `NightLimit` | numeric, nullable | Limit-like source field; same caveat as `DayLimit`. |

The full source has three missing `Day` and ten missing `Night` values. Chennai has 715 records, ten station IDs, and no missing Day/Night values, duplicate station-month keys, or internal month gaps over the stations' observed spans.

### Station metadata (`stations.csv`)

| Field | Type | Meaning and handling |
|---|---|---|
| `Station` | string | Station identifier, unique in the metadata file. |
| `Name` | string | Source station/locality label; not necessarily an exact sensor address. |
| `City` | string | Source city label. `Mumbai` and `Navi Mumbai` are separate labels. |
| `State` | string | Source state label. |
| `Type` | string | Source category; not harmonized with TNPCB zone classes. |

No exact latitude/longitude is supplied. See [station geography audit](./STATION_GEOGRAPHY_AUDIT.md).

## Derived Chennai modelling panel

Generated at `outputs/india/models/chennai_station_month_modeling_panel.csv`.

| Field | Meaning |
|---|---|
| Source fields | Original monthly values and station metadata, unchanged except parsed numeric/year/month types. |
| `month_index` | Integer calendar-month ordering key (`Year * 12 + Month - 1`); not a timestamp. |
| `observation_period` | Source period formatted `YYYY-MM`; no day/time is implied. |
| `data_source` | Explicit source attribution string. |
| `Day_lag_1`, `Night_lag_1` | Prior station observation value; with verified contiguous Chennai series, prior month. |
| `*_lag_2`, `*_lag_3`, `*_lag_12` | Corresponding earlier station values, only from earlier rows. |
| `*_rolling_mean_3`, `*_rolling_mean_6` | Means of the preceding 3 or 6 station-month values; current target excluded. |
| `month_sin`, `month_cos` | Cyclic month-of-year encoding calculated from the observation month. |

The lags/rolling values are constructed separately for each station. A 12-month lag is required for baseline eligibility, so early rows are retained in the panel but not included in the applicable backtest.

## Historical replay JSONL events

Output: `outputs/india/historical_monthly_noise_events.jsonl`.

| Key | Meaning |
|---|---|
| `station_id`, `city` | Source station and city labels. |
| `observation_period` | Historical year-month only, formatted `YYYY-MM`. |
| `day_measurement`, `night_measurement` | Source values, or JSON `null` if absent. Units/definitions remain unresolved. |
| `data_source` | Provenance string. |
| `event_kind` | `historical_monthly_observation`. |
| `is_live_measurement` | Always `false`. |

The event schema intentionally contains no fabricated timestamp.

## Local historical event store

Output: `outputs/india/system/historical_noise_events.sqlite3`.

The `historical_noise_events` table stores one row per station/source/month and uses `(station_id, observation_period, data_source)` as its primary key. Replaying a batch is idempotent through an upsert. `is_live_measurement` is constrained to false. Events are ordered by source `observation_period` and station ID; ingestion order does not represent sensor collection time.

## Hotspot behaviour outputs

- `outputs/india/system/chennai_hotspot_behaviour_timeline.csv`: source station/month/metric fields; pooled relative threshold; relative high flag; transition state (`emergence`, `persistence`, `reduction`, below threshold, or no comparison across a gap); trailing three-period means and change classification.
- `outputs/india/system/chennai_recurring_hotspot_patterns.csv`: station, metric, calendar month, observation count, relative-high count, and observed relative-high frequency.
- `outputs/india/comparative/india_station_month_hotspot_behaviour.csv` and `india_recurring_hotspot_patterns.csv`: corresponding separate descriptive outputs for all source city labels.

Relative high flags use strict greater-than against the pooled quantile threshold, avoiding ties at the threshold from marking all equal values as high. Transition states describe changes in source-relative ranking, not regulatory exceedances. Rolling intensity comparisons are descriptive and require a contiguous four-month window to compare adjacent three-month averages; gaps are not silently bridged. Exact metric semantics remain unresolved.

## Historical next-period estimate

Output: `outputs/india/system/next_month_station_forecasts.csv`.

| Field | Meaning |
|---|---|
| `forecast_period`, `last_observed_period` | Source year-month labels only; for the current files, the historical extension is January 2019 after the last source period December 2018. |
| `predicted_value`, `forecast_method` | One-step persistence estimate from the last observed station value. |
| `hotspot_probability_estimate` | Relative high-reading frequency among up to 24 latest station-month values against the pooled historical 75th-percentile threshold. This is an uncalibrated historical frequency proxy, not a learned/calibrated probability. |
| `prediction_lower_heuristic`, `prediction_upper_heuristic` | Persistence estimate plus station-level 10th/90th percentile historical residuals from the 2017–2018 chronological backtest (pooled metric fallback if needed). Heuristic range, not a confidence interval. |
| `is_current_forecast` | Always false. |

## Separate reference datasets

- `tnpcb_2023_24_chennai_annual_noise.csv`: annual `Leq dB(A)` summary, April 2023–March 2024. Different temporal aggregation and unproven site correspondence; not joined to the monthly model data.
- `srmist_january_2025_ambient_noise.csv`: one January 2025 campus day/night pair near Tech Park Tower I. Local reference only, not a training series.
- `outputs/india/system/separate_reference_source_values.csv`: displays the reported TNPCB annual averages and SRMIST pair in separate rows with source period, unit as reported, measurement description, and source URL. It does not merge them with CPCB values.
- `outputs/india/system/source_comparability.csv`: source coverage/definitions and explicit merge status for CPCB Chennai, other city labels, TNPCB, SRMIST, and the unavailable IIT/TNPCB study.

## Reuse and attribution

Retain CPCB attribution and the Kaggle CC0 listing in derivative analysis. The source endpoint and original CPCB reuse terms are not confirmed in this audit. TNPCB/IIT Madras study data, coordinates, or detailed tables are not included; obtain written permission and follow the provided terms before using or redistributing those records.

## Streamlit presentation layer

`app.py` and `streamlit_app/` load the generated CSV/JSON artifacts described above. Interactive filters change only the displayed records; they do not alter source files or recompute model metrics. The replay action calls the existing schema validator and SQLite upsert function for a limited chronological batch or the full historical JSONL file. No database write occurs during ordinary page navigation.
