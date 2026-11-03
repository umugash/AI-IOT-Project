# India Project Implementation Plan

**Status: Phases 0–4 implemented; data authorization and scientific-definition gates remain open.**

## Final scope

- Chennai is the primary modelling region for historical monthly station-level behaviour and baseline forecasting.
- SRMIST/Kattankulathur is a separate local reference, not a Chennai training row.
- Other source city labels are used for separate descriptive comparison; Mumbai and Navi Mumbai remain separate in the underlying source.
- Uganda remains a separate supplementary experiment. No Indian records are joined to its audio/GPS dataset or model.

See [FINAL_PROJECT_SCOPE.md](./FINAL_PROJECT_SCOPE.md) for scope boundaries, claims, and answers to feasibility questions.

## Implemented pipeline

```text
Static CPCB-attributed monthly source files
        ↓
Validate schema, station IDs, numeric fields, and unique station-month keys
        ↓
Source-separated analysis tables
        ├── Chennai station-month panel + prior-only lags/rolling features
        │       ├── relative Day/Night hotspot behaviour by station label
        │       └── persistence / seasonal naive / moving average / Ridge / RF
        │           expanding-origin one-month-ahead historical backtest
        ├── other-city and station descriptive comparison
        └── chronological monthly JSONL historical replay (not live IoT)

TNPCB annual summary ── separate reference only
SRMIST one-day/night pair ── separate local reference only
Uganda audio/GPS system ── separate supplementary implementation
```

## Phase results

### Phase 0/1 — Source research

Identified the CPCB-attributed/Kaggle monthly multi-city dataset, TNPCB annual report summary, SRMIST single local reference, and IIT Madras/TNPCB 60-site monitoring study. The IIT/TNPCB study's raw data/API and reuse conditions are not publicly verified here; the formal [data-access request draft](./DATA_ACCESS_REQUEST_DRAFT.md) has not been sent.

### Phase 2 — Dataset validation and EDA

Completed CSV validation, Chennai station coverage, missingness, distributions, trends, seasonality, station comparisons, and exact-versus-approximate location review. Chennai has 715 rows at ten stations; five have 95 consecutive monthly rows and five have 48. No exact sensor locations or verified Day/Night unit/aggregation definitions were found. See [INDIA_EDA_REPORT.md](./INDIA_EDA_REPORT.md) and [STATION_GEOGRAPHY_AUDIT.md](./STATION_GEOGRAPHY_AUDIT.md).

### Phase 3 — Baseline modelling, hotspot behaviour, comparison, and replay

Implemented and executed:

- A Chennai modelling panel with 1/2/3/12-month lags, shifted 3/6-month rolling means, and cyclic month features. Features are restricted to strictly earlier consecutive calendar months; gaps are not treated as adjacent lags.
- Separate Day and Night relative hotspot summaries using a pooled top-quartile threshold. These describe relative source-value ranks, not legal noise standards. No geographic surface is produced.
- Historical one-month expanding-origin baseline comparisons for 2017–2018, including per-station scores.
- City-label and station-level descriptive comparisons across all 70 stations and 5,005 source rows.
- Relative high-reading frequency by city and station, calculated against pooled source-value Day/Night top-quartile thresholds. This is a relative comparison, not a regulatory assessment.
- Ordered historical JSONL replay events containing year-month periods and `is_live_measurement=false`; no artificial timestamps are created.

The actual metrics and model limits are in [FORECASTING_BASELINE_REPORT.md](./FORECASTING_BASELINE_REPORT.md). Persistence yielded the best MAE for both Day and Night. Pooled Ridge and Random Forest did not outperform it on MAE. No XGBoost or LSTM is justified on this panel.

### Phase 4 — Local intelligence system

Implemented in `scripts/run_india_system.py`:

- Re-runs the full Chennai EDA and figures before the model/hotspot pipeline.
- Ordered historical JSONL replay as a virtual source, then strict event validation and idempotent local SQLite storage.
- Chennai station-month behaviour labels for relative hotspot emergence, persistence, reduction, recurring calendar-month frequency, and descriptive three-period intensification/reduction.
- One-period-ahead persistence estimates beyond the latest available month, plus a recent empirical high-reading frequency proxy and historical-residual heuristic range. These are labelled historical, uncalibrated, and not current forecasts.
- Source compatibility ledger keeping CPCB Chennai, other cities, TNPCB annual report, SRMIST reference, and the inaccessible IIT/TNPCB study distinct.
- Static local HTML dashboard with historical backtest, behaviour, comparison and forecast figures.
- Separate-source TNPCB/SRMIST reference-value table and machine-readable source comparability ledger; no incompatible values are pooled.

The current environment has no Kafka broker, PySpark/Spark Structured Streaming, or Delta Lake runtime. A Java executable exists, but not the required streaming stack. For 5,005 monthly records, the implemented Python/SQLite pipeline is the local, dependency-free option. The proposed Kafka → Spark → Delta architecture is documented as a future scale-out path, not falsely reported as running.

### Phase 5 — Streamlit application

Implemented as `app.py` plus the `streamlit_app/` presentation modules:

- Eight sidebar sections: overview, Chennai analysis, hotspot intelligence, predictive analytics, India comparisons, IoT replay, SRMIST case study, and methodology/sources.
- Cached read-only source/output loaders with schema checks and explicit missing/corrupt-data messages.
- Interactive station, year, month, category, metric, city, method, and historical replay-batch controls.
- Plotly visualizations, filtered tables, CSV download controls, actual generated model metrics, and transparent historical-only warnings.
- Replay action is explicit and session-scoped; limited batches or the full JSONL stream call the existing event validator and idempotent SQLite module.
- No invented coordinates or measurements, no fabricated current forecast, and no SQL mutation during ordinary navigation.

## Open gate A — authorize and define contemporary records

Before any local, spatial, live, or operational forecast work:

1. Ask TNPCB/IIT Madras for the official dashboard/API or documented export, access requirements, historical 2025/2026 station-level measurements, and written data-use/publication terms.
2. Request a data dictionary covering metric (Leq/Lday/Lnight or other), dB weighting, measurement interval and aggregation, time zone, QA flags, calibration, missing data, maintenance, and any sensor relocations.
3. Request exact station IDs and sensor coordinates/addresses with accuracy and effective dates. Do not copy restricted report coordinates before written authorization.
4. Ask TNPCB whether 2021/2022 Tamil Nadu station/month noise observations can be obtained. Searches documented in the audit have not verified a public downloadable city/station/month dataset for those years; absence from the public pages checked is not proof of no records.
5. Request repeated, timestamped, method-documented SRMIST/Kattankulathur observations if a local case study is desired.
6. Resolve the original source definition and reuse conditions for the Kaggle/TUMI-distributed CPCB-attributed table. Kaggle currently lists CC0 and CPCB attribution, but the original source page and terms were not independently confirmed.
7. Keep CPCB, TNPCB, IIT Madras/TNPCB, and SRMIST data separate until metric semantics, time aggregation, station identity, spatial identity, and reuse permission are compatible.

## Phase 4 — conditional extension

Only after Gate A:

- Audit new authorized files and preserve immutable originals, source hashes, license/permission, and retrieval metadata.
- Build a separate analysis partition for the newly documented source; do not overwrite existing outputs or silently merge panels.
- Re-run chronological baseline evaluation using the actual measurement interval and adequate history. Preserve one-step monthly comparisons only when source frequency supports them.
- Add mapped station displays only for exact, authorized sensor coordinates and matched station identities. Do not interpolate a city-wide surface without a defensible sampling design and independent validation sites.
- Evaluate additional models only after measured improvement against baselines on chronological holdouts. Reconsider LSTM only if the resulting sample size and temporal resolution justify it.
- If local sensor collection is later approved, document hardware, calibration, placement, privacy, and consent separately; the present replay is software-only and historical.

## Syllabus-aligned implementation

The current solution demonstrates source validation, EDA, time-series feature engineering, scikit-learn baselines, chronological backtesting, visualization, and historical event replay using a static monthly dataset. Kafka, Spark, and Delta Lake are not added: the monthly static file does not require distributed/stream processing. Explain these as optional architectural extensions, not implemented services.
