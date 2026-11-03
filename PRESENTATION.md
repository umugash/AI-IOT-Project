# Presentation outline

## Slide 1 — Title

**AIoT-Based Urban Noise Hotspot Behaviour Modelling and Spatiotemporal Prediction**

India localization: Chennai historical baseline analysis completed; not a current forecast  
Legacy supplementary implementation: Urban Noise Uganda 61K  
Presenter: `[Name]` · Course: AI & IoT · Institution: `[Institution]`

> Existing Uganda data, map, and metrics in this outline are legacy results and are not Chennai/SRMIST observations. See the [final scope](./FINAL_PROJECT_SCOPE.md) and [forecasting report](./FORECASTING_BASELINE_REPORT.md).

Phase 4 local system: historical JSONL replay → validation → SQLite → hotspot behaviour and historical next-period persistence estimates. Phase 5 adds eight Streamlit research sections over generated outputs. No Kafka/Spark/Delta runtime or physical sensor is deployed.

## Slide 2 — Problem and motivation

- Urban noise is produced by different sources and varies by location.
- A sound-level dashboard alone does not classify sources or show where sampled high values concentrate.
- Defensible forecasting requires real time-stamped repeated observations; dataset fields must be verified before modelling.

## Slide 3 — Dataset research and selection

- Current India candidate: CPCB-attributed Kaggle monthly data, with 715 Chennai station-month rows (10 stations, 2011-2018).
- Chennai CSV has monthly Day/Night values but no coordinates; it supports station-level time analysis, not a precise map.
- Day/Night measurement definitions and units remain unverified; call these source values, not dB(A)/Leq.
- TNPCB 2023-24 report: annual summaries for 10 Chennai stations, not a continuous series.
- SRMIST report: one campus day/night reference near Tech Park Tower I in January 2025.
- IIT Madras/TNPCB 60-site monitoring: promising, but no verified public data endpoint and reuse permission is required.
- Legacy Uganda Figshare/SONYC sources remain separate; no Uganda values are merged into Indian data.

## Slide 4 — Feasibility decision

- Supported from currently accessible India data: exploratory Chennai station-level monthly analysis; use the annual TNPCB and SRMIST values only as separately sourced context.
- Chennai station mapping is not supported by the CPCB file because it has no coordinates.
- Kattankulathur/SRMIST time-series forecasting is not supported by one January 2025 day/night reference.
- Chennai one-month historical backtesting was performed on the 2011–2018 monthly file; it is not a forecast of present-day conditions.
- No local values are inferred from Uganda, interpolated, or invented.
- Existing source-classification and GPS mapping results apply only to the legacy Uganda prototype.

## Slide 5 — Software AIoT architecture

```text
Legacy Uganda public dataset → metadata/audio validation → preprocessing
   → geographic holdout comparison: log-Mel CNN vs. MFCC Random Forest
   → GPS-screened spatial grid → hotspot map and EDA
   → timestamp guard → optional replay only for real timestamps
```

- This depicts the distinct legacy Uganda prototype; the India pipeline uses its own monthly source, JSONL replay, SQLite store, and Streamlit presentation.
- Historical replay is simulated ingestion, not live microphone sensing.
- No physical sensor, cloud account, or broker is required.

## Slide 6 — Acoustic classification

- **Legacy Uganda pilot only; not an Indian model or result.**
- Mono audio resampled to 16 kHz.
- 64-bin log-Mel features for the compact PyTorch CNN; MFCC summary statistics for the Random Forest baseline.
- Kampala labels used for supervised ground truth by default.
- Nested geographic group split helps limit nearby-location leakage.
- Pilot: 1,200 samples, geographic split 762/198/240 (train/validation/test), 2 CNN epochs.
- Test result: CNN 14.58% accuracy / 10.29% macro F1; MFCC Random Forest 38.33% / 35.67%.
- Random Forest led in this limited pilot; metrics are initial baselines, not deployment estimates.
- Full per-class metrics, confusion matrices, and CNN learning curves: `models/classification_metrics.json` and `models/`.

## Slide 7 — Geographic hotspot mapping

- **Legacy Uganda map only; no Chennai or Kattankulathur map has been created.**
- Project GPS points to a 250 m grid.
- Exclude reported GPS accuracy above 100 m by default.
- Require at least five observations per cell.
- Display mean/median/90th percentile and sample counts.
- Candidate hotspot: top decile of cell means, an exploratory relative threshold, not a regulatory limit.
- Actual metadata pilot: 1,000 populated cells and 100 top-decile cells at these settings.

## Slide 8 — EDA and behaviour

- **Legacy Uganda dataset statistics only; not Chennai measurements.**
- 61,685 rows; Kampala 37,512; Entebbe 24,173; 19 categories.
- 60,581 exact coordinate pairs; repeated exact pairs cover 1,622 observations.
- Noise measurement range 1–84, median 48; weighting/calibration not specified in the CSV.
- Timestamp-dependent transition functions are tested but cannot be run on this public CSV.
- **Separate India Phase 3:** ten Chennai station labels ranked by relative Day/Night high-reading frequency; exact monitor coordinates unresolved, so no Chennai map or interpolation.
- Other Indian station/city comparisons use source-value quantiles; Mumbai and Navi Mumbai remain separate source labels.
- Phase 4 describes emergence, persistence, reduction, recurring month patterns, and intensity changes at station level.

## Slide 9 — IoT and syllabus alignment

- IoT data collection and CSV ingestion: dataset acquisition + strict schema validation.
- Edge AI / deep learning: compact PyTorch CNN inference, compared with a structured-feature baseline.
- Data science: spatial aggregation, EDA, class-imbalance-aware metrics.
- Uganda timestamped replay accepts source timestamps only when present and validated; India monthly replay preserves the source's `YYYY-MM` observation period with no fabricated timestamp.
- Kafka, Spark Structured Streaming, Delta Lake, and GraphFrames are not installed; the local India pipeline uses validated JSONL and SQLite without cloud services.

## Slide 10 — Limitations and future work

- Obtain the IIT Madras/TNPCB dashboard endpoint, permission terms, station metadata, and measurement/calibration documentation.
- Seek repeated timestamped SRMIST/Kattankulathur observations before making local maps or forecasts.
- The CPCB Chennai monthly file ends in 2018 and does not document the Day/Night definitions; baseline scores are historical source-value comparisons only.
- Persistence achieved the best MAE in the one-month 2017–2018 backtest; neither Ridge nor Random Forest beat it on MAE.
- The source ends in 2018; the next-period estimate is 2019-01, not 2026/current conditions. Frequency/range outputs are uncalibrated historical proxies.
- TNPCB and SRMIST report values appear as separate references, not merged into CPCB modelling data.
- Keep the TNPCB annual summary separate from the 2011-2018 monthly series.
- Keep Ugandan source labels and SONYC multilabel NYC annotations as separate domains.

## Slide 11 — Conclusion and references

- Chennai has a station-level retrospective modelling experiment, not a current forecast; exact monitor coordinates and metric definitions remain unresolved.
- No sensor-level Indian map, live feed, or Kattankulathur forecast has been produced.
- Phase 5 application: `.\.venv\Scripts\python.exe -m streamlit run app.py`; eight interactive sections, including an explicit historical replay action.
- End-to-end local run: `.\.venv\Scripts\python.exe scripts\run_india_system.py`; dashboard: `outputs\india\system\index.html`.
- Existing Uganda software and results remain supplementary and geographically separate.
- Dataset evidence and source caveats: [India Noise Dataset Research](./INDIA_DATASET_RESEARCH.md).
- Legacy dataset: [Urban Noise Uganda 61K](https://doi.org/10.6084/m9.figshare.30168901.v1).
- Data descriptor: [Nsumba et al., Scientific Data (2026)](https://doi.org/10.1038/s41597-026-06658-w).
- Supplement: [SONYC-UST v2.3](https://doi.org/10.5281/zenodo.3966543).
