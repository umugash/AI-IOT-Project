# Final Project Scope

## Project

**AIoT-Based Urban Noise Hotspot Behaviour Modeling and Spatiotemporal Prediction**

## Locked geographic scope

1. **Chennai, Tamil Nadu — primary modelling region.** Use the CPCB-attributed monthly station panel for station-level descriptive behaviour analysis and historical one-month-ahead baseline backtesting.
2. **SRMIST, Kattankulathur — separate local reference/case study.** The January 2025 day/night report values are contextual references only. They are not part of the Chennai model panel. Repeated, timestamped campus measurements are needed for local temporal or predictive analysis.
3. **Other Indian cities — comparative analysis.** Compare the CPCB-attributed source values by the source's city labels. Mumbai and Navi Mumbai remain separate labels, so the file contains eight city labels and seven broader metropolitan groups if those two are grouped for a presentation. Comparisons are descriptive and retain the source's unresolved metric definition.
4. **Uganda — separate supplementary prototype.** Preserve its audio classification and spatial workflow. Do not mix Uganda records or model inputs with Indian data.

## What this implementation does

- Validates and joins the 2011–2018 monthly observation table to station metadata.
- Builds a Chennai station-month feature panel using only prior observations, including 1/2/3/12-month lags, shifted 3/6-month rolling means, and month-of-year encoding.
- Identifies relative high-reading frequency and persistence by station, separately for Day and Night. Thresholds are exploratory pooled quantiles, not regulatory limits.
- Runs persistence, 12-month seasonal-naive, 3-month moving-average, pooled Ridge, station-specific Ridge, and pooled Random Forest one-month-ahead historical backtests.
- Creates city/station descriptive comparison tables and a chronological JSONL replay clearly labelled as historical monthly data.
- Compares relative high-reading frequency across cities and stations using source-value quantiles, not regulatory standards.
- Provides a local end-to-end historical replay adapter: JSONL virtual events → schema validation → SQLite storage → analysis/forecast outputs → generated dashboard.
- Provides a Streamlit research interface with eight navigable sections and interactive filters/charts over generated, validated outputs.
- Produces a one-step persistence estimate beyond the latest historical month, with a recent relative-high frequency proxy and heuristic empirical-error range. It is explicitly not a current forecast, calibrated confidence interval, or learned hotspot probability.

## Scientific and product boundaries

- `Day` and `Night` have no verified unit, weighting, measurement interval, or aggregation definition in the available CPCB-attributed CSV. Reports call them **source values**, not dB(A), Leq, Lday, or Lnight.
- TNPCB 2023–24 annual Leq dB(A) summaries and the SRMIST January 2025 reference remain separate products; they are not merged or used for model fitting.
- Exact sensor coordinates for the ten Chennai station labels remain unresolved. Do not make a sensor map, city-wide heatmap, or interpolated noise surface from approximate locality points.
- All forecasts are retrospective backtests of historical data ending in 2018. They are not forecasts of present-day Chennai or SRMIST conditions.
- Replay is a software simulation from historical monthly rows, not live IoT collection. No actual sensor, public live API, Kafka broker, Spark cluster, or Delta Lake service is connected.
- The laptop environment does not have Kafka, Spark Structured Streaming, or Delta Lake installed. SQLite and the local replay adapter are the implemented and tested architecture; no unnecessary cloud or paid service is required.
- Streamlit/Plotly form the local presentation layer; charts show only generated source observations and backtest files. The app never reports live acquisition.
- No regulatory-compliance or legal-violation claims are made.

## Current answer to feasibility questions

| Question | Evidence-based answer |
|---|---|
| Can this build an actual Chennai hotspot model? | A station-level relative hotspot-behaviour analysis is implemented. A geographically located sensor hotspot model is blocked by unresolved measurement definitions and exact coordinates. |
| Can it generate a reliable geographical map? | No. Ten sensor locations are unresolved; eight locality candidates are approximate context only. |
| Can it forecast future noise conditions? | It can compare historical one-month-ahead baselines in chronological backtesting. It cannot provide a reliable current or future operational forecast from data ending in 2018. |
| What can be done for Kattankulathur/SRMIST? | Preserve the documented January 2025 day/night pair as a local reference. Obtain repeated, timestamped, method-documented observations before local modelling. |
| Is additional data needed? | Yes: authorized IIT Madras/TNPCB station-level records and metadata, exact sensor locations, verified metric/aggregation definitions, and repeated SRMIST observations. |
| Which algorithms are most suitable now? | Persistence is the strongest MAE baseline in this backtest; pooled Ridge and Random Forest are useful comparisons but do not beat persistence on MAE. LSTM is not justified by these short series. |
| Is there a future hotspot probability? | The system exports a station-level recent historical relative-high frequency proxy for the next historical period. It is not a calibrated probability or geographic prediction. |

## Next steps

The [formal data-access request](./DATA_ACCESS_REQUEST_DRAFT.md) is prepared but not sent. Prioritize written authorization and provenance resolution; do not extend model complexity until those gates are met. If access is granted, audit measurement semantics, site identity, timestamps, quality controls, coordinate accuracy, and reuse permission before adding a separate, compatible contemporary stream.
