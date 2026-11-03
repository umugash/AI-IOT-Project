# India Noise Dataset Research

**Research snapshot: 2026-11-01 | Scope: Phase 0/1 source discovery followed by Phase 2/3 implementation**

> Phases 2–4 have since audited the downloaded CSVs, generated the Chennai EDA charts, individually checked station locality geocoding, run retrospective monthly baseline backtests, and integrated the local historical replay/SQLite pipeline. See [INDIA_DATA_AUDIT.md](./INDIA_DATA_AUDIT.md), [INDIA_EDA_REPORT.md](./INDIA_EDA_REPORT.md), [FINAL_PROJECT_SCOPE.md](./FINAL_PROJECT_SCOPE.md), [FORECASTING_BASELINE_REPORT.md](./FORECASTING_BASELINE_REPORT.md), and [DATA_ACCESS_REQUEST_DRAFT.md](./DATA_ACCESS_REQUEST_DRAFT.md).

## Executive summary

The existing application, map, audio data, and model results are a Uganda prototype. The separate India Phase 3 now has historical baseline models and station-level relative hotspot analysis, but no operational or current forecast. The verified Indian sources offer different levels of evidence:

- The CPCB-attributed monthly dataset distributed on Kaggle is the best currently accessible candidate for Chennai station-level temporal analysis: 715 monthly records across ten named Chennai stations from February 2011 through December 2018. It contains no coordinates, and its values are monthly `Day`/`Night` readings, not daily or live observations.
- TNPCB's 2024 annual report provides a separate April 2023-March 2024 summary for ten Chennai stations, not a continuous series. It must not be joined to the older monthly data as if the intervening years were observed.
- An SRMIST report provides one January 2025 campus ambient-noise day/night reference near Tech Park Tower I. It is not a time series and does not provide exact observation times or coordinates.
- An IIT Madras/TNPCB study describes continuous monitoring at 60 Chennai sites and a web dashboard with historical display/download. The report prohibits reproduction without written permission and does not identify the dashboard URL. TNPCB's public `noise.php` page contains standards, not readings; no public dashboard endpoint, export, or API was verified in this audit.

**Recommendation:** Keep Chennai as the data-analysis candidate and Kattankulathur/SRMIST as the local deployment context and explicit data gap. Do not make Kattankulathur forecasts or maps from the Uganda data, the Chennai-wide monthly dataset, or one campus report. The 60-site monitoring data could materially improve the local project if TNPCB/IIT Madras provides a usable endpoint and reuse permission.

## Scope and safeguards

This document records source research, not a deployment. Phase 3 trained Ridge and Random Forest only within historical one-month backtests and produced station-level relative summaries; it did not create an operational/current forecast, interpolated surface, or sensor map. Uganda code, audio, outputs, and model artifacts remain intact as a separate legacy component. Do not label its map or pilot metrics as Chennai/SRMIST results.

The term *monthly record* below means a row indexed by source-provided year and month. It does not imply an exact sampling timestamp, full monthly coverage at every hour, or a continuous sensor stream.

## Dataset comparison

| Source | Verified coverage and contents | Access / reuse | Fit and limits |
|---|---|---|---|
| [CPCB Noise Monitoring Data in India (2011-2018), Kaggle](https://www.kaggle.com/datasets/rohanrao/noise-monitoring-data-in-india) | Downloaded files: `station_month.csv` (5,005 rows) and `stations.csv` (70 stations). The monthly file has station ID, year, month, `Day`, `Night`, `DayLimit`, and `NightLimit`; station metadata has station name, city, state, and type. Coverage is February 2011-December 2018. Chennai has 715 rows at ten stations: five have complete monthly coverage from February 2011-December 2018 (95 months each), and five from January 2015-December 2018 (48 months each). No Chennai values are missing in `Day`/`Night`; across the full file there are 3 missing `Day` and 10 missing `Night` values. No coordinates are supplied. | Kaggle lists CC0 and credits CPCB as the original public source. The original CPCB Envi URL cited by the Kaggle description could not be reached during this audit, so the copy and license metadata were verified through Kaggle, not independently against an active CPCB download page. | **Best accessible Chennai time-series candidate**, restricted to monthly station comparisons/trends. Keep station values, IDs, and source-supplied categories; do not infer coordinates, daily readings, exact timestamps, or an hourly series. |
| [TNPCB Annual Report 2024](https://tnpcb.gov.in/PDF/Information/annualreports/English/AnnualRpt_Eng2024.pdf), Annexures XVIII A/B | Ten Chennai stations, daytime/nighttime summaries for April 2023-March 2024. The table reports average, maximum, and minimum Leq dB(A), station zone, and day/night limits. Extracted rows are preserved in [`data/external/india/tnpcb_2023_24_chennai_annual_noise.csv`](data/external/india/tnpcb_2023_24_chennai_annual_noise.csv). | Official public PDF; no separate open-data license was identified in the inspected report. The extracted file cites the source and period, but that citation should not be read as a license grant. | A separate annual context/checkpoint only. It is not timestamped raw data and must not be concatenated with the 2011-2018 CPCB monthly series. No coordinates appear in these annexure tables. |
| [SRMIST Air and Noise Monitoring Report](https://webstor.srmist.edu.in/web_assets/downloads/2025/air-and-noise-monitoring.pdf) | Reports one ambient location near Tech Park Tower I, January 2025: daytime 48.3 dB(A), nighttime 41.7 dB(A), against 55/45 limits. The values are preserved in [`data/external/india/srmist_january_2025_ambient_noise.csv`](data/external/india/srmist_january_2025_ambient_noise.csv). | Official SRMIST PDF. No separate dataset license was identified. | A one-off campus reference only. The report does not supply an exact timestamp, coordinates, repeated observations, or a local series. Do not confuse its separate generator-set test with ambient campus noise. |
| [IIT Madras/TNPCB Chennai real-time noise mapping study](https://cms_tnccm.cloudbases.in/uploads/1_Report_Real_Time_Noise_Monitoring_f1523806af.pdf) | The report describes 60 locations across 15 Chennai zones, continuous day/night monitoring, and one-minute data transmission. It discusses March-December 2025 maps and analyses. | The original PDF URL returned 503; an archived copy was inspected. The report explicitly prohibits reproducing/redistributing its contents without written permission. It says the dashboard supports real-time/historical viewing and downloads, but supplies no dashboard URL or API. No coordinates or station tables from this report are copied into this project. | High-value potential source, but not a usable project dataset until access and permission are established. A screenshot inside the report shows a January 2026 timestamp and a sensor last updated 33 days earlier; that is not evidence that a live feed is currently accessible. |
| TNPCB public noise page and dashboard lead | [TNPCB Noise page](https://tnpcb.gov.in/noise.php) is titled “Noise Standards” and provides a standards document, not sensor readings. A direct request to `dashboard.php` returned 404 in this audit. | Official pages checked 2026-11-01. | No current official noise data API, download, or live reading endpoint was verified through these pages. This does not prove that no private or unlinked endpoint exists. |
| TUMI Datahub / other discovery | The supplied TUMI dataset page was unavailable during this audit. Search indexing surfaced a TUMI Chennai entry pointing back to Kaggle, rather than a separate underlying series. Bounded GitHub, Zenodo, and data.gov.in searches did not identify another verified downloadable Kattankulathur time series; data.gov.in returned 403. | Search results are leads, not verified data access or license evidence. | No additional independently verified local observations were added. |

## Verified files and interpretation

### CPCB-attributed monthly data

The two unchanged Kaggle CSVs are stored under [`data/external/india/cpcb_noise_2011_2018/`](data/external/india/cpcb_noise_2011_2018/). The Chennai station IDs and source names are:

| Station ID | Source name | Monthly rows |
|---|---|---:|
| CHE01 | Eye Hospital | 95 |
| CHE02 | Guindy | 95 |
| CHE03 | Perambur | 95 |
| CHE04 | T. Nagar | 95 |
| CHE05 | Triplicane | 95 |
| CHE06 | Pallikaranai | 48 |
| CHE07 | Velachery | 48 |
| CHE08 | Washermanpet | 48 |
| CHE09 | Anna Nagar | 48 |
| CHE10 | Sowcarpet | 48 |

The file columns are preserved as supplied. `Year` and `Month` identify the source's monthly period, not an exact measurement timestamp. The public file does not describe the measurement aggregation/calibration procedure in sufficient detail to independently establish it here; preserve the source field names and avoid stronger claims than the source documentation supports.

Station classifications differ between the monthly Kaggle metadata and the TNPCB annual report for some station names (for example, T. Nagar, Velachery, and Sowcarpet). Retain source-specific classifications and standards; do not silently harmonize or join those labels.

### Chennai annual summary and SRMIST reference

The TNPCB CSV contains 20 rows: two day/night rows per station, with the original report period, reported average/minimum/maximum Leq dB(A), zone limits, annexure, and source URL. The SRMIST CSV contains two rows: day/night values and the limits printed in the report. Blank timestamp and coordinate fields are intentional: the source did not provide them. Neither file should be treated as a time series.

## Current 60-site lead and access status

The report's dashboard claim is a lead, not a verified public data endpoint. TNPCB's public noise page reviewed here contains standards only. A 2026-02-18 *New Indian Express* article reports release of IIT Madras/TNPCB findings from 60 sites, while a 2026-09-27 *Times of India* article says the same stations were to continue monitoring and capture Diwali noise. As of this research snapshot, no post-Diwali downloadable station-level series or API was verified.

Before using that network, request from TNPCB/IIT Madras:

1. The dashboard URL and any documented API/export route, including archived 2025 data and the 2026 extension.
2. Written permission/licensing terms for analysis, derived charts, redistribution, and use of station coordinates.
3. A data dictionary covering timestamps/time zone, aggregation windows, dB metric/weighting, calibration, station IDs, maintenance, missingness, and validation flags.

References: [New Indian Express, 2026-02-18](https://www.newindianexpress.com/states/tamil-nadu/2026/Feb/18/road-traffic-drives-chennais-rising-noise-levels-iit-m); [Times of India, 2026-09-27](https://timesofindia.indiatimes.com/city/chennai/60-sensors-to-track-noise-pollution-across-chennai-this-diwali/articleshow/134523840.cms).

## Feasibility decision

- **Chennai monthly analysis:** feasible for exploratory station-level trends, seasonal comparison, and retrospective evaluation at monthly resolution, subject to method/provenance caveats. Use chronological evaluation and simple baselines before making predictive claims.
- **Chennai mapping:** not yet supported by the CPCB monthly files because station coordinates are absent. Do not guess locations from station names. The 60-site report's coordinates are not reused pending permission.
- **Kattankulathur/SRMIST time series or forecast:** not supported by the verified campus evidence: only one January 2025 day/night reference was found. Report a local data gap rather than extrapolating Chennai-wide or Ugandan data.
- **India-wide comparison:** possible as a separate comparative analysis of the available cities, but not as a substitute for local campus data. Preserve city/station structure and source limitations.
- **Legacy Uganda workflow:** retain code, audio, map, and pilot metrics separately. They do not validate or represent Indian observations.

## References and provenance

- CPCB-attributed monthly dataset: [Kaggle dataset page](https://www.kaggle.com/datasets/rohanrao/noise-monitoring-data-in-india); the card credits CPCB's former Envi site, which was unreachable in this audit.
- SRMIST source: [Air and Noise Monitoring, January 2025](https://webstor.srmist.edu.in/web_assets/downloads/2025/air-and-noise-monitoring.pdf).
- TNPCB annual source: [Annual Report 2024](https://tnpcb.gov.in/PDF/Information/annualreports/English/AnnualRpt_Eng2024.pdf), Annexures XVIII A/B.
- IIT Madras/TNPCB report: [original report URL](https://cms_tnccm.cloudbases.in/uploads/1_Report_Real_Time_Noise_Monitoring_f1523806af.pdf); the original endpoint returned 503, and an archived copy was used for inspection. Report terms restrict reuse.
- TNPCB page checked: [Noise Standards](https://tnpcb.gov.in/noise.php).
- TUMI lead supplied for research: [Noise Monitoring Data in India, 2011-2018](https://hub.tumidata.org/dataset/noise_monitoring_data_in_india_2011_2018_delhi); unavailable directly during this audit.
