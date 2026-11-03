# India Dataset Audit — Phases 2 and 3

**Audit date: 2026-11-01 | Project scope: Chennai data validation, modelling, and SRMIST/Kattankulathur feasibility**

## Executive decision

The downloaded CPCB-attributed/Kaggle data form a clean **monthly table** for ten Chennai station labels. Phase 3 has implemented station-level relative hotspot behaviour and historical baseline backtesting. The data still do not contain coordinates or a verified definition of the `Day`/`Night` value and its aggregation. They do not support a reliable geospatial hotspot map, a live system, or a current Chennai/SRMIST forecast.

The TNPCB 2023–24 annual summaries and the January 2025 SRMIST reference are different products and must remain separate from that table. There is no verified, public, programmatic feed for the 60-site IIT Madras/TNPCB study in the source materials checked. Written data-use permission and the dashboard/API details are required before using its restricted report content or data.

Phase 3 models were run only as historical baseline experiments; see [FORECASTING_BASELINE_REPORT.md](./FORECASTING_BASELINE_REPORT.md). They are not current forecasts.

## Files and provenance checked

| File | Actual contents | SHA-256 / size | Audit result |
|---|---|---|---|
| [`station_month.csv`](data/external/india/cpcb_noise_2011_2018/station_month.csv) | 5,005 rows; columns `Station, Year, Month, Day, Night, DayLimit, NightLimit`; dates February 2011–December 2018. | 131,420 bytes; `79335ddcc15d50a6c58c904d75b63df2ee3e8672939f0d87b67dd43e87b78b64` | Kaggle distribution of CPCB-attributed monthly noise data; no latitude/longitude or timestamp. |
| [`stations.csv`](data/external/india/cpcb_noise_2011_2018/stations.csv) | 70 rows; columns `Station, Name, City, State, Type`; Chennai has ten station IDs. | 3,318 bytes; `001c662b772b4a1664ae4442761fe0969ca50512749fa85e673530da93d67c6f` | No addresses or coordinates. |
| [`tnpcb_2023_24_chennai_annual_noise.csv`](data/external/india/tnpcb_2023_24_chennai_annual_noise.csv) | 20 rows: day/night annual-summary rows for ten Chennai station names, April 2023–March 2024. | 3,334 bytes; `ccdc3e218a599001a830615503df9d38dc4002ce383c7e9e350798ee757dcf74` | Manually transcribed from Annexures XVIII A/B of the official TNPCB annual report; includes report period, averages, minima/maxima, category limits, annexure, and source URL. |
| [`srmist_january_2025_ambient_noise.csv`](data/external/india/srmist_january_2025_ambient_noise.csv) | Two rows: one Day and one Night value for the ambient location near Tech Park Tower I. | 610 bytes; `5e1d8d8256038c0c876b91a4062f5ca03a1df32a0f916ff2cfd624872deda5f4` | Transcribed from the official SRMIST report. Exact sample times and coordinates are not supplied. |

The Kaggle page lists CC0 and credits CPCB as the original source. The currently cited CPCB Envi page could not be resolved directly in this audit, so the data copy and the Kaggle license statement are verified; the original CPCB endpoint and its metadata were not independently revalidated. Retain the source attribution and these hashes when refreshing the downloaded files.

## CPCB-attributed table: row-level validation

- The complete distributed files contain 5,005 monthly records and 70 unique station IDs. The source covers eight city labels and February 2011–December 2018.
- Chennai filter: 715 rows across ten station IDs. All ten have contiguous month sequences through their observed spans; there are no duplicate station-month keys.
- Chennai `Day`, `Night`, `DayLimit`, `NightLimit`, station, year, and month fields are complete for all 715 rows. No observations were imputed.
- Across the full 5,005-row source, `Day` has 3 missing values and `Night` has 10. The full file has no duplicate records or duplicate station-month keys.
- The station table does not provide an explicit unit/weighting/Leq field for `Day` and `Night`. The dataset is described as decibel/noise readings, but this CSV alone does not establish a measurement definition, aggregation procedure, calibration, or daily sensor completeness. Do not silently relabel these values `dB(A)`, `Leq`, or 24-hour averages.
- `DayLimit`/`NightLimit` and `Type` exist, but categories and limits do not consistently match the TNPCB annual report's zone labels. Do not use either classification as a validated, harmonized label without resolving the source discrepancy.

The five longer series run February 2011–December 2018 (95 months each); the five later series run January 2015–December 2018 (48 months each). See the generated [station coverage CSV](outputs/india/eda/chennai_station_coverage.csv) and [station summary CSV](outputs/india/eda/chennai_station_descriptive_summary.csv).

## Station locations: exact versus approximate

No exact sensor-installation coordinates were found in either Kaggle CSV or the reviewed public CPCB/TNPCB station summaries. The source identifies monitoring stations by abbreviated/locality names, not an address, site plan, or GPS coordinate. OpenStreetMap Nominatim was queried individually on 2026-11-01 for the place labels. Those results locate named facilities/neighbourhoods, not the noise sensors. The exact coordinate columns in [`chennai_station_coordinates.csv`](data/external/india/chennai_station_coordinates.csv) are therefore blank for every station.

Eight approximate place/facility representations are recorded in separate `approximate_latitude`/`approximate_longitude` columns; they are explicitly not monitoring points. Washermanpet returned rail-station features rather than a verified noise site/locality point, and Anna Nagar returned conflicting features; those approximate fields remain blank. OpenStreetMap data are © OpenStreetMap contributors, ODbL; the file records each query and caveat. The 60-site IIT Madras/TNPCB report contains location details but prohibits reproduction without written permission; none of those report coordinates were copied.

| ID | Name in station table | Approximate-place result | Monitor coordinate |
|---|---|---|---|
| CHE01 | Eye Hospital | Egmore Eye Hospital facility point: 13.0703627, 80.2600948 | Unresolved |
| CHE02 | Guindy | Guindy locality feature: 13.0087095, 80.2203646; other Guindy result(s) exist | Unresolved |
| CHE03 | Perambur | Perambur suburb feature: 13.1121242, 80.2450220; a separate rail-station feature also exists | Unresolved |
| CHE04 | T. Nagar | Thiyagaraya Nagar suburb feature: 13.0385946, 80.2335793 | Unresolved |
| CHE05 | Triplicane | Triplicane suburb feature: 13.0604285, 80.2750704 | Unresolved |
| CHE06 | Pallikaranai | Pallikaranai suburb feature: 12.9304748, 80.2077692 | Unresolved |
| CHE07 | Velachery | Velachery suburb feature: 12.9772611, 80.2209183; a separate MRTS station feature also exists | Unresolved |
| CHE08 | Washermanpet | Rail-station results only; not a verified monitoring point or neighbourhood centroid | Unresolved |
| CHE09 | Anna Nagar | Conflicting locality/administrative results | Unresolved |
| CHE10 | Sowcarpet | Sowcarpet neighbourhood feature: 13.0884967, 80.2783092 | Unresolved |

The approximate points may be used only as labelled locality context. They are **not safe inputs to a sensor-level hotspot map**, spatial interpolation, or estimates of the measurement site. Map feasibility remains **no** until station-site coordinates are confirmed by CPCB/TNPCB or another reliable station-specific source.

## Cross-source comparability

The TNPCB annual report reports annual average/minimum/maximum **Leq dB(A)** in day and night annexures for April 2023–March 2024. The Kaggle copy has integer `Day`/`Night` columns and month/year, but does not state the metric, weighting, or aggregation method in its schema. Therefore:

- The names mostly correspond, with `Eye Hospital Egmore` abbreviated as `Eye Hospital` and `Annanagar` written as `Anna Nagar`; this does not prove that the exact monitor positions are identical.
- Category matches exist for Guindy (Industrial), Perambur (Commercial), Pallikaranai (Commercial), Triplicane (Residential), and Washermanpet (Commercial). Other labels differ: `Silence` versus `Sensitive` at Eye Hospital and Anna Nagar; Commercial versus Sensitive at T. Nagar; Residential versus Commercial at Velachery and Sowcarpet.
- The 2011–2018 rows are monthly, while the TNPCB table is an annual summary. The time windows do not overlap, and there is a multi-year gap.
- The unit/metric/aggregation and exact site identity are not demonstrably compatible. **Do not concatenate, calibrate against, or train jointly on these products.** Present the annual report as an independently sourced later reference only.

The SRMIST report supplies a single January 2025 campus ambient reference near Tech Park Tower I: Day 48.3 and Night 41.7 dB(A), with stated limits 55/45. It is not a repeated series, a campus-wide sample, or an exact geolocated monitor record. It must not be used as a model training set or merged into the Chennai station panel.

## Public data access investigation

Checked on 2026-11-01:

- [TNPCB Noise page](https://tnpcb.gov.in/noise.php) presents noise standards, not a sensor-reading portal.
- Direct request to the previously surfaced [TNPCB dashboard URL](https://tnpcb.gov.in/dashboard.php) returned HTTP 404. The search-indexed [TNPCB `aqidatachn.php` route](https://tnpcb.gov.in/aqidatachn.php) returned HTTP 404 as well and is an air-quality route, not verified noise data.
- The CPCB Envi noise-quality page cited by Kaggle appeared in search indexing as a historical overview of the National Ambient Noise Monitoring Network, but its hostname could not be resolved by direct request. No active CPCB noise download/API endpoint was verified.
- The official CPCB [2015 Diwali noise assessment PDF](https://cpcb.gov.in/displaypdf.php?id=bm9pc2VfZGF0YS9QcmVzc1JlbGVhc2VOb2lzZURpd2FsaTIwMTUucGRm) is an event-specific historical report, not a current time-series endpoint.
- The IIT Madras/TNPCB study says that its dashboard supports current/historical viewing and downloads. Its cited report does not include a dashboard URL, public API documentation, data dictionary, or a public-access/reuse policy. The report's reproduction restriction means permission is needed before copying or redistributing its data, coordinates, maps, or detailed tables.
- The study's described March–December 2025 record and 2026 extension are not available here as an authorized raw CSV/XLSX/API export. A reported dashboard/download function does not establish a public endpoint or permission; obtain the official link and access terms directly from TNPCB/IIT Madras.
- News reporting in February 2026 described release of the 60-site study, and reporting in September 2026 described continued monitoring/Diwali observations; neither article supplied an authorized raw-data download or API. See [New Indian Express](https://www.newindianexpress.com/states/tamil-nadu/2026/Feb/18/road-traffic-drives-chennais-rising-noise-levels-iit-m) and [Times of India](https://timesofindia.indiatimes.com/city/chennai/60-sensors-to-track-noise-pollution-across-chennai-this-diwali/articleshow/134523840.cms).
- Searches of the public TNPCB/CPCB entry points and bounded dataset leads did not verify a downloadable Tamil Nadu city/station/month noise dataset for 2021 or 2022. This is a negative search result only, not confirmation that such records do not exist; ask TNPCB about archived records in the access request.
- Bounded official-domain searches surfaced the SRMIST January 2025 air/noise report but no independently verifiable additional SRMIST noise series. This does not prove no internal records exist.
- The supplied TUMI Datahub page was unavailable, and bounded searches did not identify another independently verifiable downloadable Tamil Nadu noise series. These searches are leads, not proof that no additional records exist.

Thus, the only programmatically downloaded multi-year Chennai noise file verified here is the static Kaggle-hosted monthly table ending in 2018. No current/live Chennai or SRMIST noise API has been verified. Search results and news articles are leads, not substitute measurement files.

### Day/Night source definition and reuse conditions

The downloaded schema, Kaggle description, and reviewed CPCB public noise material do not establish whether the monthly `Day` and `Night` columns are Leq, Lday/Lnight, an arithmetic average of readings, or another aggregation; the CSV also does not specify weighting, interval, or sensor completeness. CPCB noise standards and event-specific assessment reports are not source documentation for the monthly table's calculation. Keep neutral `Day`/`Night` labels and report values in source-value units until the CPCB source metadata or an authoritative data dictionary confirms otherwise.

Kaggle's dataset listing identifies CPCB as the original source and lists CC0 for its distributed copy. The TUMI page could not be independently retrieved in this audit, and the original CPCB endpoint/terms remain unverified. Retain CPCB attribution and cite the Kaggle distribution; do not interpret a third-party listing as confirmation of terms governing any original or restricted source material.

## Phase 3/4 analysis and system outputs

- The Chennai lagged station-month panel, forecasts, aggregate/per-station metrics, chronological split table, and plots are under `outputs/india/models/`.
- Relative Day/Night station behaviour summaries and station comparison plots are under `outputs/india/`.
- Other-city and station descriptive comparisons are under `outputs/india/comparative/`.
- `historical_monthly_noise_events.jsonl` contains ordered year-month records explicitly marked historical and not live. It does not create timestamps.
- The modeling conclusions and actual MAE/RMSE are in [FORECASTING_BASELINE_REPORT.md](./FORECASTING_BASELINE_REPORT.md). Do not combine these outputs with TNPCB annual or SRMIST records.
- `outputs/india/system/` contains the SQLite replay store, event-based transition/recurrence outputs, historical January 2019 persistence extension, source-comparability ledger, local dashboard, and system status manifest. The January 2019 period is merely the next historical month after the 2011–2018 dataset and is not a present-day forecast.
- `separate_reference_source_values.csv` displays, without pooling, the audited TNPCB annual Leq dB(A) values and SRMIST report-period dB(A) pair. Values retain each source's separate measurement definition and dates.

### Phase 4 source availability re-check

Checked on 2026-11-01:

- The official [TNPCB Noise page](https://tnpcb.gov.in/noise.php) returns the Board's contact information and noise-standards material; no reading export or sensor API is documented there.
- The official [CPCB noise rules page](https://cpcb.nic.in/noise-pollution/) describes the Noise Pollution (Regulation and Control) Rules/standards, not a downloadable monitoring time series.
- The official [SRMIST January 2025 air/noise report](https://webstor.srmist.edu.in/web_assets/downloads/2025/air-and-noise-monitoring.pdf) is fetchable. It is the same one-report-period Day/Night reference already audited; no additional repeated Kattankulathur noise series was verified.
- The IIT Madras/TNPCB report link remains unavailable with a 503 response in this environment. No official authorized raw CSV/XLSX export, documented API, data dictionary, or reuse terms were located. Do not scrape dashboard routes or bypass authorization.
- Publicly checked pages and dataset leads did not yield a verified 2021/2022 Tamil Nadu city/station/month noise file. This is not proof that archived records do not exist; the request draft asks the data owner.
- No newer authorized observations were obtained. The comparison pipeline uses no newly acquired IIT/TNPCB observations and preserves TNPCB/SRMIST files as separate contextual sources.

### Formal data-access request draft

The standalone [DATA_ACCESS_REQUEST_DRAFT.md](./DATA_ACCESS_REQUEST_DRAFT.md) requests the 2025/2026 60-site records, exact station metadata/coordinates, temporal intervals, measurement/unit definitions, calibration/QA methods, reuse terms, and possible 2021/2022 archives. It also contains a separate request for repeated SRMIST ambient measurements. No email or form has been sent.

**To:** `tnpcb-chn@gov.in`; `memsec@tnpcb.gov.in`  
**Subject:** Request for authorized access to Chennai noise-monitoring historical records and station metadata

> Dear Tamil Nadu Pollution Control Board,
>
> I am working on an academic AIoT urban-noise project focused on Chennai, with SRM Institute of Science and Technology (Kattankulathur) as the local reference area. I reviewed the public CPCB-attributed monthly dataset and the IIT Madras/TNPCB Chennai noise-monitoring study. I have not copied or redistributed the study's restricted tables, maps, or coordinates.
>
> Could you please advise whether the Board can provide authorized access to the historical records from the 60-site Chennai monitoring study (including the 2025 observation period and any subsequent extension), or direct me to the official dashboard/API and its access process?
>
> If access is available, I would be grateful for:
>
> 1. Download/API instructions and the available date range, including whether timestamped station-level records can be exported.
> 2. Written data-use terms covering academic analysis, derived charts, publication, and any redistribution restrictions.
> 3. Station IDs, exact sensor coordinates or addresses, installation/relocation history, and site category.
> 4. A data dictionary for timestamps/time zone, reporting interval, Leq/Lmax/Lmin/L10/L50/L90 fields, dB weighting, calibration/validation flags, missingness, and maintenance.
> 5. Any required acknowledgement, citation, permission form, data-sharing agreement, or institutional approval.
>
> The data would be kept separate from other sources unless measurement definitions, station identity, time aggregation, and reuse permission are shown to be compatible. I will not represent historical observations as live readings or make campus-specific predictions from citywide data.
>
> Thank you for your guidance.
>
> Sincerely,  
> `[Student name]`  
> `[Course / department / institution]`  
> `[Supervisor and institutional contact, if applicable]`

No request has been sent. The report does not establish that the portal requires a login; it establishes that the public route and access rules are not disclosed in the material verified here. Seek the Board's directions rather than probing undocumented endpoints.

## Audit conclusion

The monthly source is useful for a carefully bounded Chennai station/time analysis after its metric definition is clarified. It is inadequate for a reliable geographic hotspot surface, a current/live forecast, or Kattankulathur campus forecasting. Obtain written data access, exact site coordinates, metric definitions, and repeated local observations before building spatial or local predictive components.

The Phase 5 Streamlit interface presents these audited outputs without relaxing their source limitations. Its eight pages use generated artifacts, keep SRMIST/TNPCB sources separate, and label replay as historical simulation; see [README](./README.md) for the local launch command.
