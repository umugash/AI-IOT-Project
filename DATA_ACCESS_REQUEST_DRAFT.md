# Draft Request for Authorized Noise-Monitoring Data

**Status:** Draft only. Do not send without the student's explicit approval.  
**Prepared:** 2026-11-01  
**Project:** AIoT-Based Urban Noise Hotspot Behaviour Modeling and Spatiotemporal Prediction

## Request to TNPCB and IIT Madras

**To:** `tnpcb-chn@gov.in`; `memsec@tnpcb.gov.in`  
**Suggested subject:** Academic request for authorized Chennai noise-monitoring records, metadata, and reuse conditions

Dear Tamil Nadu Pollution Control Board and IIT Madras project team,

I am conducting an academic AIoT project on urban noise behaviour. Chennai is the primary study region, SRMIST Kattankulathur is a separate local reference, and other Indian cities are used only for comparative analysis. I have reviewed the publicly accessible TNPCB noise standards page and the publicly described IIT Madras/TNPCB 60-site monitoring study. I have not accessed a restricted dashboard or copied restricted study data, maps, or coordinates.

Could you please advise whether authorized access can be granted to the station-level historical observations from the Chennai network, including the reported March–December 2025 measurements and any observations collected during the 2026 extension? If a public dashboard/export or API exists, please share its official URL and documentation and confirm the applicable access and reuse rules.

For any available data, I request the following information or documentation:

1. **Observations and coverage**
   - Historical station-level records for 2025 and records available during 2026.
   - Any official Tamil Nadu city/station/month noise records for 2021 and 2022, including Chennai, with source citations and coverage.
   - Original reporting interval, whether observations are raw or aggregated, and any missing periods.
2. **Station identity and geography**
   - Stable station/sensor IDs and station-name crosswalks.
   - Exact latitude/longitude or addresses, coordinate reference system, stated location accuracy, and permission to use coordinates in derived academic visualizations.
   - Installation, relocation, downtime, and sensor-to-station identity history.
3. **Measurement definitions**
   - Data dictionary and units for every field.
   - Whether reported fields include Leq (and its interval), Lday/Lnight, Lmax/Lmin, statistical levels, or another measure.
   - dB frequency weighting, day/night time windows, averaging/aggregation formula, sample interval, time zone, and daylight/time-boundary convention.
4. **Quality and methodology**
   - Instrument and microphone/sensor make/model, placement/height, siting protocol, calibration procedure/frequency, maintenance, quality-control flags, and exclusion rules.
   - Availability of calibration logs, field validation, sensor uptime, and known measurement limitations.
5. **Access and reuse**
   - Download/API format, authentication requirements, rate limits, and any data agreement or institutional approval required.
   - Written terms for academic analysis, storage, derived statistics/plots, publication/presentation, and redistribution. Please specify whether raw data, station coordinates, maps, or screenshots may be shared.
   - Required citation, acknowledgement, and attribution text.

The existing historical analysis uses a separate CPCB-attributed monthly file ending in 2018 whose Day/Night measurement definition is unresolved. I will not merge it with these records unless the measurement definitions, temporal aggregation, station identity, and permissions are demonstrated to be compatible. I will not report historical replay as live sensing or infer geographic locations.

Thank you for advising whether these materials can be shared and what formal process should be followed.

Sincerely,  
`[Student name]`  
`[Course / department / institution]`  
`[Supervisor name and institutional contact, if applicable]`  
`[Student/institution email and phone]`

## Separate request to SRMIST

**Recipient:** SRMIST office responsible for environmental monitoring / sustainability / campus facilities (please direct to the appropriate authorized office).  
**Suggested subject:** Academic request for repeated ambient-noise monitoring records at Kattankulathur

Dear SRMIST team,

I am using SRMIST Kattankulathur as a local reference in an academic urban-noise project. The publicly available January 2025 Air and Noise Monitoring Report provides a Day/Night ambient reference near Tech Park Tower I, but does not provide exact observation times, coordinates, or a repeated time series.

Could SRMIST advise whether repeated ambient-noise measurements for Kattankulathur are available for academic use? If so, please provide or document:

- Actual observation timestamps and time zone, monitoring intervals, site names/IDs, and exact coordinates or location accuracy.
- Metric definitions, units, dB weighting, day/night windows, aggregation method, instrument placement, calibration, quality controls, and missing-data flags.
- Coverage dates, station relocations, and whether values are ambient observations rather than equipment/source tests.
- Written permissions and attribution requirements for analysis, derived charts, presentations, and data storage.

I will keep any campus observations separate from Chennai city monitoring data unless source definitions and station identity are proven compatible. The January 2025 report pair will remain a reference only and will not be used as a training series.

Sincerely,  
`[Student name]`  
`[Course / department / institution]`  
`[Supervisor name and institutional contact, if applicable]`

## Verification notes

- TNPCB's public [Noise page](https://tnpcb.gov.in/noise.php) was checked on 2026-11-01; it presents noise standards, not an observation download.
- The [SRMIST January 2025 report](https://webstor.srmist.edu.in/web_assets/downloads/2025/air-and-noise-monitoring.pdf) is publicly fetchable and is already represented by a two-row reference CSV. No additional repeated SRMIST noise series was verified.
- The IIT Madras/TNPCB study describes a monitoring network, but no authorized public raw-data export, documented API, data dictionary, or reuse terms were verified in the material available to this project.
- This file is a draft. No message has been sent and no restricted dashboard was accessed.
