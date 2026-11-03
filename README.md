# AIoT Urban Noise Hotspot Behaviour Modelling

> **Current project status (2026-11-01):** India Phase 5 adds a local Streamlit interface to the Phase 4 historical replay and analysis system. The application is not live monitoring or a current forecast. Exact Chennai sensor coordinates and the CPCB-attributed Day/Night measurement definitions remain unresolved. The Uganda prototype remains separate. See [final scope](./FINAL_PROJECT_SCOPE.md), [forecasting report](./FORECASTING_BASELINE_REPORT.md), and the [draft data-access request](./DATA_ACCESS_REQUEST_DRAFT.md).

The existing software-only implementation uses the public Urban Noise Uganda dataset. Its Uganda audio, map, and model metrics are retained as a separate supplementary component; they are not Indian observations. No microphone, Arduino, Raspberry Pi, Kafka broker, cloud account, or other hardware/service is required for the legacy local workflow.

## Research scope and data integrity

The public [Figshare archive](https://doi.org/10.6084/m9.figshare.30168901.v1) provides audio, source categories, coordinates, GPS accuracy, and the dataset-reported noise measurement. The downloaded archive is approximately 2.97 GB.

Important legacy feasibility finding: the archive's `metadata.csv` has 61,685 rows and **does not contain a timestamp**. Although the dataset landing page says 61,821 records and a Hugging Face card advertises a timestamp field, these disagree with the archive inspected here. Hugging Face files require an account/access agreement; their rows could not be inspected anonymously. The Uganda implementation does not fabricate times or claim temporal forecasting. Its verified scope is source classification plus retrospective geographic noise mapping. Temporal behaviour, forecasting, and timestamp-ordered replay will fail explicitly until a real timestamped source is provided and audited. See [DATASET_FEASIBILITY.md](./DATASET_FEASIBILITY.md) for the historical Uganda assessment and the India reports above for current scope.

The metadata's `noise_measurement` field is described as SPL by the dataset record/card. Its CSV does not state the unit, weighting, or calibration procedure. Outputs call it the *dataset-reported noise measurement* and do not claim independent SPL calibration or legal-limit compliance. The Hugging Face dataset card also distinguishes hand-labelled Kampala observations from model-predicted Entebbe labels. Legacy classification training/evaluation should use Kampala only unless annotation provenance is independently confirmed.

## Legacy Uganda workflow

### 1. Install

Python 3.10+ is recommended. From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Download and validate the public dataset

```powershell
python scripts\download_dataset.py
```

This downloads the official Figshare ZIP, verifies its published MD5, and extracts only the actual metadata CSV and OGG recordings (ignoring macOS sidecars). The full archive is about 2.97 GB. To obtain and validate only the 5.6 MB metadata CSV without downloading the archive, use bounded HTTP range reads:

```powershell
python scripts\download_dataset.py --metadata-only
```

To supply the extracted audio/metadata in another location, pass `--archive` and `--data-dir`. The data are licensed CC BY 4.0; cite the dataset and article as listed in the dataset report.

### 3. Analyze real metadata and create a hotspot map

```powershell
python scripts\analyze_dataset.py
python scripts\create_hotspot_map.py
```

The analysis writes an audit JSON and plots under `outputs\`. The map aggregates observations into configurable metric grid cells, screens out GPS accuracy estimates above 100 m by default, requires a configurable minimum sample count, displays the number of observations, and ranks intensity using the reported measurement. These are exploratory sample maps, not an unbiased city-wide exposure surface.

### 4. Train and run acoustic classification

```powershell
python scripts\train_classifier.py --epochs 5 --max-samples 4000
python scripts\predict_audio.py data\raw\audio_files\RECORDING.ogg
```

Use the real `filename` values in `metadata.csv`; training filters to Kampala labels by default. Training compares a compact PyTorch CNN on log-Mel features with a Random Forest baseline on MFCC summary features, using the same geographic group split. `--max-samples` is a development limit; report it alongside any resulting metrics. For a full run omit it only after confirming available compute and disk space.

### 5. Timestamp-dependent operations

```powershell
python scripts\replay_iot.py
```

With the verified public Figshare metadata this intentionally reports that chronological replay is unavailable because no timestamp column exists. The script accepts a future audited file with an actual `timestamp` column and then replays rows in timestamp order as a software simulation. It never labels historical replay as live sensor acquisition.

## Components

- `src\urbannoise/dataset.py`: schema validation and metadata audit.
- `src\urbannoise/spatial.py`: GPS-accuracy-aware metric grid aggregation and hotspot flags.
- `src\urbannoise/temporal.py`: temporal transitions only for actual timestamped data; no timestamp synthesis.
- `src\urbannoise/audio_model.py`: OGG loading, log-Mel/MFCC extraction, and PyTorch CNN.
- `src\urbannoise/training.py`: group-aware CNN/baseline training and classification metrics.
- `src\urbannoise/iot.py`: timestamp-validated simulated replay/event envelopes.
- `scripts/`: acquisition, analysis, map creation, training, prediction, and replay commands.

## India Phase 2 outputs

The Chennai monthly data audit and visualizations are separate from the Uganda workflow:

```powershell
.\.venv\Scripts\python.exe scripts\analyze_india_eda.py
```

This writes station coverage/statistics tables and PNG charts under `outputs\india\eda\`. It performs descriptive analysis only; it does not train a model, infer observation timestamps, or geocode sensor locations.

## India Phase 3 analysis

The approved Chennai/India analysis generates the feature panel, historical baseline backtests, relative station-level hotspot summaries, comparative city tables, charts, and explicitly historical JSONL events:

```powershell
.\.venv\Scripts\python.exe scripts\run_india_analysis.py
```

Backtests use one-month-ahead expanding origins over 2017–2018, with Day and Night kept separate. The dataset ends in 2018 and does not verify units or aggregation; do not describe the metrics as present-day forecasts or verified dB(A). The hotspot analysis is station-based only, not a sensor map or interpolated geographic surface. Output details are in [FORECASTING_BASELINE_REPORT.md](./FORECASTING_BASELINE_REPORT.md).

## India Phase 4 complete local system

Run the end-to-end pipeline from the project root:

```powershell
.\.venv\Scripts\python.exe scripts\run_india_system.py
```

The command regenerates EDA, the data panel, historical backtests, hotspot behaviour outputs, comparative tables, and JSONL replay; validates and ingests the 5,005 replay events into a local SQLite store; writes a one-step persistence estimate for the period following the last observation (currently 2019-01); and creates a local HTML intelligence dashboard at `outputs\india\system\index.html`. The estimate is explicitly historical, not current. Its empirical hotspot-frequency proxy and residual-based range are descriptive/heuristic—not calibrated probability or confidence bounds. SRMIST and TNPCB products remain separate.

The local adapter implements virtual-source → JSONL → schema validation → SQLite ingestion → batch analysis/inference. The current laptop has Java but no Kafka broker, Spark/PySpark, or Delta Lake runtime. Those distributed components are not installed or claimed as implemented; for 5,005 monthly records they add operational complexity without scientific or performance benefit. SQLite is the local durable store; an authorized larger/finer-grained source can later be integrated behind the same event contract.

## Streamlit research application

The presentation layer is a separate Streamlit app that reuses the India processing outputs:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Eight sidebar sections cover overview, Chennai behaviour, hotspot intelligence, predictive analytics, India comparisons, controlled local replay, the separate SRMIST case study, and methodology/data sources. Plotly filters and tables work with the available CSV/JSON outputs; replay has a user-triggered limited-batch/full-dataset control and writes through the existing idempotent SQLite event store. The app does not show a geographic heatmap, live sensor feed, or current forecast. It reports source-quality limitations and explicitly marks the January 2019 extension as historical.

The SRMIST and TNPCB products are shown as separate references, never as Chennai model training data. The unsent data-access request draft is accessible from the Methodology & Data Sources section.

## Validation

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover source validation, geographic aggregation, missing-timestamp safeguards, replay ordering and idempotent local storage, India panel and prior-only features, hotspot transitions, chronological splits, the next-period forecast product, and all eight Streamlit sections via AppTest.
