# NoiseScope Chennai

**AIoT-based urban sound source classification and spatiotemporal noise hotspot prediction using acoustic sensor data.**

Urban areas are continuously exposed to noise from traffic, construction, emergency vehicles, machinery and human activity. Conventional monitoring reports a decibel number and little else: it does not say *which source* is responsible or *how* the noise moves across the city and the day. NoiseScope combines:

1. **Real acoustic sensor data**: the 10 Chennai stations of the CPCB National Ambient Noise Monitoring Network (10-minute LAF readings), with a live connector to the CPCB portal.
2. **Sound source classification (edge AI)**: a machine-learning model that recognises 8 urban source categories from audio, run on edge nodes (phone, laptop or Raspberry Pi).
3. **Spatiotemporal analysis**: hourly LAeq, Lday/Lnight, exceedance of the CPCB legal limits, hour × weekday rhythms, and hotspot episodes.
4. **Hotspot prediction**: 1–24 hour forecasts of noise level and hotspot probability for every station.
5. **A dashboard** that brings it all together.

```
 Sensing                 Edge AI                     Storage                Analytics & prediction           Product
 ───────                 ───────                     ───────                ──────────────────────           ───────
 CPCB stations ──► live poller / CSV loader ──► QC (range, stuck sensor) ──► SQLite ──► hourly LAeq, limits ──► Streamlit
 Edge mic ──► A-weighted level + log-mel ──► source classifier ──► JSON events ──►   │      hotspot labelling        dashboard
                                                                                    └──► LightGBM forecaster ───►
```

## Quick start

Python 3.10+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app builds its database from the bundled CPCB exports on first launch. The trained sound classifier ships in `models/`, and the forecaster trains itself in about 5 seconds.

### Live data from CPCB (run on your laptop)

```powershell
python scripts/poll_cpcb_live.py --once            # one poll
python scripts/poll_cpcb_live.py --interval 600    # every 10 min, leave running
```

The CPCB portal (`http://cpcbnoise.com`, public guest login) only serves plain HTTP, so it must be reachable from the machine running the poller. Every poll updates the **Live CPCB network** page and appends readings to the store. The forecaster then predicts from current conditions instead of from the last bundled month. To add more history, export station CSVs from the portal and drop them into `data/cpcb/raw/`. Keep the station id in the file name (e.g. `chennai_guindy_267_mar2026.csv`), then run `python scripts/build_database.py`.

### Your own acoustic sensor (edge node)

Record with a phone (e.g. at SRM main gate or a Chennai junction), then:

```powershell
python scripts/edge_node.py --node-id srm-main-gate --lat 12.8231 --lon 80.0442 --audio recordings\main_gate.wav --start "2026-11-03 08:30:00"
```

You can also listen live with `--mic --minutes 10` (needs `pip install sounddevice`), or record in the browser on the **Sound source classification** page. Phone microphones are not calibrated. To report dB SPL, record next to a sound level meter once and pass `--calibration-db <meter_dB − reported_dBFS>`. Add `--mqtt-host <broker>` to publish events over MQTT (`pip install paho-mqtt`).

## Results

### Sound source classifier

Trained on ESC-50 (2,000 labelled clips; 1,160 used in 8 urban categories), with augmentation: pitch and time shift, plus street background mixed in at 5–20 dB SNR. Scores are from **ESC-50's official 5-fold cross-validation**, evaluated on unaugmented held-out folds.

| Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|
| **84.6%** | 81.6% | 0.833 |

| Category | ESC-50 classes | Precision | Recall |
|---|---|---|---|
| Vehicle horn | car_horn | 0.86 | 0.60 |
| Traffic & engines | engine, train, airplane, helicopter | 0.79 | 0.81 |
| Emergency siren | siren | 0.89 | 0.82 |
| Construction & machinery | chainsaw, hand_saw, vacuum_cleaner | 0.92 | 0.82 |
| Human activity | crying_baby, laughing, coughing, footsteps, clapping, sneezing | 0.84 | 0.87 |
| Bells & alarms | church_bells, clock_alarm | 0.88 | 0.85 |
| Animals | dog, rooster, crow, cat, hen, cow | 0.85 | 0.90 |
| Nature & weather | rain, wind, thunderstorm, sea_waves, crickets, chirping_birds | 0.84 | 0.85 |

### Hotspot forecaster

Chronological hold-out: the first 70% of each station's hours train the model and the last 30% (9,949 forecasts) test it. Mean absolute error in dB:

| Hours ahead | 1 | 2 | 3 | 6 | 12 | 24 |
|---|---|---|---|---|---|---|
| **NoiseScope (LightGBM)** | 2.48 | **2.95** | **3.21** | **3.55** | **3.84** | **3.70** |
| Persistence (last hour) | **2.39** | 3.30 | 3.95 | 5.21 | 6.18 | 3.98 |
| Same hour yesterday | 3.88 | 4.00 | 4.09 | 4.09 | 4.13 | 3.98 |
| Station hourly profile | 6.71 | 6.83 | 6.96 | 6.90 | 6.90 | 6.67 |

Hotspot probability AUC is 0.98 at 1 h and 0.94 at 24 h. The predicted 80% range contains the actual value 82–85% of the time. At 1 hour, "same as now" is still marginally better. From 2 hours onward the model clearly wins, with 32% lower error at 6 h.

### What the Chennai data shows

**Hotspot** = hourly LAeq more than 10 dB above the CPCB limit for the station's zone (Noise Rules 2000: Industrial 75/70, Commercial 65/55, Residential 55/45, Silence 50/40 dB(A), day/night).

| Station | Zone | Lday | Lnight | Limits | Hotspot hours | Valid data |
|---|---|---|---|---|---|---|
| Anna Nagar | Silence | 98.0 | 87.6 | 50/40 | 85% | 88% |
| Egmore Eye Hospital | Silence | 64.2 | 64.8 | 50/40 | 93% | 100% |
| Guindy | Industrial | 89.3 | 89.1 | 75/70 | 52% | 98% |
| Sowcarpet | Residential | 63.9 | 58.1 | 55/45 | 34% | 100% |
| Washermanpet | Commercial | 82.9 | 81.5 | 65/55 | 39% | 94% |
| Pallikaranai | Commercial | 71.3 | 71.4 | 65/55 | 29% | 100% |
| Triplicane | Residential | 60.7 | 56.3 | 55/45 | 15% | 100% |
| Velachery ⚠ | Residential | 97.4 | 94.4 | 55/45 | 100% | 54% |
| T. Nagar ⚠ | Commercial | 78.2 | 58.2 | 65/55 | 1% | 46% |
| Perambur ⚠ | Commercial | 53.6 | 54.1 | 65/55 | 0% | 58% |

- **78%** of all station-hours exceed the legal limit.
- The **silence zones** (Egmore Eye Hospital, Anna Nagar) are hotspots for most of the day *and night*. The places that should be quietest get the least protection.
- Anna Nagar's extreme Lday is driven by **14–19 January (Bhogi/Pongal)**, with 10-minute levels above 100 dB.
- ⚠ **Sensor faults**: T. Nagar (from 10 Dec 2025), Velachery and Perambur report near-constant values for hours at a time. NoiseScope detects these stuck stretches automatically (readings varying by less than 0.2 dB over 3 h) and excludes them. Treat these stations' remaining values with caution.

## Repository layout

```
app.py                     Streamlit entry point (7 pages)
dashboard/                 page code and cached data access
src/noisescope/
  config.py                stations, CPCB zone limits, paths
  cpcb.py                  CPCB CSV export parser + live portal connector
  store.py                 SQLite store (readings, live_status, edge_events)
  analytics.py             hourly LAeq, Lday/Lnight, exceedance, hotspots, episodes, QC
  forecasting.py           multi-horizon LightGBM forecaster, baselines, backtest
  audio.py                 features, A-weighted level, classifier training / inference
  edge.py                  edge node: audio -> events; source mix per location
scripts/
  build_database.py        CSV exports + snapshots -> SQLite
  poll_cpcb_live.py        live CPCB poller
  train_sound_classifier.py  downloads ESC-50, trains, writes CV report
  train_forecaster.py      backtest + final forecaster
  edge_node.py             run an edge node on a file or microphone
  make_demo_audio.py       builds data/demo/demo_street_mix.ogg
data/cpcb/raw/             CPCB 10-minute exports (10 Chennai stations)
data/cpcb/live_snapshots/  saved live portal snapshot(s)
data/reference/            TNPCB 2023-24 annual and SRMIST 2025 reference values
models/                    trained sound classifier (LightGBM text + JSON report)
tests/                     pytest suite
```

Run the tests with `pytest`. To retrain everything: `python scripts/train_sound_classifier.py` (downloads ESC-50, about 600 MB) and `python scripts/train_forecaster.py`.

## Limitations and next steps

- The bundled CPCB history is about **one month per station**, and the months differ by station (May 2025 – Feb 2026). Leaving the live poller running builds an aligned, longer record. Longer data allows weekly and seasonal patterns and stronger forecasts.
- Station coordinates are **approximate locality points**. CPCB does not publish surveyed sensor positions, so there is no interpolated city-wide surface.
- CPCB stations do not provide audio, so **source attribution comes from edge nodes**. Deploying a few phones or Raspberry Pis near CPCB stations would let the two be fused directly.
- ESC-50 is not Chennai audio. Recording and labelling local clips (auto-rickshaw and bus horns, temple bells, metro construction) and fine-tuning would make the classifier more relevant. A pretrained audio network (e.g. YAMNet or PANNs embeddings) is a natural upgrade.
- Edge levels are relative (dBFS) unless a calibration offset is supplied.

## Data credits

- CPCB National Ambient Noise Monitoring Network, <http://cpcbnoise.com> (public guest access).
- K. J. Piczak, *ESC: Dataset for Environmental Sound Classification*, ACM MM 2015, CC BY-NC 3.0. The model and `data/demo/demo_street_mix.ogg` derive from it, so use them for non-commercial purposes only.
- Tamil Nadu Pollution Control Board Annual Report 2023-24; SRMIST air and noise monitoring report, January 2025.
