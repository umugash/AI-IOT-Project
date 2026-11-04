# 🛠️ Development

[← Back to README](../README.md) · [Architecture](ARCHITECTURE.md) · [Data](DATA.md) · [Methodology](METHODOLOGY.md) · [Results](RESULTS.md) · [User guide](USER_GUIDE.md) · [Deployment](DEPLOYMENT.md)

---

## 1. Project structure

```text
AI-IOT-Project/
├── app.py                        # Streamlit entry point (7 pages)
├── dashboard/
│   ├── data.py                   # cached data access, model loading
│   └── views.py                  # the seven pages
├── src/noisescope/
│   ├── config.py                 # stations, CPCB zone limits, paths
│   ├── acoustics.py              # energy averaging (Leq)
│   ├── cpcb.py                   # CSV export parser, live portal client, quality flags
│   ├── store.py                  # SQLite: readings, live_status, edge_events
│   ├── analytics.py              # hourly LAeq, Lday/Lnight, hotspots, episodes, QC summary
│   ├── forecasting.py            # multi-horizon LightGBM forecaster, baselines, backtest
│   ├── audio.py                  # features, A-weighted level, classifier train/infer
│   └── edge.py                   # edge node, source mix, node summary
├── scripts/
│   ├── build_database.py         # CSV exports + snapshots → SQLite
│   ├── poll_cpcb_live.py         # live CPCB poller
│   ├── train_sound_classifier.py # ESC-50 download, training, CV report
│   ├── train_forecaster.py       # backtest + final model
│   ├── edge_node.py              # run an edge node (file / microphone / MQTT)
│   ├── make_demo_audio.py        # build the demo street mix
│   └── make_figures.py           # regenerate docs/images charts
├── models/
│   ├── sound_source_classifier.{txt,json}   # LightGBM text model + CV report
│   └── forecaster/                          # 4 boosters, meta.json, evaluation/*.csv
├── data/
│   ├── cpcb/raw/                 # CPCB 10-min exports (10 stations)
│   ├── cpcb/live_snapshots/      # saved live snapshot(s)
│   ├── reference/                # TNPCB 2023-24, SRMIST 2025 reference values
│   └── demo/demo_street_mix.ogg  # demo audio (ESC-50 derived)
├── docs/                         # this documentation + images
├── tests/test_core.py            # pytest suite
├── requirements.txt · packages.txt · pyproject.toml
└── .streamlit/config.toml
```

---

## 2. Tests

```bash
pytest
```

All 14 tests run in under 15 seconds. They cover:

| Area | Tests |
|---|---|
| Acoustics | Energy averaging, day/night limits |
| CPCB parsing | Both export column layouts, date format, range flags, all bundled files load without duplicates |
| Live portal | XML parsing (attribute and child layouts), malformed-response fallback, IST epoch decoding, failed polls never stored |
| Quality control | Stuck-sensor detection |
| Analytics | End-labelled hour assignment, exceedance and status |
| Storage | Idempotent SQLite round-trip |
| Forecasting | **No look-ahead leakage** between train and test; the model beats persistence on a learnable daily cycle; 24-hour outlook shape |
| Audio | Fixed feature length, A-weighting reference (−6 dB per halving of amplitude) |
| Edge | Event generation and source energy shares summing to 1 |

---

## 3. Retraining

| What | Command | Time | Output |
|---|---|---|---|
| Sound classifier | `python scripts/train_sound_classifier.py` | ~2 min plus ~600 MB ESC-50 download | `models/sound_source_classifier.*`, `outputs/sound_classifier_report.json` |
| Forecaster | `python scripts/train_forecaster.py` | ~10 s | `models/forecaster/` (models + evaluation) |
| Figures | `pip install matplotlib`, then `python scripts/make_figures.py` | ~5 s | `docs/images/*.png` |
| Demo audio | `python scripts/make_demo_audio.py` | ~5 s | `data/demo/demo_street_mix.ogg` |

**Retrain the forecaster whenever new CPCB data is added.** The app compares the data the bundled model was trained on with the current data. If they differ, it retrains in memory automatically, but committing a fresh model keeps the hosted app's first load fast.

---

## 4. Extending NoiseScope

### Add a monitoring station or another city
Add a `Station(...)` entry to `STATIONS` in `src/noisescope/config.py`, with its id, name, zone and coordinates. Then drop its CPCB CSV exports into `data/cpcb/raw/`.

### Add a sound category
Edit `URBAN_CATEGORIES` in `src/noisescope/audio.py` (and add a colour to `CATEGORY_COLORS`), then retrain. To use your own labelled clips, build a table with the same columns as `esc50_table()` (`path`, `urban_category`, `fold`, `category`) and pass it through `featurize_paths` and `cross_validate`.

### Change the hotspot definition
`HOTSPOT_MARGIN_DB` in `src/noisescope/analytics.py` (default 10 dB) is used by the analytics, the forecaster's hotspot classifier and the dashboard. Retrain the forecaster after changing it.

### Add a dashboard page
Write a function in `dashboard/views.py` and register it in the `pages` list in `app.py` with `st.Page(...)`.

---

## 5. Conventions

- **Times** are IST wall-clock throughout. CPCB 10-minute stamps mark the **end** of the interval.
- **Levels** are always energy-averaged (`acoustics.leq`); never take an arithmetic mean of decibels.
- **Models** are saved as LightGBM text plus JSON (not pickles), so they load across library versions.
- **Writes** to SQLite are idempotent upserts.
