# 🚀 Deployment

[← Back to README](../README.md) · [Architecture](ARCHITECTURE.md) · [Data](DATA.md) · [Methodology](METHODOLOGY.md) · [Results](RESULTS.md) · [User guide](USER_GUIDE.md) · [Development](DEVELOPMENT.md)

---

## 1. Run locally

Requirements: Python 3.10+ (tested on 3.12) and about 1 GB of disk for dependencies.

```bash
git clone https://github.com/umugash/AI-IOT-Project.git
cd AI-IOT-Project
python -m venv .venv
```

Activate the environment:

```powershell
.\.venv\Scripts\Activate.ps1      # Windows PowerShell
```

```bash
source .venv/bin/activate         # Linux / macOS
```

Install and run:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501. On first launch the app builds `data/noisescope.db` from the bundled CPCB exports, which takes a few seconds. Trained models ship in `models/`.

**Offline demo:** set `NOISESCOPE_MAP_STYLE=white-bg` to draw maps without downloading basemap tiles.

---

## 2. Streamlit Community Cloud (free hosting)

The live app runs here: https://noisescope.streamlit.app/

1. Sign in at https://share.streamlit.io with GitHub.
2. **Create app → Deploy a public app from GitHub**.
3. Repository `umugash/AI-IOT-Project`, branch `main` (or the working branch), main file `app.py`.
4. **Advanced settings → Python 3.12 → Deploy.**

| File | Purpose on the cloud |
|---|---|
| `requirements.txt` | Exact pinned versions, so a restart cannot swap the Streamlit front-end mid-session |
| `packages.txt` | System packages: `libgomp1` (OpenMP for LightGBM) and `ffmpeg` (MP3/M4A uploads) |
| `.streamlit/config.toml` | Theme and upload limit |
| `models/` | Pre-trained models and backtest results, so the first visit does no training |

**Notes**
- The first build takes 5–10 minutes. Later visits load in seconds; an idle app sleeps and wakes in about 30 s.
- The cloud filesystem is temporary: data added with **Poll CPCB now** or logged edge events last until the app restarts.
- After each redeploy, a browser holding the old page may show *"Failed to fetch dynamically imported module"*. A hard refresh (Ctrl/Cmd + Shift + R) fixes it.

---

## 3. Live CPCB data collection

Run this on a machine that can reach `http://cpcbnoise.com` (e.g. your laptop in India):

```bash
python scripts/poll_cpcb_live.py --once
```

Keep it running to poll every 10 minutes:

```bash
python scripts/poll_cpcb_live.py --interval 600
```

Add `--save-raw` to keep the portal's raw XML responses (in `data/cpcb/live_snapshots/raw_xml/`) for debugging.

Output format (values illustrative):

```text
[2026-11-04 09:10:02] 4/10 stations OK, stored 4 readings
   Guindy               2026-11-04 09:00:00  LAF=78.4
   Anna Nagar           2026-11-04 09:00:00  LAF=61.2
   Egmore Eye Hospital  EMPTY: HTTP 200, 87 bytes: ...
```

**Run it in the background**
- **Windows:** Task Scheduler → *Create Basic Task* → *At log on* → program `python`, arguments `scripts\poll_cpcb_live.py --interval 600`, start in the project folder.
- **Linux / Raspberry Pi:** add a crontab entry:

```bash
*/10 * * * * cd /home/pi/AI-IOT-Project && .venv/bin/python scripts/poll_cpcb_live.py --once >> poll.log 2>&1
```

**Add historical exports:** download station CSVs from the portal into `data/cpcb/raw/`, keeping the station id in the file name, then run:

```bash
python scripts/build_database.py
```

---

## 4. Edge nodes

### 4.1 From a recording (phone / any recorder)

```bash
python scripts/edge_node.py --node-id srm-main-gate --lat 12.8231 --lon 80.0442 --audio recordings/main_gate.wav --start "2026-11-04 08:30:00"
```

### 4.2 Live microphone (laptop / Raspberry Pi)

```bash
pip install sounddevice
```

```bash
python scripts/edge_node.py --node-id my-desk --mic --minutes 10
```

### 4.3 Calibration

Hold the microphone next to a sound level meter, read both values, then pass `--calibration-db <meter dB − reported dBFS>`. Without it, levels are relative (dBFS).

### 4.4 Raspberry Pi node (recommended hardware)

| Part | Example |
|---|---|
| Board | Raspberry Pi 4 / 5 (2 GB+) |
| Microphone | USB measurement mic or I²S MEMS mic (e.g. INMP441) |
| Power | 5 V supply or solar + battery for outdoor use |
| Enclosure | Weatherproof box with a windscreen on the mic |

Then on the Pi:

```bash
sudo apt install libportaudio2 libgomp1
```

```bash
pip install -r requirements.txt sounddevice
```

```bash
python scripts/edge_node.py --node-id pi-tnagar --lat 13.0418 --lon 80.2341 --mic --minutes 1440 --calibration-db 94
```

### 4.5 MQTT

```bash
pip install paho-mqtt
```

```bash
python scripts/edge_node.py --node-id pi-tnagar --mic --minutes 60 --mqtt-host broker.local
```

Each event is published to `noisescope/<node_id>/events` as JSON:

```json
{"node_id": "pi-tnagar", "ts": "2026-11-04 08:30:05", "window_s": 5.0, "lat": 13.0418, "lon": 80.2341,
 "level_db": 81.7, "calibrated": true, "source_class": "Vehicle horn", "confidence": 0.97,
 "top3": [["Vehicle horn", 0.97], ["Traffic & engines", 0.02], ["Animals", 0.01]], "origin": "microphone"}
```

Events are also appended to `outputs/edge_events.jsonl` and stored in SQLite.
