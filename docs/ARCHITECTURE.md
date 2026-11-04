# 🏗️ Architecture

[← Back to README](../README.md) · [Data](DATA.md) · [Methodology](METHODOLOGY.md) · [Results](RESULTS.md) · [User guide](USER_GUIDE.md) · [Deployment](DEPLOYMENT.md) · [Development](DEVELOPMENT.md)

---

## 1. Layered AIoT design

NoiseScope follows the classic AIoT stack: **sense → compute at the edge → ingest & store → analyse & predict → present**.

```mermaid
flowchart TB
    subgraph L1["1 · Sensing"]
        A1["CPCB NANMN stations<br/>(10 Chennai sites)"]
        A2["Edge microphones<br/>(phone / laptop / Raspberry Pi)"]
    end
    subgraph L2["2 · Edge computing"]
        B1["5-s windowing"]
        B2["A-weighted level (dBFS / dB SPL)"]
        B3["Feature extraction<br/>log-mel · MFCC · spectral"]
        B4["Source classifier (LightGBM)"]
    end
    subgraph L3["3 · Ingestion & storage"]
        C1["CPCB live poller<br/>every 10 min"]
        C2["CPCB CSV export loader"]
        C3["Quality control"]
        C4[("SQLite store")]
        C5["JSONL log / MQTT"]
    end
    subgraph L4["4 · Analytics & AI"]
        D1["Hourly LAeq · Lday / Lnight"]
        D2["Limit exceedance · hotspot labels"]
        D3["Hotspot episodes · hour×weekday patterns"]
        D4["Multi-horizon forecaster<br/>point · 80% range · P(hotspot)"]
        D5["Source mix per location"]
    end
    subgraph L5["5 · Presentation"]
        E1["Streamlit dashboard (7 pages)"]
    end
    A1 --> C1 & C2
    A2 --> B1 --> B2 --> B3 --> B4
    B4 --> C4 & C5
    C1 --> C3
    C2 --> C3 --> C4
    C4 --> D1 --> D2 --> D3
    D2 --> D4
    C4 --> D5
    D1 & D3 & D4 & D5 --> E1
```

| Layer | Component | Code |
|---|---|---|
| Sensing | CPCB 10-minute LAF readings | `data/cpcb/raw/`, live portal |
| Sensing | Microphone or audio file | `scripts/edge_node.py`, dashboard recorder |
| Edge | Level, features, classification | `src/noisescope/audio.py`, `src/noisescope/edge.py` |
| Ingestion | CSV parser, live poller | `src/noisescope/cpcb.py`, `scripts/poll_cpcb_live.py` |
| Quality | Range checks, stuck-sensor detection | `cpcb.quality_flags`, `cpcb.flag_flatlines` |
| Storage | SQLite (3 tables) | `src/noisescope/store.py` |
| Analytics | Hourly levels, exceedance, hotspots, episodes | `src/noisescope/analytics.py` |
| Prediction | LightGBM forecaster + baselines | `src/noisescope/forecasting.py` |
| Presentation | Dashboard | `app.py`, `dashboard/` |

---

## 2. Data flows

### 2.1 Official station data (where & when)

```mermaid
sequenceDiagram
    participant P as CPCB portal
    participant L as Live poller
    participant Q as Quality control
    participant DB as SQLite
    participant A as Analytics
    participant F as Forecaster
    participant UI as Dashboard
    loop every 10 minutes
        L->>P: guest login + getlastchannelsdata (per station)
        P-->>L: latest channel values (XML)
        L->>DB: upsert live_status + reading (only successful stations)
    end
    UI->>DB: load readings
    DB-->>Q: 10-min readings
    Q-->>A: valid readings (range + stuck-sensor filtered)
    A-->>F: hourly LAeq + limits
    F-->>UI: 1–24 h forecasts, P(hotspot)
    A-->>UI: patterns, exceedance, episodes
```

### 2.2 Edge acoustic data (what)

```mermaid
sequenceDiagram
    participant M as Microphone / recording
    participant N as Edge node
    participant DB as SQLite / JSONL / MQTT
    participant UI as Dashboard
    M->>N: audio stream
    loop each 5-second window
        N->>N: A-weighted level
        N->>N: features → classifier → top-3 sources
        N->>DB: event {node_id, ts, lat, lon, level_db, source_class, confidence}
    end
    UI->>DB: events
    UI-->>UI: source mix · dominant source · man-made energy share
```

An edge event is about 300 bytes, while 5 seconds of raw 22 kHz audio is about 220 kB. Sending events instead of audio cuts bandwidth by roughly **700×**, and no recognisable speech ever leaves the device.

---

## 3. Storage schema

```mermaid
erDiagram
    READINGS {
        int station_id PK
        text ts PK "IST, end of 10-min interval"
        real laeq "10-min LAF average dB(A)"
        real lmax
        real lmin
        real lins "instantaneous LAF"
        text metric "LAF_avg | LAF_ins"
        text quality "valid | flags"
        text source "cpcb_csv_export | cpcb_live"
        text ingested_at
    }
    LIVE_STATUS {
        int station_id PK
        text observed_ts "IST"
        text retrieved_at "UTC"
        real lins
        real lcins "dB(C)"
        real battery_v
        text raw_json
    }
    EDGE_EVENTS {
        text node_id PK
        text ts PK "IST window start"
        real window_s
        real lat
        real lon
        real level_db
        int calibrated
        text source_class
        real confidence
        text top3 "JSON"
        text origin
    }
```

- **Idempotent writes.** Primary keys plus `INSERT OR REPLACE` mean that re-running an import or poll never creates duplicates.
- **Failed polls are never stored**, so the last good live reading is always kept.
- The database (`data/noisescope.db`) is generated and git-ignored. It rebuilds automatically from `data/cpcb/raw/`.

---

## 4. Module map

```mermaid
flowchart LR
    config["config.py<br/>stations · limits · paths"]
    acoustics["acoustics.py<br/>energy averaging"]
    cpcb["cpcb.py<br/>CSV + live portal"]
    store["store.py<br/>SQLite"]
    analytics["analytics.py<br/>hourly · hotspots · QC"]
    forecasting["forecasting.py<br/>LightGBM forecaster"]
    audio["audio.py<br/>features · classifier"]
    edge["edge.py<br/>edge node · source mix"]
    dash["dashboard/<br/>data.py · views.py"]
    config --> cpcb & analytics & forecasting & store
    acoustics --> analytics & edge
    cpcb --> analytics
    analytics --> forecasting
    audio --> edge
    store --> dash
    analytics & forecasting & audio & edge & cpcb --> dash
```

---

## 5. Key design decisions

| Decision | Why |
|---|---|
| **LightGBM for both tasks** | Strong on tabular features, fast on a laptop CPU, no GPU needed, and the model saves as plain text that loads across library versions |
| **Forecast the *change* from the current level** | Makes "no change" the model's starting point; together with stronger regularisation it cut 1-hour MAE from 3.15 to 2.48 dB |
| **One direct multi-horizon model** | Horizon is a feature; avoids error build-up from recursive forecasting |
| **Quantile models for the range** | A data-driven 80% range; the 5th/95th percentile pair achieves 82–85% real coverage |
| **SQLite instead of Kafka / Spark** | Ten stations every 10 minutes is about 1,440 rows a day; a broker or cluster would add cost and complexity without benefit. The edge events already follow an MQTT-ready envelope for scale-out |
| **Classify at the edge** | Privacy (no raw audio leaves the device), about 700× less bandwidth, works offline |
| **Pre-trained models committed** | The hosted app starts in seconds with no training on first visit |
| **Automatic stuck-sensor detection** | Real Chennai sensors froze for days; without it a broken sensor would top the hotspot ranking |
