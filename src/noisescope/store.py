"""SQLite storage for station readings (CPCB) and edge-node acoustic events."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS readings (
    station_id   INTEGER NOT NULL,
    ts           TEXT    NOT NULL,   -- IST wall-clock 'YYYY-MM-DD HH:MM:SS', end of 10-min interval
    laeq         REAL,               -- 10-min LAF average (dB(A)); falls back to instantaneous LAF if absent
    lmax         REAL,
    lmin         REAL,
    lins         REAL,
    metric       TEXT NOT NULL,      -- 'LAF_avg' or 'LAF_ins'
    quality      TEXT NOT NULL,      -- 'valid' or reason flag
    source       TEXT NOT NULL,      -- 'cpcb_csv_export' or 'cpcb_live'
    ingested_at  TEXT NOT NULL,
    PRIMARY KEY (station_id, ts)
);
CREATE TABLE IF NOT EXISTS edge_events (
    node_id      TEXT NOT NULL,
    ts           TEXT NOT NULL,      -- IST wall-clock of the window start
    window_s     REAL NOT NULL,
    lat          REAL,
    lon          REAL,
    level_db     REAL,               -- LAeq of the window (dBFS + calibration offset)
    calibrated   INTEGER NOT NULL,   -- 1 if a calibration offset to dB SPL was supplied
    source_class TEXT,
    confidence   REAL,
    top3         TEXT,               -- JSON list of [class, probability]
    origin       TEXT,               -- audio file name / 'microphone'
    PRIMARY KEY (node_id, ts)
);
CREATE TABLE IF NOT EXISTS live_status (
    station_id   INTEGER PRIMARY KEY,
    observed_ts  TEXT,               -- IST wall-clock of the latest portal reading
    retrieved_at TEXT NOT NULL,      -- UTC ISO time of the poll
    lins         REAL,
    lcins        REAL,
    battery_v    REAL,
    raw_json     TEXT
);
"""

READING_COLS = ["station_id", "ts", "laeq", "lmax", "lmin", "lins", "metric", "quality", "source"]
EDGE_COLS = ["node_id", "ts", "window_s", "lat", "lon", "level_db", "calibrated",
             "source_class", "confidence", "top3", "origin"]


@contextmanager
def connect(db_path: Path | str = DB_PATH):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA)
        yield con
        con.commit()
    finally:
        con.close()


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def upsert_readings(df: pd.DataFrame, db_path: Path | str = DB_PATH) -> int:
    """Insert or replace 10-min readings. Returns the number of rows written."""
    if df.empty:
        return 0
    missing = set(READING_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"readings frame missing columns: {sorted(missing)}")
    rows = df[READING_COLS].copy()
    rows["ts"] = pd.to_datetime(rows["ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")
    rows = rows.astype(object).where(rows.notna(), None)
    now = _now_utc()
    payload = [tuple(r) + (now,) for r in rows.itertuples(index=False, name=None)]
    with connect(db_path) as con:
        con.executemany(
            f"INSERT OR REPLACE INTO readings ({', '.join(READING_COLS)}, ingested_at) "
            f"VALUES ({', '.join('?' * (len(READING_COLS) + 1))})",
            payload,
        )
    return len(payload)


def load_readings(db_path: Path | str = DB_PATH, valid_only: bool = True) -> pd.DataFrame:
    with connect(db_path) as con:
        q = "SELECT * FROM readings" + (" WHERE quality = 'valid'" if valid_only else "")
        df = pd.read_sql_query(q, con)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.sort_values(["station_id", "ts"]).reset_index(drop=True)


def upsert_live_status(records: Iterable[dict], db_path: Path | str = DB_PATH) -> None:
    cols = ["station_id", "observed_ts", "retrieved_at", "lins", "lcins", "battery_v", "raw_json"]
    with connect(db_path) as con:
        con.executemany(
            f"INSERT OR REPLACE INTO live_status ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
            [tuple(r.get(c) for c in cols) for r in records],
        )


def load_live_status(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    with connect(db_path) as con:
        return pd.read_sql_query("SELECT * FROM live_status", con)


def insert_edge_events(events: Iterable[dict], db_path: Path | str = DB_PATH) -> int:
    rows = []
    for e in events:
        e = dict(e)
        if not isinstance(e.get("top3"), str):
            e["top3"] = json.dumps(e.get("top3"))
        e["calibrated"] = int(bool(e.get("calibrated")))
        rows.append(tuple(e.get(c) for c in EDGE_COLS))
    if not rows:
        return 0
    with connect(db_path) as con:
        con.executemany(
            f"INSERT OR REPLACE INTO edge_events ({', '.join(EDGE_COLS)}) VALUES ({', '.join('?' * len(EDGE_COLS))})",
            rows,
        )
    return len(rows)


def load_edge_events(db_path: Path | str = DB_PATH) -> pd.DataFrame:
    with connect(db_path) as con:
        df = pd.read_sql_query("SELECT * FROM edge_events", con)
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"])
    return df


def clear_edge_events(node_id: str | None = None, db_path: Path | str = DB_PATH) -> None:
    with connect(db_path) as con:
        if node_id:
            con.execute("DELETE FROM edge_events WHERE node_id = ?", (node_id,))
        else:
            con.execute("DELETE FROM edge_events")
