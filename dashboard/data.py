"""Cached data access for the dashboard."""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from noisescope import analytics, audio, cpcb, forecasting, store
from noisescope.config import CPCB_SNAPSHOT_DIR, DB_PATH, STATIONS

STATUS_COLORS = {"Compliant": "#2e9e5b", "Exceeding": "#f0a202", "Hotspot": "#d8343b", "Unknown": "#8a8f98"}


def ensure_database() -> None:
    """Build the SQLite store from the bundled CPCB exports on first run."""
    with store.connect() as con:
        n = con.execute("SELECT COUNT(*) FROM readings").fetchone()[0]
    if n == 0:
        store.upsert_readings(cpcb.load_all_exports())
        for snap in sorted(CPCB_SNAPSHOT_DIR.glob("*.json")):
            store.upsert_live_status(cpcb.snapshot_json_to_records(snap))


@st.cache_data(ttl=300, show_spinner=False)
def readings() -> pd.DataFrame:
    ensure_database()
    return store.load_readings(valid_only=False)


@st.cache_data(ttl=300, show_spinner=False)
def hourly() -> pd.DataFrame:
    return analytics.hourly_levels(analytics.clean_readings(readings()))


RELIABLE_VALID_PCT = 70.0  # stations with less valid (non-stuck) data are flagged


@st.cache_data(ttl=300, show_spinner=False)
def station_summary() -> pd.DataFrame:
    s = analytics.station_summary(hourly())
    q = analytics.quality_summary(readings())[["station_id", "valid_pct"]]
    s = s.merge(q, on="station_id", how="left")
    s["reliable"] = s["valid_pct"] >= RELIABLE_VALID_PCT
    s["label"] = s["station"] + np.where(s["reliable"], "", " ⚠")
    return s


@st.cache_data(ttl=300, show_spinner=False)
def quality() -> pd.DataFrame:
    return analytics.quality_summary(readings())


@st.cache_data(ttl=300, show_spinner=False)
def live_status() -> pd.DataFrame:
    df = store.load_live_status()
    if df.empty:
        return df
    df["station"] = df["station_id"].map(lambda s: STATIONS[s].label)
    df["zone"] = df["station_id"].map(lambda s: STATIONS[s].zone)
    df["observed_ts"] = pd.to_datetime(df["observed_ts"])
    return df


@st.cache_resource(show_spinner="Training hotspot forecaster on the latest data ...")
def forecaster(data_signature: str) -> forecasting.ForecastModels:
    """Load the saved model if it matches the current data, otherwise retrain (~5 s)."""
    h = hourly()
    try:
        m = forecasting.load_models()
        if m.trained_on.get("to") == str(h["hour_start"].max()) and m.trained_on.get("rows"):
            return m
    except Exception:
        pass
    return forecasting.train_final(h)


def data_signature() -> str:
    h = hourly()
    return f"{len(h)}-{h['hour_start'].max()}"


@st.cache_data(show_spinner="Running chronological backtest ...")
def evaluation(data_signature: str) -> dict:
    return forecasting.evaluate(hourly())


@st.cache_data(ttl=300, show_spinner=False)
def latest_forecast(data_signature: str) -> pd.DataFrame:
    return forecasting.forecast_latest(forecaster(data_signature), hourly())


@st.cache_resource(show_spinner="Loading sound classifier ...")
def classifier() -> audio.SoundClassifier | None:
    try:
        return audio.load_classifier()
    except FileNotFoundError:
        return None


@st.cache_data(ttl=60, show_spinner=False)
def edge_events() -> pd.DataFrame:
    return store.load_edge_events()


def refresh() -> None:
    st.cache_data.clear()


def fmt_age(ts: pd.Timestamp | None, now: datetime | None = None) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    now = now or pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    delta = now - ts
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{int(delta.total_seconds() // 60)} min ago"
    if hours < 48:
        return f"{hours:.0f} h ago"
    return f"{hours / 24:.0f} days ago"


def footer_note() -> None:
    st.caption(f"Data store: `{DB_PATH.relative_to(DB_PATH.parents[1])}` · Times are IST · "
               "Limits: Noise Pollution (Regulation and Control) Rules, 2000")
