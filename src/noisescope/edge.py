"""AIoT edge node: turns an audio stream into compact acoustic events.

An edge node (laptop, Raspberry Pi, phone recording) runs the source classifier locally
and only transmits small JSON events (level + source class), never raw audio. Events
are written to the local SQLite store and to a JSONL log, and optionally published to
an MQTT broker (topic ``noisescope/<node_id>/events``) if ``paho-mqtt`` is installed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import audio
from .store import insert_edge_events


@dataclass
class EdgeNode:
    node_id: str
    lat: float | None = None
    lon: float | None = None
    calibration_db: float | None = None   # add to dBFS to get dB SPL (from a reference meter)
    window_s: float = audio.CLIP_SECONDS

    def events_from_signal(self, y: np.ndarray, start: datetime, origin: str,
                           clf: audio.SoundClassifier, sr: int = audio.SR) -> list[dict]:
        frames = audio.analyse_signal(y, clf, sr=sr, window_s=self.window_s,
                                      calibration_db=self.calibration_db)
        events = []
        for r in frames.itertuples(index=False):
            ts = start + timedelta(seconds=float(r.offset_s))
            events.append({
                "node_id": self.node_id,
                "ts": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "window_s": float(r.duration_s),
                "lat": self.lat, "lon": self.lon,
                "level_db": round(float(r.level_db), 2),
                "calibrated": bool(r.calibrated),
                "source_class": r.source_class,
                "confidence": None if pd.isna(r.confidence) else round(float(r.confidence), 3),
                "top3": r.top3,
                "origin": origin,
            })
        return events


def publish(events: Iterable[dict], jsonl_path: Path | None = None, db: bool = True,
            mqtt_host: str | None = None, mqtt_port: int = 1883) -> int:
    events = list(events)
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with open(jsonl_path, "a", encoding="utf-8") as fh:
            for e in events:
                fh.write(json.dumps(e) + "\n")
    if db:
        insert_edge_events(events)
    if mqtt_host:
        import paho.mqtt.publish as mqtt_publish  # optional dependency
        msgs = [{"topic": f"noisescope/{e['node_id']}/events", "payload": json.dumps(e)} for e in events]
        mqtt_publish.multiple(msgs, hostname=mqtt_host, port=mqtt_port)
    return len(events)


def source_mix(events: pd.DataFrame) -> pd.DataFrame:
    """Share of time and energy contributed by each source class, per node."""
    if events.empty:
        return pd.DataFrame(columns=["node_id", "source_class", "time_share", "energy_share", "windows"])
    e = events.copy()
    e["energy"] = np.power(10.0, e["level_db"] / 10.0) * e["window_s"]
    g = e.groupby(["node_id", "source_class"]).agg(windows=("ts", "size"), seconds=("window_s", "sum"),
                                                   energy=("energy", "sum")).reset_index()
    g["time_share"] = g["seconds"] / g.groupby("node_id")["seconds"].transform("sum")
    g["energy_share"] = g["energy"] / g.groupby("node_id")["energy"].transform("sum")
    return g.drop(columns=["energy", "seconds"])


def node_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Per-node level, dominant source (by acoustic energy) and man-made energy share."""
    from .acoustics import leq
    if events.empty:
        return pd.DataFrame()
    mix = source_mix(events)
    rows = []
    for node, g in events.groupby("node_id"):
        m = mix[mix["node_id"] == node].sort_values("energy_share", ascending=False)
        man_made = m.loc[m["source_class"].isin(audio.ANTHROPOGENIC), "energy_share"].sum()
        rows.append({"node_id": node, "lat": g["lat"].dropna().mean() if g["lat"].notna().any() else None,
                     "lon": g["lon"].dropna().mean() if g["lon"].notna().any() else None,
                     "windows": len(g), "first": g["ts"].min(), "last": g["ts"].max(),
                     "leq_db": leq(g["level_db"]), "calibrated": bool(g["calibrated"].all()),
                     "dominant_source": m["source_class"].iloc[0] if len(m) else None,
                     "man_made_energy_share": man_made})
    return pd.DataFrame(rows)
