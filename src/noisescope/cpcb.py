"""CPCB National Ambient Noise Monitoring Network (NANMN) data access.

Two real data paths are supported:

1. CSV exports downloaded from the CPCB noise portal (http://cpcbnoise.com, guest login),
   one file per station, 10-minute intervals. Drop them into ``data/cpcb/raw/`` with the
   station id in the filename (e.g. ``chennai_guindy_267_mar2026.csv``).
2. The portal's "latest channel values" endpoint, polled live. The portal timestamps are
   IST wall-clock values encoded as epoch seconds, so they are decoded as naive IST.

Note: the portal is plain HTTP and must be reachable from the machine running the poller.
"""
from __future__ import annotations

import json
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import CPCB_RAW_DIR, STATIONS

PORTAL = "http://cpcbnoise.com"
VALID_RANGE_DB = (25.0, 120.0)

# ---------------------------------------------------------------------------
# CSV exports
# ---------------------------------------------------------------------------

_STATION_RE = re.compile(r"_(\d{3})(?:_|\.)")


def station_id_from_filename(path: Path) -> int:
    m = _STATION_RE.search(path.name)
    if not m or int(m.group(1)) not in STATIONS:
        raise ValueError(f"Cannot find a known CPCB station id in file name: {path.name}")
    return int(m.group(1))


def _find_col(columns: list[str], *needles: str) -> str | None:
    for c in columns:
        low = c.lower()
        if all(n.lower() in low for n in needles):
            return c
    return None


def parse_export_csv(path: Path | str, station_id: int | None = None) -> pd.DataFrame:
    """Parse one portal CSV export into the standard readings frame.

    Column order differs between stations, so columns are matched by header text.
    """
    path = Path(path)
    sid = station_id or station_id_from_filename(path)
    raw = pd.read_csv(path, index_col=False)
    raw = raw.loc[:, ~raw.columns.astype(str).str.startswith("Unnamed")]
    if raw.empty:
        return pd.DataFrame(columns=["station_id", "ts", "laeq", "lmax", "lmin", "lins", "metric", "quality", "source"])
    cols = list(raw.columns)
    c_time = _find_col(cols, "date") or cols[0]
    c_avg = _find_col(cols, "laf", "avg")
    c_ins = _find_col(cols, "laf", "ins")
    c_max = _find_col(cols, "laf", "max")
    c_min = _find_col(cols, "laf", "min")

    ts = pd.to_datetime(raw[c_time], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    num = lambda c: pd.to_numeric(raw[c], errors="coerce") if c else pd.Series(np.nan, index=raw.index)
    out = pd.DataFrame({
        "station_id": sid,
        "ts": ts,
        "lins": num(c_ins),
        "lmax": num(c_max),
        "lmin": num(c_min),
    })
    if c_avg:
        out["laeq"] = num(c_avg)
        out["metric"] = "LAF_avg"
    else:  # some exports carry no 10-min average; use the instantaneous LAF sample
        out["laeq"] = out["lins"]
        out["metric"] = "LAF_ins"
    out["quality"] = quality_flags(out["laeq"])
    out.loc[(out["lmin"] <= 0), "lmin"] = np.nan
    out.loc[(out["lmax"] > 140), "lmax"] = np.nan
    out["source"] = "cpcb_csv_export"
    out = out.dropna(subset=["ts"]).drop_duplicates(subset=["station_id", "ts"], keep="last")
    return out.reset_index(drop=True)


def quality_flags(values: pd.Series) -> pd.Series:
    lo, hi = VALID_RANGE_DB
    flags = pd.Series("valid", index=values.index, dtype=object)
    flags[values.isna()] = "missing"
    flags[values.notna() & (values < lo)] = "below_range_or_dropout"
    flags[values.notna() & (values > hi)] = "above_range_spike"
    return flags


def flag_flatlines(df: pd.DataFrame, window: int = 18, min_std_db: float = 0.2) -> pd.DataFrame:
    """Flag stuck-sensor stretches: 3 h (18 x 10 min) windows whose readings vary < 0.2 dB.

    Real ambient sound fluctuates by several dB within hours; near-constant values over
    hours are a known failure mode in the Chennai exports (e.g. T. Nagar after 10 Dec 2025).
    """
    df = df.sort_values(["station_id", "ts"]).copy()
    sd = df.groupby("station_id")["laeq"].transform(
        lambda s: s.rolling(window, center=True, min_periods=window // 2).std())
    df.loc[(sd < min_std_db) & (df["quality"] == "valid"), "quality"] = "flatline_suspect"
    return df


def load_all_exports(raw_dir: Path | str = CPCB_RAW_DIR) -> pd.DataFrame:
    frames = [parse_export_csv(p) for p in sorted(Path(raw_dir).glob("*.csv"))]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError(f"No CPCB CSV exports found in {raw_dir}")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(subset=["station_id", "ts"], keep="last").sort_values(["station_id", "ts"])


# ---------------------------------------------------------------------------
# Live portal
# ---------------------------------------------------------------------------

_LOGIN_FORM = {
    "username": "Guest", "passwd": "guest", "op2": "login", "lang": "english",
    "return": "index.php", "message": "0", "force_session": "1", "option": "login",
    "Submit": "Access",
}
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; NoiseScope-research/1.0)",
    "Origin": PORTAL,
    "Referer": f"{PORTAL}/index.php",
}


def epoch_to_ist(epoch: float | str) -> datetime:
    """Portal epochs encode IST wall-clock time; decode without a timezone shift."""
    return datetime(1970, 1, 1) + timedelta(seconds=float(epoch))


def parse_channels_xml(xml_text: str | bytes) -> dict[str, dict[str, Any]]:
    """Parse the portal's ``getlastchannelsdata`` XML into {channel_id: {...}}.

    The parser is tolerant of attribute vs. child-element layouts because the response
    format is not publicly documented. Use ``poll_live(save_raw_dir=...)`` to keep raw
    responses if the format ever changes.
    """
    if isinstance(xml_text, bytes):
        xml_text = xml_text.decode("utf-8", errors="replace")
    root = ET.fromstring(xml_text.strip())
    channels: dict[str, dict[str, Any]] = {}
    for el in root.iter():
        if not el.tag.lower().startswith("canal"):
            continue
        fields = {k.lower(): v for k, v in el.attrib.items()}
        for child in el:
            if child.text and child.text.strip():
                fields[child.tag.lower()] = child.text.strip()
        cid = next((fields[k] for k in ("id", "canal", "numero", "num", "channel") if k in fields), None)
        if cid is None:
            continue
        val = next((fields[k] for k in ("valor", "value", "val", "dato") if k in fields), None)
        when = next((fields[k] for k in ("fecha", "epoch", "timestamp", "time", "hora") if k in fields), None)
        alias = next((fields[k] for k in ("alias", "nombre", "unidad", "name", "unit") if k in fields), "")
        try:
            value = float(val) if val is not None else None
        except ValueError:
            value = None
        observed = None
        if when is not None:
            try:
                observed = epoch_to_ist(when)
            except ValueError:
                observed = pd.to_datetime(when, errors="coerce")
                observed = None if pd.isna(observed) else observed.to_pydatetime()
        channels[str(cid)] = {"alias": alias, "value": value, "observed_ist": observed, "raw_time": when}
    return channels


def channels_to_record(station_id: int, channels: dict[str, dict[str, Any]], retrieved_at: str) -> dict[str, Any]:
    """Channel 3 is LAF instantaneous (dB(A)); channel 7 is LCF instantaneous (dB(C))."""
    def val(cid):
        ch = channels.get(cid)
        return None if ch is None else ch["value"]
    main = channels.get("3") or {}
    battery = next((c["value"] for c in channels.values() if str(c.get("alias", "")).upper() == "V"), None)
    observed = main.get("observed_ist")
    return {
        "station_id": station_id,
        "observed_ts": observed.strftime("%Y-%m-%d %H:%M:%S") if observed else None,
        "retrieved_at": retrieved_at,
        "lins": val("3"),
        "lcins": val("7"),
        "battery_v": battery,
        "raw_json": json.dumps({k: {**v, "observed_ist": str(v["observed_ist"])} for k, v in channels.items()}),
    }


def snapshot_json_to_records(path: Path | str) -> list[dict[str, Any]]:
    """Convert a saved JSON snapshot (list of stations with 'channels') to live records."""
    data = json.loads(Path(path).read_text())
    records = []
    for st in data:
        chans = {}
        for cid, ch in st["channels"].items():
            observed = epoch_to_ist(ch["raw_epoch"]) if ch.get("raw_epoch") else None
            chans[str(cid)] = {"alias": ch.get("alias", ""), "value": ch.get("value"),
                               "observed_ist": observed, "raw_time": ch.get("raw_epoch")}
        records.append(channels_to_record(int(st["station_id"]), chans, st.get("retrieval_timestamp_utc", "")))
    return records


def live_records_to_readings(records: list[dict[str, Any]]) -> pd.DataFrame:
    """Turn live status records into 10-min reading rows (instantaneous LAF)."""
    rows = []
    for r in records:
        if r.get("observed_ts") is None or r.get("lins") is None:
            continue
        rows.append({"station_id": r["station_id"], "ts": pd.Timestamp(r["observed_ts"]),
                     "laeq": r["lins"], "lmax": np.nan, "lmin": np.nan, "lins": r["lins"],
                     "metric": "LAF_ins", "source": "cpcb_live"})
    df = pd.DataFrame(rows, columns=["station_id", "ts", "laeq", "lmax", "lmin", "lins", "metric", "source"])
    df["quality"] = quality_flags(df["laeq"]) if not df.empty else pd.Series(dtype=object)
    return df


def poll_live(station_ids: list[int] | None = None, timeout: float = 10.0,
              save_raw_dir: Path | None = None) -> list[dict[str, Any]]:
    """Log in as the public guest user and fetch the latest channel values per station."""
    import requests

    station_ids = station_ids or list(STATIONS)
    session = requests.Session()
    resp = session.post(f"{PORTAL}/index.php", data=_LOGIN_FORM, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    retrieved = datetime.now(timezone.utc).isoformat(timespec="seconds")
    records = []
    for sid in station_ids:
        url = f"{PORTAL}/index3.php?option=datos&task=getlastchannelsdata&estacion={sid}"
        try:
            r = session.get(url, headers=_HEADERS, timeout=timeout)
            r.raise_for_status()
            if save_raw_dir:
                Path(save_raw_dir).mkdir(parents=True, exist_ok=True)
                (Path(save_raw_dir) / f"station_{sid}_{int(time.time())}.xml").write_bytes(r.content)
            channels = parse_channels_xml(r.content)
            if channels:
                records.append(channels_to_record(sid, channels, retrieved))
        except Exception as exc:  # one failing station must not stop the poll
            records.append({"station_id": sid, "observed_ts": None, "retrieved_at": retrieved,
                            "lins": None, "lcins": None, "battery_v": None,
                            "raw_json": json.dumps({"error": str(exc)})})
        time.sleep(0.5)  # be gentle with the public portal
    return records
