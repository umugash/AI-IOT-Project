"""Spatiotemporal analytics and hotspot labelling over station readings.

Hotspot definition (used consistently across the product):
    exceedance = hourly LAeq - applicable CPCB ambient limit (zone, Day/Night)
    Compliant  : exceedance <= 0 dB
    Exceeding  : 0 < exceedance <= HOTSPOT_MARGIN_DB
    Hotspot    : exceedance >  HOTSPOT_MARGIN_DB  (default 10 dB)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .acoustics import leq_agg
from .config import STATIONS, limit_for, period_for
from .cpcb import flag_flatlines

HOTSPOT_MARGIN_DB = 10.0
MIN_READINGS_PER_HOUR = 3  # of 6 possible 10-min readings
STATUS_ORDER = ["Compliant", "Exceeding", "Hotspot"]
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def status_for(exceedance: float, margin: float = HOTSPOT_MARGIN_DB) -> str:
    if pd.isna(exceedance):
        return "Unknown"
    if exceedance <= 0:
        return "Compliant"
    return "Hotspot" if exceedance > margin else "Exceeding"


def clean_readings(readings: pd.DataFrame) -> pd.DataFrame:
    """Apply stuck-sensor detection and keep only valid readings."""
    df = flag_flatlines(readings)
    return df[df["quality"] == "valid"].copy()


def quality_summary(readings: pd.DataFrame) -> pd.DataFrame:
    df = flag_flatlines(readings)
    out = (df.groupby("station_id")
             .agg(readings=("ts", "size"), first=("ts", "min"), last=("ts", "max"),
                  metric=("metric", "first"),
                  valid_pct=("quality", lambda q: 100 * (q == "valid").mean()),
                  flatline_pct=("quality", lambda q: 100 * (q == "flatline_suspect").mean()))
             .reset_index())
    out.insert(1, "station", out["station_id"].map(lambda s: STATIONS[s].label))
    return out


def hourly_levels(readings: pd.DataFrame, margin: float = HOTSPOT_MARGIN_DB) -> pd.DataFrame:
    """Energy-average valid 10-min readings into hourly LAeq with limits and hotspot status.

    10-min readings are end-labelled (00:10 covers 00:00-00:10), so each reading is
    assigned to the hour of its interval start.
    """
    df = readings.copy()
    df["hour_start"] = (df["ts"] - pd.Timedelta(minutes=10)).dt.floor("h")
    h = (df.groupby(["station_id", "hour_start"])
           .agg(laeq=("laeq", leq_agg), n=("laeq", "size"), lmax=("lmax", "max"))
           .reset_index())
    h = h[h["n"] >= MIN_READINGS_PER_HOUR].copy()
    h["hour"] = h["hour_start"].dt.hour
    h["dow"] = h["hour_start"].dt.dayofweek
    h["weekday"] = h["dow"].map(dict(enumerate(WEEKDAYS)))
    h["is_weekend"] = h["dow"] >= 5
    h["date"] = h["hour_start"].dt.date
    h["station"] = h["station_id"].map(lambda s: STATIONS[s].label)
    h["zone"] = h["station_id"].map(lambda s: STATIONS[s].zone)
    h["period"] = h["hour"].map(period_for)
    h["limit"] = [limit_for(z, hr) for z, hr in zip(h["zone"], h["hour"])]
    h["exceedance"] = h["laeq"] - h["limit"]
    h["status"] = h["exceedance"].map(lambda e: status_for(e, margin))
    h["is_hotspot"] = h["status"] == "Hotspot"
    return h.reset_index(drop=True)


def day_night_levels(readings: pd.DataFrame) -> pd.DataFrame:
    """Daily Lday (06-22) and Lnight (22-06) per station, as CPCB reports them."""
    df = readings.copy()
    start = df["ts"] - pd.Timedelta(minutes=10)
    df["period"] = start.dt.hour.map(period_for)
    # A night belongs to the date on which it starts (22:00 .. 06:00 next day).
    df["date"] = np.where((df["period"] == "Night") & (start.dt.hour < 6),
                          (start - pd.Timedelta(days=1)).dt.date, start.dt.date)
    out = (df.groupby(["station_id", "date", "period"])
             .agg(leq=("laeq", leq_agg), n=("laeq", "size")).reset_index())
    expected = out["period"].map({"Day": 96, "Night": 48})
    out = out[out["n"] >= 0.75 * expected].copy()
    out["station"] = out["station_id"].map(lambda s: STATIONS[s].label)
    out["zone"] = out["station_id"].map(lambda s: STATIONS[s].zone)
    out["limit"] = [STATIONS[s].day_limit if p == "Day" else STATIONS[s].night_limit
                    for s, p in zip(out["station_id"], out["period"])]
    out["exceedance"] = out["leq"] - out["limit"]
    return out


def station_summary(hourly: pd.DataFrame) -> pd.DataFrame:
    """Per-station hotspot profile: levels, exceedance, hotspot share, worst hour."""
    rows = []
    for sid, g in hourly.groupby("station_id"):
        st = STATIONS[sid]
        prof = g.groupby("hour")["exceedance"].mean()
        rows.append({
            "station_id": sid, "station": st.label, "zone": st.zone,
            "lat": st.lat, "lon": st.lon,
            "hours": len(g),
            "period": f"{g['hour_start'].min():%d %b %Y} – {g['hour_start'].max():%d %b %Y}",
            "lday": leq_agg(g.loc[g["period"] == "Day", "laeq"]),
            "lnight": leq_agg(g.loc[g["period"] == "Night", "laeq"]),
            "day_limit": st.day_limit, "night_limit": st.night_limit,
            "median_hourly_laeq": g["laeq"].median(),
            "mean_exceedance": g["exceedance"].mean(),
            "exceeding_pct": 100 * (g["exceedance"] > 0).mean(),
            "hotspot_pct": 100 * g["is_hotspot"].mean(),
            "worst_hour": int(prof.idxmax()),
        })
    out = pd.DataFrame(rows)
    # Hotspot index (0-100): share of hotspot hours weighted by how far limits are exceeded.
    sev = out["mean_exceedance"].clip(lower=0) / 20.0
    out["hotspot_index"] = (0.6 * out["hotspot_pct"] + 40 * sev.clip(upper=1)).round(1)
    return out.sort_values("hotspot_index", ascending=False).reset_index(drop=True)


def hour_weekday_matrix(hourly: pd.DataFrame, value: str = "laeq") -> pd.DataFrame:
    """Hour x weekday matrix (energy mean for levels, arithmetic mean otherwise)."""
    agg = leq_agg if value == "laeq" else "mean"
    m = hourly.pivot_table(index="hour", columns="weekday", values=value, aggfunc=agg)
    return m.reindex(index=range(24), columns=WEEKDAYS)


def hotspot_episodes(hourly: pd.DataFrame, min_hours: int = 2) -> pd.DataFrame:
    """Contiguous runs of hotspot hours per station (emergence -> persistence -> dissipation)."""
    rows = []
    for sid, g in hourly.sort_values("hour_start").groupby("station_id"):
        g = g.set_index("hour_start")
        full = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
        flag = full["is_hotspot"].astype("boolean").fillna(False).astype(bool)
        run_id = (flag != flag.shift()).cumsum()
        for _, run in full[flag].groupby(run_id[flag]):
            if len(run) < min_hours:
                continue
            rows.append({"station_id": sid, "station": STATIONS[sid].label,
                         "start": run.index.min(), "end": run.index.max() + pd.Timedelta(hours=1),
                         "hours": len(run), "peak_laeq": run["laeq"].max(),
                         "mean_exceedance": run["exceedance"].mean()})
    return pd.DataFrame(rows, columns=["station_id", "station", "start", "end", "hours",
                                       "peak_laeq", "mean_exceedance"])
