"""Dashboard pages."""
from __future__ import annotations

import io
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from noisescope import analytics, audio, cpcb, store
from noisescope.acoustics import leq
from noisescope.config import DATA_DIR, STATIONS, ZONE_LIMITS, limit_for
from noisescope.edge import EdgeNode, node_summary, publish, source_mix

from . import data as D

CHENNAI_CENTER = {"lat": 13.025, "lon": 80.24}
# Set NOISESCOPE_MAP_STYLE=white-bg to run fully offline (no basemap tiles).
MAP_STYLE = os.environ.get("NOISESCOPE_MAP_STYLE", "carto-positron")


def _map(df: pd.DataFrame, color: str, size: str | None, hover: list[str], color_map=None,
         color_scale=None, range_color=None, height: int = 460, zoom: float = 10.0, center=None):
    fig = px.scatter_map(df, lat="lat", lon="lon", color=color, size=size, size_max=28,
                         hover_name="station" if "station" in df else None, hover_data=hover,
                         color_discrete_map=color_map, color_continuous_scale=color_scale,
                         range_color=range_color, zoom=zoom, center=center or CHENNAI_CENTER,
                         map_style=MAP_STYLE, height=height)
    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.02))
    return fig


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------

def overview() -> None:
    st.title("NoiseScope Chennai")
    st.markdown("**AIoT-based urban sound source classification and spatiotemporal noise hotspot "
                "prediction** — real CPCB acoustic sensor data for Chennai, machine-learning "
                "source classification at the edge, and hotspot forecasting.")
    h, summ = D.hourly(), D.station_summary()
    clf = D.classifier()
    c = st.columns(4)
    c[0].metric("CPCB stations", f"{h['station_id'].nunique()}", help="Chennai stations of the National Ambient Noise Monitoring Network")
    c[1].metric("Station-hours analysed", f"{len(h):,}")
    c[2].metric("Hours above legal limit", f"{100 * (h['exceedance'] > 0).mean():.0f}%")
    c[3].metric("Sound classifier accuracy", f"{100 * clf.report['accuracy']:.0f}%" if clf else "not trained",
                help="ESC-50 5-fold cross-validation, 8 urban source categories")

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Where are the hotspots?")
        m = summ.copy()
        m["Hotspot index"] = m["hotspot_index"]
        m["size"] = m["hotspot_index"].clip(lower=5)
        fig = _map(m, color="Hotspot index", size="size",
                   hover=["zone", "lday", "lnight", "hotspot_pct", "period"],
                   color_scale=["#2e9e5b", "#f0a202", "#d8343b"], range_color=[0, 100])
        fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Zone %{customdata[0]}<br>"
                                        "Lday %{customdata[1]:.1f} dB · Lnight %{customdata[2]:.1f} dB"
                                        "<br>Hotspot hours %{customdata[3]:.0f}%<br>%{customdata[4]}<extra></extra>")
        st.plotly_chart(fig, width="stretch")
        st.caption("Marker positions are approximate locality points, not surveyed sensor positions.")
    with right:
        st.subheader("Hotspot ranking")
        bar = px.bar(summ.sort_values("hotspot_index"), x="hotspot_index", y="label", orientation="h",
                     color="hotspot_index", color_continuous_scale=["#2e9e5b", "#f0a202", "#d8343b"],
                     range_color=[0, 100], labels={"hotspot_index": "Hotspot index (0–100)", "label": ""},
                     hover_data={"zone": True, "hotspot_pct": ":.0f", "mean_exceedance": ":.1f",
                                 "valid_pct": ":.0f"}, height=440)
        bar.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=10, t=10, b=0))
        st.plotly_chart(bar, width="stretch")
        if (~summ["reliable"]).any():
            st.caption(f"⚠ = under {D.RELIABLE_VALID_PCT:.0f}% valid readings (stuck sensor); treat with caution.")

    st.subheader("Key findings from the data")
    good = summ[summ["reliable"]]
    worst = good.iloc[0]
    unreliable = summ.loc[~summ["reliable"], "station"].tolist()
    night = good.assign(gap=good["lnight"] - good["night_limit"]).sort_values("gap", ascending=False).iloc[0]
    city_profile = h.groupby("hour")["exceedance"].mean()
    st.markdown(
        f"- Among stations with reliable data, **{worst['station']}** ({worst['zone']} zone) is the strongest hotspot: {worst['hotspot_pct']:.0f}% of hours "
        f"exceed the limit by more than {analytics.HOTSPOT_MARGIN_DB:.0f} dB (Lday {worst['lday']:.1f} dB vs limit {worst['day_limit']} dB).\n"
        f"- Night-time is the biggest compliance gap: **{night['station']}** has Lnight {night['lnight']:.1f} dB against a "
        f"night limit of {night['night_limit']} dB.\n"
        f"- Across the city, exceedance peaks around **{int(city_profile.idxmax()):02d}:00** and is lowest around "
        f"**{int(city_profile.idxmin()):02d}:00**.\n"
        f"- Silence zones (hospitals, institutions) are among the noisiest relative to their limits — the sensitive sites "
        f"get the least protection.\n"
        + (f"- Sensor faults: {', '.join(unreliable)} had long stuck (flat-lined) stretches. Those readings are removed "
           f"automatically, but what remains is less trustworthy (marked ⚠)."
           if unreliable else ""))
    D.footer_note()


# ---------------------------------------------------------------------------
# Live network
# ---------------------------------------------------------------------------

def live() -> None:
    st.title("Live CPCB network")
    st.markdown("Latest readings from the CPCB National Ambient Noise Monitoring Network portal "
                "(cpcbnoise.com, public guest access). Keep `python scripts/poll_cpcb_live.py` running "
                "on your laptop to collect a reading every 10 minutes.")
    col1, col2 = st.columns([1, 3])
    if col1.button("Poll CPCB now", type="primary"):
        with st.spinner("Contacting cpcbnoise.com (about 10 s) ..."):
            recs = cpcb.poll_live()
        good = cpcb.good_records(recs)
        if good:
            store.upsert_live_status(good)
            store.upsert_readings(cpcb.live_records_to_readings(good))
            D.refresh()
        msg = f"{len(good)}/{len(recs)} stations returned a reading."
        (col2.success if good else col2.warning)(
            msg + ("" if good else " Earlier readings are kept. The portal may refuse requests from "
                   "cloud servers outside India; run the poller on your laptop instead."))
        failed = [r for r in recs if r.get("status") != "ok"]
        if failed:
            with st.expander(f"Details for {len(failed)} station(s) without a reading"):
                st.dataframe(pd.DataFrame([{"Station": STATIONS[r["station_id"]].label, "Result": r["status"],
                                            "Detail": r["detail"]} for r in failed]),
                             hide_index=True, width="stretch")
    ls = D.live_status()
    if ls.empty:
        st.info("No live data yet. Click **Poll CPCB now** or run the poller script.")
        return
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    ls = ls.copy()
    ls["age_h"] = (now - ls["observed_ts"]).dt.total_seconds() / 3600
    ls["Freshness"] = np.select([ls["age_h"] <= 1, ls["age_h"] <= 24], ["Live", "Today"], "Offline")
    ls["limit_now"] = [limit_for(z, t.hour) if pd.notna(t) else np.nan for z, t in zip(ls["zone"], ls["observed_ts"])]
    ls["exceedance"] = ls["lins"] - ls["limit_now"]
    ls["Status"] = ls["exceedance"].map(analytics.status_for)
    ls["lat"] = ls["station_id"].map(lambda s: STATIONS[s].lat)
    ls["lon"] = ls["station_id"].map(lambda s: STATIONS[s].lon)
    c = st.columns(3)
    c[0].metric("Stations reporting in last hour", int((ls["Freshness"] == "Live").sum()))
    c[1].metric("Offline > 24 h", int((ls["Freshness"] == "Offline").sum()))
    c[2].metric("Last poll (UTC)", str(ls["retrieved_at"].max())[:16].replace("T", " "))
    fresh = ls[ls["Freshness"] != "Offline"]
    if not fresh.empty:
        fig = _map(fresh.assign(size=fresh["lins"].clip(lower=30) - 25), color="Status", size="size",
                   hover=["lins", "limit_now", "observed_ts"], color_map=D.STATUS_COLORS, height=380)
        st.plotly_chart(fig, width="stretch")
    table = ls.sort_values("age_h")[["station", "zone", "observed_ts", "lins", "limit_now", "exceedance",
                                      "Status", "Freshness", "battery_v"]]
    table["Last reading"] = table["observed_ts"].map(D.fmt_age)
    st.dataframe(table.drop(columns="observed_ts").rename(columns={
        "station": "Station", "zone": "Zone", "lins": "LAF (dB(A))", "limit_now": "Limit now",
        "exceedance": "Above limit (dB)", "battery_v": "Battery (V)"}),
        hide_index=True, width="stretch",
        column_config={"LAF (dB(A))": st.column_config.NumberColumn(format="%.1f"),
                       "Above limit (dB)": st.column_config.NumberColumn(format="%+.1f")})
    st.caption("LAF = instantaneous A-weighted level at the time of the latest 10-minute record. "
               "Several stations have not reported for months; they are shown as Offline.")


# ---------------------------------------------------------------------------
# Spatiotemporal patterns
# ---------------------------------------------------------------------------

def patterns() -> None:
    st.title("Spatiotemporal patterns")
    h = D.hourly()
    st.markdown("Hourly LAeq (energy average of valid 10-minute readings) compared with the CPCB limit "
                f"for each station's zone. **Hotspot hour** = more than {analytics.HOTSPOT_MARGIN_DB:.0f} dB above the limit.")

    st.subheader("City view: when and where limits are exceeded")
    mat = h.pivot_table(index="station", columns="hour", values="exceedance", aggfunc="mean")
    mat = mat.loc[D.station_summary()["station"]]
    fig = px.imshow(mat, color_continuous_scale="RdYlGn_r", color_continuous_midpoint=0, aspect="auto",
                    labels=dict(x="Hour of day", y="", color="dB above limit"), height=380)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")

    st.subheader("Station explorer")
    labels = {STATIONS[s].label: s for s in sorted(h["station_id"].unique())}
    name = st.selectbox("Station", list(labels), index=list(labels).index(D.station_summary().iloc[0]["station"]))
    sid = labels[name]
    g = h[h["station_id"] == sid].sort_values("hour_start")
    st.caption(f"{STATIONS[sid].zone} zone · limits {STATIONS[sid].day_limit} dB day / {STATIONS[sid].night_limit} dB night · "
               f"{len(g)} valid hours, {g['hour_start'].min():%d %b %Y} – {g['hour_start'].max():%d %b %Y}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=g["hour_start"], y=g["laeq"], mode="lines", name="Hourly LAeq",
                             line=dict(color="#4c78a8", width=1.2)))
    for status, color in D.STATUS_COLORS.items():
        sub = g[g["status"] == status]
        if len(sub):
            fig.add_trace(go.Scatter(x=sub["hour_start"], y=sub["laeq"], mode="markers", name=status,
                                     marker=dict(color=color, size=4)))
    fig.add_trace(go.Scatter(x=g["hour_start"], y=g["limit"], mode="lines", name="Legal limit",
                             line=dict(color="#333", dash="dot", shape="hv")))
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="dB(A)",
                      legend=dict(orientation="h", y=1.08))
    st.plotly_chart(fig, width="stretch")

    a, b = st.columns(2)
    with a:
        st.markdown("**Weekly rhythm — LAeq by hour and weekday**")
        mat = analytics.hour_weekday_matrix(g, "laeq")
        fig = px.imshow(mat, color_continuous_scale="Inferno", aspect="auto",
                        labels=dict(x="", y="Hour", color="dB(A)"), height=420)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**Daily Lday / Lnight vs limits**")
        dn = analytics.day_night_levels(analytics.clean_readings(D.readings()))
        dn = dn[dn["station_id"] == sid]
        fig = px.line(dn, x="date", y="leq", color="period", markers=True,
                      color_discrete_map={"Day": "#f58518", "Night": "#4c78a8"}, height=420,
                      labels={"leq": "dB(A)", "date": ""})
        fig.add_hline(y=STATIONS[sid].day_limit, line_dash="dot", line_color="#f58518", annotation_text="Day limit")
        fig.add_hline(y=STATIONS[sid].night_limit, line_dash="dot", line_color="#4c78a8", annotation_text="Night limit")
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.08))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Hotspot episodes (emergence → persistence → dissipation)")
    ep = analytics.hotspot_episodes(h, min_hours=3)
    if ep.empty:
        st.info("No hotspot episodes of 3+ hours.")
    else:
        stats = ep.groupby("station").agg(episodes=("hours", "size"), mean_duration_h=("hours", "mean"),
                                          longest_h=("hours", "max"), peak_laeq=("peak_laeq", "max")).reset_index()
        x, y = st.columns([2, 3])
        x.dataframe(stats.sort_values("episodes", ascending=False), hide_index=True, width="stretch",
                    column_config={"mean_duration_h": st.column_config.NumberColumn(format="%.1f"),
                                   "peak_laeq": st.column_config.NumberColumn(format="%.1f")})
        epi = ep.copy()
        epi["start_hour"] = epi["start"].dt.hour
        fig = px.histogram(epi, x="start_hour", color="station", nbins=24, height=320,
                           labels={"start_hour": "Hour hotspot episodes begin"})
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), bargap=0.05)
        y.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Hotspot forecast
# ---------------------------------------------------------------------------

def forecast() -> None:
    st.title("Hotspot prediction")
    sig = D.data_signature()
    fc = D.latest_forecast(sig)
    h = D.hourly()
    st.markdown("A LightGBM model forecasts each station's hourly LAeq for the next 24 hours from its recent "
                "history, daily rhythm and zone limit, and gives the probability that each hour will be a hotspot "
                f"(> {analytics.HOTSPOT_MARGIN_DB:.0f} dB above the limit).")
    issued = fc.groupby("station")["issue_time"].first()
    oldest = issued.min()
    now = pd.Timestamp.now(tz="Asia/Kolkata").tz_localize(None)
    if (now - issued.max()).total_seconds() > 6 * 3600:
        st.warning(f"Forecasts are issued from each station's **latest available data** (most recent: "
                   f"{issued.max():%d %b %Y %H:%M}). Run the live poller to forecast from current conditions.")

    hz = st.slider("Hours ahead", 1, 24, 3)
    f = fc[fc["horizon"] == hz].copy()
    f["lat"] = f["station_id"].map(lambda s: STATIONS[s].lat)
    f["lon"] = f["station_id"].map(lambda s: STATIONS[s].lon)
    f["Hotspot probability"] = (100 * f["p_hotspot"]).round(0)
    f["size"] = f["Hotspot probability"].clip(lower=8)
    if True:
        fig = _map(f, color="Hotspot probability", size="size",
                   hover=["pred_laeq", "target_limit", "target_time"],
                   color_scale=["#2e9e5b", "#f0a202", "#d8343b"], range_color=[0, 100], height=420)
        fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>Predicted %{customdata[0]:.1f} dB(A)"
                                        "<br>Limit %{customdata[1]} dB<br>For %{customdata[2]}<extra></extra>")
        st.plotly_chart(fig, width="stretch")
    if True:
        t = f.sort_values("p_hotspot", ascending=False)[["station", "target_time", "pred_laeq", "pred_lo", "pred_hi",
                                                          "target_limit", "Hotspot probability", "status"]]
        st.dataframe(t, hide_index=True, width="stretch", height=420, column_config={
            "target_time": st.column_config.DatetimeColumn("For hour", format="DD MMM HH:mm"),
            "pred_laeq": st.column_config.NumberColumn("Predicted dB", format="%.1f"),
            "pred_lo": st.column_config.NumberColumn("Low", format="%.1f"),
            "pred_hi": st.column_config.NumberColumn("High", format="%.1f"),
            "target_limit": "Limit",
            "Hotspot probability": st.column_config.ProgressColumn("P(hotspot)", format="%d%%", min_value=0, max_value=100),
            "status": "Expected status"})

    st.subheader("24-hour outlook for one station")
    name = st.selectbox("Station", sorted(fc["station"].unique()), key="fc_station")
    s = fc[fc["station"] == name]
    past = h[(h["station"] == name)].sort_values("hour_start").tail(48)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=past["hour_start"], y=past["laeq"], name="Observed", line=dict(color="#4c78a8")))
    fig.add_trace(go.Scatter(x=list(s["target_time"]) + list(s["target_time"][::-1]),
                             y=list(s["pred_hi"]) + list(s["pred_lo"][::-1]), fill="toself",
                             fillcolor="rgba(216,52,59,0.15)", line=dict(width=0), name="80% range"))
    fig.add_trace(go.Scatter(x=s["target_time"], y=s["pred_laeq"], name="Forecast", line=dict(color="#d8343b")))
    lim = pd.concat([past[["hour_start", "limit"]].rename(columns={"hour_start": "t"}),
                     s[["target_time", "target_limit"]].rename(columns={"target_time": "t", "target_limit": "limit"})])
    fig.add_trace(go.Scatter(x=lim["t"], y=lim["limit"], name="Limit", line=dict(color="#333", dash="dot", shape="hv")))
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="dB(A)",
                      legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig, width="stretch")

    st.subheader("How good is the model? (chronological backtest)")
    ev = D.evaluation(sig)
    st.markdown(f"Each station's first 70% of hours trained the model; the last 30% "
                f"({ev['n_test']:,} forecasts) were held out and predicted without peeking ahead.")
    a, b = st.columns(2)
    with a:
        fig = px.line(ev["regression"], x="horizon_h", y="MAE_dB", color="method", markers=True, height=340,
                      labels={"horizon_h": "Hours ahead", "MAE_dB": "Mean absolute error (dB)", "method": ""},
                      color_discrete_map={"NoiseScope LightGBM": "#d8343b"})
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(fig, width="stretch")
    with b:
        cls = ev["classification"].merge(ev["interval_coverage_80"], on="horizon_h")
        st.dataframe(cls[["horizon_h", "AUC", "F1_model", "F1_persistence", "Brier", "coverage"]].rename(columns={
            "horizon_h": "Hours ahead", "F1_model": "F1 (model)", "F1_persistence": "F1 (persistence)",
            "coverage": "80% range coverage"}), hide_index=True, width="stretch",
            column_config={c: st.column_config.NumberColumn(format="%.3f")
                           for c in ["AUC", "F1 (model)", "F1 (persistence)", "Brier", "80% range coverage"]})
        st.caption("AUC: how well hotspot probabilities rank hotspot vs normal hours (1.0 = perfect). "
                   "Persistence = 'the next hours will look like the last hour'.")
    with st.expander("Backtest detail: predicted vs actual"):
        tp = ev["test_predictions"]
        tp = tp.assign(station=tp["station_id"].map(lambda x: STATIONS[x].label))
        c1, c2 = st.columns(2)
        st_name = c1.selectbox("Station", sorted(tp["station"].unique()), key="bt_station")
        hz2 = c2.selectbox("Hours ahead", sorted(tp["horizon"].unique()), index=3, key="bt_h")
        sub = tp[(tp["station"] == st_name) & (tp["horizon"] == hz2)].sort_values("target_time")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sub["target_time"], y=sub["target"], name="Actual", line=dict(color="#4c78a8")))
        fig.add_trace(go.Scatter(x=sub["target_time"], y=sub["pred_laeq"], name="Predicted", line=dict(color="#d8343b")))
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="dB(A)")
        st.plotly_chart(fig, width="stretch")
        fi = ev["feature_importance"]
        st.plotly_chart(px.bar(fi, x="gain_share", y="feature", orientation="h", height=380,
                               labels={"gain_share": "Share of model gain", "feature": ""}).update_layout(
            yaxis=dict(autorange="reversed"), margin=dict(l=0, r=0, t=10, b=0)), width="stretch")


# ---------------------------------------------------------------------------
# Sound source classification
# ---------------------------------------------------------------------------

def _spectrogram_fig(y: np.ndarray, sr: int):
    import librosa
    S = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=1024), ref=np.max)
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=1024)
    f = librosa.mel_frequencies(n_mels=64, fmax=sr / 2)
    fig = go.Figure(go.Heatmap(z=S, x=t, y=f, colorscale="Magma", showscale=False))
    fig.update_layout(height=220, margin=dict(l=0, r=0, t=10, b=0), xaxis_title="seconds",
                      yaxis=dict(type="log", title="Hz"))
    return fig


def sound() -> None:
    st.title("Sound source classification")
    clf = D.classifier()
    if clf is None:
        st.error("No trained classifier found. Run `python scripts/train_sound_classifier.py`.")
        return
    st.markdown("Record with your microphone or upload a recording. The audio is cut into 5-second windows; "
                "each window gets an A-weighted level and a predicted urban sound source. This is exactly what "
                "an edge node does before sending compact events instead of raw audio.")
    tab_rec, tab_up, tab_demo = st.tabs(["🎙️ Record", "📁 Upload", "▶️ Demo street mix"])
    clip, origin = None, None
    with tab_rec:
        rec = st.audio_input("Record a sound sample")
        if rec is not None:
            clip, origin = rec.getvalue(), "microphone"
    with tab_up:
        up = st.file_uploader("Audio file", type=["wav", "ogg", "flac", "mp3", "m4a"])
        if up is not None:
            clip, origin = up.getvalue(), up.name
    with tab_demo:
        demo_path = DATA_DIR / "demo" / "demo_street_mix.ogg"
        st.caption("60 s mix of ESC-50 clips (traffic, horns, siren, chainsaw, people, bells, dog). "
                   "These clips were also used for training, so this demonstrates the pipeline, not accuracy.")
        if demo_path.exists() and st.toggle("Analyse the demo street mix"):
            clip, origin = demo_path.read_bytes(), demo_path.name

    if clip is not None:
        y = audio.load_audio(io.BytesIO(clip))
        st.audio(clip)
        cal = st.number_input("Calibration offset (dB): add to dBFS to get dB SPL. Leave 0 if unknown.",
                              value=0.0, step=1.0)
        frames = audio.analyse_signal(y, clf, calibration_db=cal if cal else None)
        if frames.empty:
            st.warning("Recording too short (needs at least 0.5 s).")
            return
        st.plotly_chart(_spectrogram_fig(y, audio.SR), width="stretch")
        frames["window"] = frames["offset_s"].map(lambda s: f"{s:.0f}–{s + audio.CLIP_SECONDS:.0f}s")
        color_map = {**audio.CATEGORY_COLORS, "Quiet / background": "#bbbbbb"}
        a, b = st.columns([3, 2])
        with a:
            fig = px.bar(frames, x="window", y=frames["level_db"] - frames["level_db"].min() + 3, color="source_class",
                         color_discrete_map=color_map, height=320,
                         hover_data={"level_db": ":.1f", "confidence": ":.2f"},
                         labels={"y": "Relative level (dB)", "window": "", "source_class": "Source"},
                         category_orders={"window": frames["window"].tolist()})
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=-0.3))
            st.plotly_chart(fig, width="stretch")
        with b:
            top = frames["source_class"].value_counts()
            st.metric("Dominant source", top.index[0])
            st.metric("Equivalent level", f"{leq(frames['level_db']):.1f} {'dB SPL' if cal else 'dBFS'}")
            st.dataframe(frames[["window", "source_class", "confidence", "level_db"]], hide_index=True,
                         width="stretch", height=180,
                         column_config={"confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f"),
                                        "level_db": st.column_config.NumberColumn("level", format="%.1f")})

        with st.expander("Send these events to the AIoT network as an edge node"):
            c = st.columns(3)
            node_id = c[0].text_input("Node id", "my-node-1")
            lat = c[1].number_input("Latitude", value=13.0418, format="%.5f")
            lon = c[2].number_input("Longitude", value=80.2341, format="%.5f")
            start = st.text_input("Recording start (IST)", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            if st.button("Log events"):
                node = EdgeNode(node_id, lat, lon, cal if cal else None)
                ev = node.events_from_signal(y, datetime.strptime(start, "%Y-%m-%d %H:%M:%S"), origin, clf)
                publish(ev)
                D.refresh()
                st.success(f"Logged {len(ev)} events for node {node_id}. See the Edge sensor network page.")

    st.subheader("Model card")
    r = clf.report
    c = st.columns(4)
    c[0].metric("Accuracy", f"{100 * r['accuracy']:.1f}%")
    c[1].metric("Balanced accuracy", f"{100 * r['balanced_accuracy']:.1f}%")
    c[2].metric("Macro F1", f"{r['macro_f1']:.3f}")
    c[3].metric("Clips", f"{r['n_clips']:,}")
    a, b = st.columns([3, 2])
    with a:
        cm = np.array(r["confusion"], dtype=float)
        cmn = cm / cm.sum(axis=1, keepdims=True)
        fig = px.imshow(cmn, x=r["classes"], y=r["classes"], color_continuous_scale="Blues", text_auto=".2f",
                        labels=dict(x="Predicted", y="True", color="share"), height=470)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
    with b:
        rows = [{"Category": c, "ESC-50 classes": ", ".join(audio.URBAN_CATEGORIES[c]),
                 "Recall": r["per_class"][c]["recall"], "Precision": r["per_class"][c]["precision"]}
                for c in r["classes"]]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=470,
                     column_config={"Recall": st.column_config.NumberColumn(format="%.2f"),
                                    "Precision": st.column_config.NumberColumn(format="%.2f")})
    st.caption("Training data: ESC-50 (K. J. Piczak, CC BY-NC 3.0), 1,160 clips in 8 urban categories, "
               "with pitch/time-shift and street-background augmentation. Scores are 5-fold cross-validation on "
               "the official folds. Model: LightGBM on log-mel, MFCC and spectral statistics.")


# ---------------------------------------------------------------------------
# Edge network
# ---------------------------------------------------------------------------

# Demo nodes re-use 5-second windows of the demo mix. Window order in the mix:
# 0 engine, 1 horn, 2 engine, 3 siren, 4 horn, 5 chainsaw, 6 footsteps, 7 laughing,
# 8 horn, 9 bells, 10 dog, 11 engine.
DEMO_NODES = [
    ("DEMO-tnagar-junction", 13.0418, 80.2341, 92.0, [0, 1, 2, 4, 8, 3, 11, 1, 7]),
    ("DEMO-guindy-estate", 13.0067, 80.2206, 90.0, [5, 0, 5, 11, 2, 5, 6, 0]),
    ("DEMO-srm-campus", 12.8231, 80.0442, 72.0, [6, 7, 9, 10, 6, 7, 0]),
]


def _load_demo_nodes() -> int:
    clf = D.classifier()
    y = audio.load_audio(str(DATA_DIR / "demo" / "demo_street_mix.ogg"))
    win = int(audio.CLIP_SECONDS * audio.SR)
    rng = np.random.default_rng(1)
    total = 0
    for node_id, lat, lon, cal, windows in DEMO_NODES:
        for hour in [7, 9, 13, 18, 22]:
            order = rng.permutation(windows)
            seg = np.concatenate([y[i * win:(i + 1) * win] * rng.uniform(0.4, 1.0) for i in order])
            ev = EdgeNode(node_id, lat, lon, cal + rng.normal(0, 2)).events_from_signal(
                seg, datetime(2026, 11, 3, hour, 0, 0), "demo_street_mix.ogg", clf)
            total += publish(ev)
    return total


def edge() -> None:
    st.title("Edge sensor network (AIoT)")
    st.markdown("CPCB stations tell us **where and when** it is loud. Edge nodes — a phone, laptop or "
                "Raspberry Pi with a microphone running `scripts/edge_node.py` — tell us **what is causing it**: "
                "they classify sound sources on the device and send only small events (time, place, level, source).")
    ev = D.edge_events()
    c = st.columns([1, 1, 3])
    if c[0].button("Load demo nodes"):
        with st.spinner("Running three demo nodes ..."):
            n = _load_demo_nodes()
        D.refresh()
        st.success(f"Created {n} demo events.")
        ev = D.edge_events()
    if c[1].button("Clear demo nodes"):
        for node_id, *_ in DEMO_NODES:
            store.clear_edge_events(node_id)
        D.refresh()
        ev = D.edge_events()
    if ev.empty:
        st.info("No edge events yet. Record audio on the **Sound source classification** page and log it, run "
                "`python scripts/edge_node.py ...` on a recording, or click **Load demo nodes**.")
        return
    if ev["node_id"].str.startswith("DEMO-").any():
        st.warning("Nodes named DEMO-… use a synthetic ESC-50 mix and illustrative calibration — not field measurements.")
    ns = node_summary(ev)
    mix = source_mix(ev)
    a, b = st.columns([3, 2])
    with a:
        m = ns.dropna(subset=["lat", "lon"]).rename(columns={"node_id": "station"})
        if not m.empty:
            m["size"] = (m["leq_db"] - m["leq_db"].min() + 5)
            fig = _map(m, color="dominant_source", size="size", hover=["leq_db", "man_made_energy_share", "windows"],
                       color_map=audio.CATEGORY_COLORS, height=420, zoom=9.0,
                       center={"lat": float(m["lat"].mean()), "lon": float(m["lon"].mean())})
            st.plotly_chart(fig, width="stretch")
    with b:
        st.dataframe(ns[["node_id", "leq_db", "dominant_source", "man_made_energy_share", "windows", "calibrated"]],
                     hide_index=True, width="stretch", column_config={
                         "leq_db": st.column_config.NumberColumn("Leq (dB)", format="%.1f"),
                         "man_made_energy_share": st.column_config.ProgressColumn("Man-made share", min_value=0, max_value=1, format="%.2f")})
    st.subheader("What makes each place loud? (share of acoustic energy by source)")
    fig = px.bar(mix, x="energy_share", y="node_id", color="source_class", orientation="h",
                 color_discrete_map=audio.CATEGORY_COLORS, height=80 + 60 * mix["node_id"].nunique(),
                 labels={"energy_share": "Share of acoustic energy", "node_id": "", "source_class": "Source"})
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), xaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")
    st.subheader("Source timeline")
    node = st.selectbox("Node", sorted(ev["node_id"].unique()))
    sub = ev[ev["node_id"] == node].sort_values("ts")
    fig = px.scatter(sub, x="ts", y="level_db", color="source_class", color_discrete_map=audio.CATEGORY_COLORS,
                     height=320, labels={"ts": "", "level_db": "Level (dB)", "source_class": "Source"})
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, width="stretch")
    st.download_button("Download events (JSON)", sub.assign(ts=sub["ts"].astype(str)).to_json(orient="records"),
                       file_name=f"{node}_events.json")


# ---------------------------------------------------------------------------
# Methods & data
# ---------------------------------------------------------------------------

def about() -> None:
    st.title("How it works")
    st.graphviz_chart("""
    digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef2ff", fontname="Helvetica", fontsize=11];
      subgraph cluster_sense { label="Sensing layer"; style=dashed;
        cpcb [label="CPCB NANMN stations\\n10 Chennai sites, 10-min LAF"]; mic [label="Edge nodes\\nphone / Pi / laptop mic"]; }
      subgraph cluster_edge { label="Edge AI"; style=dashed;
        feat [label="A-weighted level +\\nlog-mel / MFCC features"]; clf [label="Sound source classifier\\n(LightGBM, 8 urban classes)"]; }
      subgraph cluster_data { label="Ingestion & storage"; style=dashed;
        poll [label="Live poller (10 min)\\n+ CSV export loader"]; qc [label="Quality control\\nrange + stuck-sensor checks"]; db [label="SQLite store\\nreadings · edge_events · live_status", fillcolor="#fef3c7"]; }
      subgraph cluster_ai { label="Analytics & prediction"; style=dashed;
        st [label="Spatiotemporal analysis\\nhourly LAeq, Lday/Lnight, exceedance"]; hs [label="Hotspot labelling\\n> limit + 10 dB"]; fc [label="Hotspot forecaster\\n1–24 h, LightGBM + quantiles"]; }
      dash [label="NoiseScope dashboard", fillcolor="#dcfce7"];
      cpcb -> poll -> qc -> db; mic -> feat -> clf -> db [label=" JSON events"];
      db -> st -> hs -> fc -> dash; st -> dash; db -> dash; }
    """)
    st.subheader("Data sources")
    st.markdown("""
| Source | What it gives | Used for |
|---|---|---|
| **CPCB National Ambient Noise Monitoring Network** (cpcbnoise.com, guest access) | 10-minute LAF readings, 10 Chennai stations; CSV exports and live latest values | Spatiotemporal analysis, hotspots, forecasting, live view |
| **ESC-50** (Piczak 2015, CC BY-NC 3.0) | 2,000 labelled 5-s environmental sound clips | Training the sound source classifier (1,160 clips in 8 urban categories) |
| **Your edge recordings** | Phone/laptop recordings with time and place | Source attribution at your own locations (e.g. SRM campus) |
| TNPCB Annual Report 2023-24, SRMIST 2025 monitoring | Annual / campus reference values | Context only (`data/reference/`) |
""")
    st.subheader("Hotspot definition")
    st.markdown(f"""
For every station-hour: **exceedance = hourly LAeq − CPCB limit** (zone, day 06–22 h / night 22–06 h).
*Compliant* ≤ 0 dB · *Exceeding* 0–{analytics.HOTSPOT_MARGIN_DB:.0f} dB · *Hotspot* > {analytics.HOTSPOT_MARGIN_DB:.0f} dB.
The station **hotspot index** (0–100) combines the share of hotspot hours (60%) with the mean exceedance (40%, saturating at 20 dB).
""")
    st.dataframe(pd.DataFrame([{"Zone": z, "Day limit dB(A)": d, "Night limit dB(A)": n} for z, (d, n) in ZONE_LIMITS.items()]),
                 hide_index=True)
    st.subheader("Data quality per station")
    st.dataframe(D.quality(), hide_index=True, width="stretch", column_config={
        "valid_pct": st.column_config.NumberColumn("Valid %", format="%.1f"),
        "flatline_pct": st.column_config.NumberColumn("Stuck-sensor %", format="%.1f")})
    st.subheader("Limitations")
    st.markdown("""
- The bundled CPCB history is about one month per station, and the months differ by station (May 2025 – Feb 2026). Leave the live poller running to build a longer, aligned record.
- Station coordinates are approximate locality points; CPCB does not publish surveyed sensor positions.
- Triplicane's export has no 10-minute average, so its instantaneous LAF samples are used.
- CPCB stations do not record audio, so sources at CPCB sites cannot be classified directly; source attribution comes from edge nodes.
- Phone microphones are uncalibrated: without a calibration offset, edge levels are relative (dBFS), not dB SPL.
- ESC-50 clips are not recorded in Chennai; recording and labelling local clips (e.g. auto-rickshaw horns, temple bells) would improve the classifier.
""")
