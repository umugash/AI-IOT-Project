"""Regenerate the README / docs figures from the project's data and saved results.

    pip install matplotlib
    python scripts/make_figures.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import _bootstrap  # noqa: F401,E402
from noisescope import analytics, audio, cpcb, forecasting  # noqa: E402
from noisescope.config import CPCB_RAW_DIR, ROOT, STATIONS  # noqa: E402

OUT = ROOT / "docs" / "images"
INK, MUTED, GRID = "#1f2933", "#6b7280", "#e5e7eb"
RED, AMBER, GREEN, BLUE = "#d8343b", "#f0a202", "#2e9e5b", "#4c78a8"

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10.5, "axes.edgecolor": GRID,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titleweight": "bold",
    "axes.titlesize": 13, "axes.titlelocation": "left", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", OUT / name)


def forecast_mae():
    reg = pd.read_csv(forecasting.EVAL_DIR / "regression.csv")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    styles = {"NoiseScope LightGBM": (RED, 2.8, "o"), "Persistence (last hour)": (BLUE, 1.6, "s"),
              "Same hour yesterday": (AMBER, 1.6, "^"), "Station hourly profile": ("#9ca3af", 1.6, "D")}
    for method, (c, lw, mk) in styles.items():
        d = reg[reg["method"] == method]
        ax.plot(d["horizon_h"], d["MAE_dB"], color=c, lw=lw, marker=mk, ms=6, label=method)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 3, 6, 12, 24], ["1 h", "2 h", "3 h", "6 h", "12 h", "24 h"])
    ax.minorticks_off()
    ax.set_ylabel("Mean absolute error (dB)")
    ax.set_xlabel("Forecast horizon")
    ax.set_title("Hotspot forecaster vs. baselines (chronological hold-out)")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.set_ylim(0, reg["MAE_dB"].max() * 1.25)
    save(fig, "forecast_mae.png")


def confusion():
    rep = json.loads(audio.MODEL_PATH.with_suffix(".json").read_text())["report"]
    cm = np.array(rep["confusion"], dtype=float)
    cm = cm / cm.sum(axis=1, keepdims=True)
    classes = rep["classes"]
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(classes)), classes, rotation=40, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j] >= 0.01:
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                        color="white" if cm[i, j] > 0.5 else INK)
    ax.set_xlabel("Predicted source")
    ax.set_ylabel("True source")
    ax.set_title(f"Sound source classifier — {100 * rep['accuracy']:.1f}% accuracy (5-fold CV)")
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "classifier_confusion.png")


def city_heatmap(hourly, order):
    mat = hourly.pivot_table(index="station", columns="hour", values="exceedance", aggfunc="mean").loc[order]
    fig, ax = plt.subplots(figsize=(10, 4.4))
    lim = 30
    im = ax.imshow(mat, cmap="RdYlGn_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.grid(False)
    ax.set_yticks(range(len(mat)), mat.index)
    ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
    ax.set_xlabel("Hour of day (IST)")
    ax.axvline(5.5, color="white", lw=1.5, ls="--")
    ax.axvline(21.5, color="white", lw=1.5, ls="--")
    ax.set_title("When and where Chennai exceeds its noise limits")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("dB above CPCB limit")
    cb.outline.set_visible(False)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "exceedance_heatmap.png")


def ranking(summary, valid):
    s = summary.merge(valid, on="station_id").sort_values("hotspot_index")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    colors = [RED if v >= 60 else AMBER if v >= 25 else GREEN for v in s["hotspot_index"]]
    bars = ax.barh(s["station"], s["hotspot_index"], color=colors, height=0.65)
    for b, ok in zip(bars, s["valid_pct"] >= 70):
        if not ok:
            b.set_hatch("///")
            b.set_alpha(0.45)
    for y, (v, z) in enumerate(zip(s["hotspot_index"], s["zone"])):
        ax.text(v + 1, y, f"{v:.0f}  ·  {z}", va="center", fontsize=9, color=MUTED)
    ax.set_xlim(0, 125)
    ax.set_xlabel("Hotspot index (0–100)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Station hotspot ranking")
    fig.text(0.01, -0.04, "Hatched = stations with long stuck-sensor stretches (under 70% valid readings) — treat with caution",
             fontsize=8.5, color=MUTED)
    save(fig, "hotspot_ranking.png")


def stuck_sensor(readings):
    d = cpcb.flag_flatlines(readings[readings["station_id"] == 265]).set_index("ts")
    d = d["2025-12-05":"2025-12-14"]
    fig, ax = plt.subplots(figsize=(10, 3.4))
    ok = d["quality"] == "valid"
    ax.plot(d.index, d["laeq"].where(ok), color=BLUE, lw=1, label="valid readings")
    ax.plot(d.index, d["laeq"].where(~ok), color=RED, lw=2, label="flagged: stuck sensor")
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.set_ylabel("10-min LAeq, dB(A)")
    ax.set_title("Automatic quality control — T. Nagar sensor freezes on 10 Dec 2025")
    ax.legend(loc="upper right")
    save(fig, "quality_control.png")


def daily_profiles(hourly):
    reliable = ["Egmore Eye Hospital", "Guindy", "Pallikaranai", "Washermanpet", "Sowcarpet", "Triplicane", "Anna Nagar"]
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for i, name in enumerate(reliable):
        prof = hourly[hourly["station"] == name].groupby("hour")["laeq"].median()
        ax.plot(prof.index, prof.values, lw=2, color=colors[i], label=name)
    ax.axvspan(-0.5, 5.5, color="#eef2ff", zorder=0)
    ax.axvspan(21.5, 23.5, color="#eef2ff", zorder=0)
    ax.text(0.2, ax.get_ylim()[1] - 1.5, "night", color=MUTED, fontsize=9)
    ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
    ax.set_xlim(-0.5, 23.5)
    ax.set_ylabel("Median hourly LAeq, dB(A)")
    ax.set_title("Daily noise rhythm by station")
    ax.legend(ncol=4, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    save(fig, "daily_profiles.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    readings = cpcb.load_all_exports(CPCB_RAW_DIR)
    hourly = analytics.hourly_levels(analytics.clean_readings(readings))
    summary = analytics.station_summary(hourly)
    valid = analytics.quality_summary(readings)[["station_id", "valid_pct"]]
    forecast_mae()
    confusion()
    city_heatmap(hourly, summary["station"].tolist())
    ranking(summary, valid)
    stuck_sensor(readings)
    daily_profiles(hourly)


if __name__ == "__main__":
    main()
