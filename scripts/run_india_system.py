from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_india_analysis import main as run_india_analysis
from scripts.analyze_india_eda import main as run_india_eda
from urbannoise.india_data import load_monthly_panel
from urbannoise.india_forecast_product import forecast_next_month_persistence
from urbannoise.india_hotspots import (
    build_hotspot_behaviour,
    summarize_recurring_hotspot_patterns,
)
from urbannoise.india_iot import ingest_historical_jsonl, read_local_events


INDIA_DIR = ROOT / "data" / "external" / "india"
RAW_DIR = INDIA_DIR / "cpcb_noise_2011_2018"
OUTPUT_DIR = ROOT / "outputs" / "india"
SYSTEM_DIR = OUTPUT_DIR / "system"
MODEL_DIR = OUTPUT_DIR / "models"
COMPARISON_DIR = OUTPUT_DIR / "comparative"


def _plot_next_month_forecast(forecast: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric in zip(axes, ("Day", "Night"), strict=True):
        subset = forecast.loc[forecast["metric"].eq(metric)].sort_values("Name")
        y = np.arange(len(subset))
        predicted = subset["predicted_value"].to_numpy(dtype=float)
        lower = subset["prediction_lower_heuristic"].to_numpy(dtype=float)
        upper = subset["prediction_upper_heuristic"].to_numpy(dtype=float)
        ax.errorbar(
            predicted,
            y,
            xerr=np.vstack((predicted - lower, upper - predicted)),
            fmt="o",
            capsize=3,
            color="#1d4ed8",
        )
        ax.set_yticks(y, subset["Name"])
        ax.set_xlabel("Predicted source value (persistence)")
        ax.set_title(f"{metric}: historical next period {subset['forecast_period'].iloc[0]}")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("One-step historical extension; ranges are heuristic, not confidence intervals")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_hotspot_frequency(forecast: pd.DataFrame, output_path: Path) -> None:
    pivot = forecast.pivot(
        index="Name", columns="metric", values="hotspot_probability_estimate"
    ).sort_values("Day")
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="barh", ax=ax, color={"Day": "#d97706", "Night": "#2563eb"})
    ax.set_xlim(0, 1)
    ax.set_xlabel("Recent relative high-reading frequency (24-month empirical proxy)")
    ax.set_ylabel("Station label (not geocoded)")
    ax.set_title("Historical hotspot frequency proxy; not a calibrated future probability")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_hotspot_transitions(behaviour: pd.DataFrame, output_path: Path) -> None:
    frequency = (
        behaviour.groupby(["observation_period", "metric"])["relative_high"]
        .mean()
        .unstack("metric")
        .sort_index()
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    for metric, color in (("Day", "#d97706"), ("Night", "#2563eb")):
        if metric in frequency:
            ax.plot(frequency.index, frequency[metric], color=color, label=metric)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share of observed stations above pooled relative threshold")
    ax.set_xlabel("Historical month")
    ax.set_title("Chennai monthly relative-high frequency (not regulatory exceedances)")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _build_source_comparability(
    all_cities: pd.DataFrame,
    tnpcb_path: Path,
    srmist_path: Path,
) -> pd.DataFrame:
    tnpcb = pd.read_csv(tnpcb_path)
    srmist = pd.read_csv(srmist_path)
    other_cities = all_cities.loc[all_cities["City"].ne("Chennai")]
    rows = [
        {
            "source": "CPCB-attributed Kaggle monthly table — all city labels",
            "records": len(all_cities),
            "stations_or_locations": all_cities["Station"].nunique(),
            "period": (
                f"{all_cities['observation_period'].min()} to "
                f"{all_cities['observation_period'].max()}"
            ),
            "measurement_definition": (
                "Day/Night source fields; units, weighting and aggregation unresolved"
            ),
            "use_in_system": "Source-wide descriptive comparisons and historical replay",
            "directly_merged_with_chennai_model": False,
            "compatibility_note": (
                "Same distributed observation table; the full all-city table is not "
                "the Chennai model input."
            ),
        },
        {
            "source": "CPCB-attributed Kaggle monthly table — Chennai subset",
            "records": int(all_cities["City"].eq("Chennai").sum()),
            "stations_or_locations": int(
                all_cities.loc[all_cities["City"].eq("Chennai"), "Station"].nunique()
            ),
            "period": "2011-02 to 2018-12",
            "measurement_definition": (
                "Day/Night source fields; units, weighting and aggregation unresolved"
            ),
            "use_in_system": "Primary station-level behaviour and historical baselines",
            "directly_merged_with_chennai_model": True,
            "compatibility_note": "Primary modelling subset.",
        },
        {
            "source": "Other CPCB-attributed city labels",
            "records": len(other_cities),
            "stations_or_locations": other_cities["Station"].nunique(),
            "period": "2011-02 to 2018-12",
            "measurement_definition": (
                "Same unresolved Day/Night source fields; Mumbai/Navi Mumbai separate"
            ),
            "use_in_system": "Descriptive cross-city comparison only",
            "directly_merged_with_chennai_model": False,
            "compatibility_note": (
                "Same source schema, but not included in the Chennai target model."
            ),
        },
        {
            "source": "TNPCB Annual Report 2024 Chennai summary",
            "records": len(tnpcb),
            "stations_or_locations": tnpcb["station"].nunique(),
            "period": "2023-04 to 2024-03",
            "measurement_definition": "Annual Leq dB(A) summary fields",
            "use_in_system": "Separate provenance/reference comparison",
            "directly_merged_with_chennai_model": False,
            "compatibility_note": (
                "Different aggregation and unresolved station/site match; no merge."
            ),
        },
        {
            "source": "SRMIST January 2025 ambient reference",
            "records": len(srmist),
            "stations_or_locations": srmist["location"].nunique(),
            "period": "2025-01 (report period; exact observation times absent)",
            "measurement_definition": "Reported Day/Night dB(A) pair",
            "use_in_system": "Separate Kattankulathur local reference only",
            "directly_merged_with_chennai_model": False,
            "compatibility_note": (
                "Single report-period pair, no timestamps or coordinates; not training data."
            ),
        },
        {
            "source": "IIT Madras/TNPCB 60-site monitoring study",
            "records": 0,
            "stations_or_locations": 60,
            "period": "Reported March-December 2025; extension into 2026 described",
            "measurement_definition": "Not independently verified from accessible raw data",
            "use_in_system": "Access request only; no observations acquired",
            "directly_merged_with_chennai_model": False,
            "compatibility_note": (
                "No authorized raw export, documented API, data dictionary or reuse terms "
                "verified in this environment."
            ),
        },
    ]
    return pd.DataFrame(rows)


def _build_reference_snapshot(tnpcb_path: Path, srmist_path: Path) -> pd.DataFrame:
    tnpcb = pd.read_csv(tnpcb_path)
    tnpcb_rows = pd.DataFrame(
        {
            "source": "TNPCB Annual Report 2024",
            "location": tnpcb["station"],
            "period": tnpcb["period_start"] + " to " + tnpcb["period_end"],
            "metric": "reported_average_leq_dba",
            "day_night": tnpcb["day_night"],
            "value": tnpcb["reported_average_leq_dba"],
            "unit_as_reported": "dB(A)",
            "definition": "Annual Leq summary; separate reference, not merged",
            "source_url": tnpcb["source_url"],
        }
    )
    srmist = pd.read_csv(srmist_path)
    srmist_rows = pd.DataFrame(
        {
            "source": "SRMIST January 2025 report",
            "location": srmist["location"],
            "period": srmist["published_period"],
            "metric": "reported_noise_dba",
            "day_night": srmist["day_night"],
            "value": srmist["reported_noise_dba"],
            "unit_as_reported": "dB(A)",
            "definition": (
                "Single ambient reference pair; exact timestamp/location unavailable"
            ),
            "source_url": srmist["source_url"],
        }
    )
    return pd.concat([tnpcb_rows, srmist_rows], ignore_index=True)


def _write_dashboard(
    metrics: pd.DataFrame,
    event_count: int,
    forecast: pd.DataFrame,
    references: pd.DataFrame,
) -> None:
    score_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.metric))}</td>"
        f"<td>{html.escape(str(row.method))}</td>"
        f"<td>{int(row.observations)}</td>"
        f"<td>{float(row.mae):.3f}</td>"
        f"<td>{float(row.rmse):.3f}</td>"
        "</tr>"
        for row in metrics.sort_values(["metric", "mae"]).itertuples(index=False)
    )
    forecast_period = html.escape(str(forecast["forecast_period"].iloc[0]))
    reference_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(row.source))}</td>"
        f"<td>{html.escape(str(row.location))}</td>"
        f"<td>{html.escape(str(row.period))}</td>"
        f"<td>{html.escape(str(row.day_night))}</td>"
        f"<td>{float(row.value):g} {html.escape(str(row.unit_as_reported))}</td>"
        f"<td>{html.escape(str(row.definition))}</td>"
        "</tr>"
        for row in references.itertuples(index=False)
    )
    document = f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Urban Noise Intelligence — historical analysis</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:1180px;margin:2rem auto;padding:0 1rem;color:#172033}}
.warning{{background:#fff7ed;border-left:5px solid #c2410c;padding:1rem}}
img{{max-width:100%;height:auto;border:1px solid #ddd;margin:1rem 0}}
table{{border-collapse:collapse;width:100%;font-size:.92rem}} th,td{{border:1px solid #ddd;padding:.45rem;text-align:left}}
th{{background:#f1f5f9}} section{{margin:2rem 0}}
</style></head>
<body>
<h1>AIoT Urban Noise Hotspot Intelligence</h1>
<p class="warning"><strong>Historical software simulation only.</strong> The source ends in 2018;
Day/Night metric units and aggregation are unresolved. No live sensors, verified sensor
coordinates, regulatory exceedance claims, or current forecasts are represented.</p>
<h2>Pipeline status</h2>
<ul><li>{event_count} historical monthly events validated and stored locally.</li>
<li>Persistence forecast period: {forecast_period}; it is a one-step historical extension,
not a present-day forecast.</li><li>Station displays are not geographic maps.</li></ul>
<section><h2>Historical forecasting backtest</h2>
<table><thead><tr><th>Metric</th><th>Method</th><th>Test observations</th><th>MAE</th><th>RMSE</th></tr></thead>
<tbody>{score_rows}</tbody></table>
<p>Errors are source-value units. Forecast ranges shown below are heuristic backtest
residual ranges, not calibrated confidence intervals.</p>
<img src="../models/chennai_day_actual_vs_predicted.png" alt="Historical Day backtest">
<img src="../models/chennai_night_actual_vs_predicted.png" alt="Historical Night backtest">
</section>
<section><h2>Station-level hotspot behaviour</h2>
<p>Relative quantile thresholds are exploratory, not legal standards. Exact sensor locations
are unresolved, so no spatial heatmap or geographic prediction is provided.</p>
<img src="../chennai_relative_hotspot_frequency.png" alt="Chennai relative hotspot frequency">
<img src="chennai_hotspot_transitions.png" alt="Chennai hotspot transition timeline">
<img src="next_month_hotspot_frequency_proxy.png" alt="Empirical frequency proxy">
</section>
<section><h2>Other Indian cities</h2>
<p>Descriptive source values only; Mumbai and Navi Mumbai remain separate labels.</p>
<img src="../comparative/india_city_comparison.png" alt="Indian city comparisons">
<img src="../comparative/india_relative_high_reading_frequency.png" alt="City relative high readings">
</section>
<section><h2>Separate-source reference observations</h2>
<p>TNPCB annual summaries and SRMIST report values are shown with their own source,
period, and definitions. They are not pooled or directly compared with the unresolved
CPCB monthly source fields.</p>
<table><thead><tr><th>Source</th><th>Location</th><th>Report period</th><th>Day/Night</th>
<th>Reported value</th><th>Definition/limitation</th></tr></thead>
<tbody>{reference_rows}</tbody></table>
</section>
<section><h2>Historical next-period station estimates</h2>
<p>Latest source observation is 2018-12; estimates refer to {forecast_period} only.</p>
<img src="next_month_station_forecasts.png" alt="Historical next month persistence estimates">
<p><a href="next_month_station_forecasts.csv">Download station forecast estimates</a> ·
<a href="source_comparability.csv">Source compatibility ledger</a> ·
<a href="pipeline_status.json">Pipeline status</a></p>
</section>
</body></html>"""
    (SYSTEM_DIR / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    for directory in (SYSTEM_DIR, COMPARISON_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    run_india_eda()
    run_india_analysis()
    all_cities = load_monthly_panel(RAW_DIR / "station_month.csv", RAW_DIR / "stations.csv")
    chennai = all_cities.loc[all_cities["City"].eq("Chennai")].copy()
    predictions_path = MODEL_DIR / "chennai_walk_forward_predictions.csv"
    backtest_predictions = pd.read_csv(predictions_path)
    forecast = forecast_next_month_persistence(chennai, backtest_predictions)
    forecast.to_csv(SYSTEM_DIR / "next_month_station_forecasts.csv", index=False)
    _plot_next_month_forecast(forecast, SYSTEM_DIR / "next_month_station_forecasts.png")
    _plot_hotspot_frequency(forecast, SYSTEM_DIR / "next_month_hotspot_frequency_proxy.png")

    chennai_behaviour = build_hotspot_behaviour(chennai)
    chennai_behaviour.to_csv(SYSTEM_DIR / "chennai_hotspot_behaviour_timeline.csv", index=False)
    recurring = summarize_recurring_hotspot_patterns(chennai_behaviour)
    recurring.to_csv(SYSTEM_DIR / "chennai_recurring_hotspot_patterns.csv", index=False)
    _plot_hotspot_transitions(
        chennai_behaviour, SYSTEM_DIR / "chennai_hotspot_transitions.png"
    )
    all_city_behaviour = build_hotspot_behaviour(all_cities)
    all_city_behaviour.to_csv(
        COMPARISON_DIR / "india_station_month_hotspot_behaviour.csv", index=False
    )
    summarize_recurring_hotspot_patterns(all_city_behaviour).to_csv(
        COMPARISON_DIR / "india_recurring_hotspot_patterns.csv", index=False
    )

    event_path = OUTPUT_DIR / "historical_monthly_noise_events.jsonl"
    database_path = SYSTEM_DIR / "historical_noise_events.sqlite3"
    ingested, stored = ingest_historical_jsonl(event_path, database_path)
    local_events = read_local_events(database_path)
    if ingested != len(all_cities) or stored != len(all_cities):
        raise RuntimeError(
            "Local replay store row counts do not match the validated source panel."
        )
    if any(event["is_live_measurement"] for event in local_events):
        raise RuntimeError("Historical replay store contains a live-labelled event.")

    tnpcb_path = INDIA_DIR / "tnpcb_2023_24_chennai_annual_noise.csv"
    srmist_path = INDIA_DIR / "srmist_january_2025_ambient_noise.csv"
    source_ledger = _build_source_comparability(
        all_cities,
        tnpcb_path,
        srmist_path,
    )
    source_ledger.to_csv(SYSTEM_DIR / "source_comparability.csv", index=False)
    reference_snapshot = _build_reference_snapshot(tnpcb_path, srmist_path)
    reference_snapshot.to_csv(
        SYSTEM_DIR / "separate_reference_source_values.csv", index=False
    )
    metrics = pd.read_csv(MODEL_DIR / "chennai_walk_forward_metrics.csv")
    status = {
        "status": "completed",
        "scope": "Chennai historical model; India source comparisons; SRMIST separate reference",
        "virtual_source": "CPCB-attributed historical monthly observations",
        "replay_events_read": ingested,
        "sqlite_rows": stored,
        "local_store": str(database_path.relative_to(ROOT)),
        "forecast_period": str(forecast["forecast_period"].iloc[0]),
        "forecast_is_current": False,
        "streaming_engine": "local Python batch replay and SQLite",
        "kafka_installed": False,
        "spark_structured_streaming_installed": False,
        "delta_lake_installed": False,
        "geographic_predictions": False,
        "coordinates_verified": False,
        "source_metric_definition_verified": False,
        "historical_backtest_metrics": metrics.to_dict(orient="records"),
    }
    (SYSTEM_DIR / "pipeline_status.json").write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    _write_dashboard(metrics, stored, forecast, reference_snapshot)
    print(
        f"India intelligence system completed: {stored} local historical events, "
        f"{len(forecast)} historical next-period station estimates, dashboard at "
        f"{(SYSTEM_DIR / 'index.html').relative_to(ROOT)}."
    )


if __name__ == "__main__":
    main()
