from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbannoise.india_data import build_lag_features, load_monthly_panel
from urbannoise.india_forecasting import (
    compute_feature_importances,
    run_forecasting_backtest,
)
from urbannoise.india_hotspots import summarize_relative_hotspots
from urbannoise.india_replay import write_historical_jsonl


INDIA_DIR = ROOT / "data" / "external" / "india"
RAW_DIR = INDIA_DIR / "cpcb_noise_2011_2018"
OUTPUT_DIR = ROOT / "outputs" / "india"
EDA_DIR = OUTPUT_DIR / "eda"
MODEL_DIR = OUTPUT_DIR / "models"
COMPARISON_DIR = OUTPUT_DIR / "comparative"


def _plot_hotspots(stations: pd.DataFrame, output_file: Path) -> None:
    chennai = stations.loc[stations["city"].eq("Chennai")].copy()
    pivot = chennai.pivot(
        index="station_name", columns="metric", values="relative_high_frequency"
    )
    pivot = pivot.sort_values("Day", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot.plot(kind="barh", ax=ax, color={"Day": "#d97706", "Night": "#2563eb"})
    ax.set_xlabel("Share of available station-months above pooled top-quartile threshold")
    ax.set_ylabel("Station (not geocoded)")
    ax.set_title("Chennai relative high-reading frequency — exploratory, not regulatory")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_station_means(panel: pd.DataFrame, output_file: Path) -> None:
    means = panel.loc[panel["City"].eq("Chennai")].groupby("Name")[["Day", "Night"]].mean()
    means = means.sort_values("Day", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    means.plot(kind="barh", ax=ax, color={"Day": "#d97706", "Night": "#2563eb"})
    ax.set_xlabel("Mean source value (measurement definition unresolved)")
    ax.set_ylabel("Monitoring-station label")
    ax.set_title("Chennai station comparison — no exact sensor coordinates")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_india_comparison(panel: pd.DataFrame, output_file: Path) -> None:
    city_means = panel.groupby("City")[["Day", "Night"]].mean()
    city_means = city_means.sort_values("Day")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    city_means.plot(kind="barh", ax=axes[0], color={"Day": "#d97706", "Night": "#2563eb"})
    axes[0].set_xlabel("Mean source value (measurement definition unresolved)")
    axes[0].set_ylabel("")
    axes[0].set_title("City-label averages (Navi Mumbai separate)")
    axes[0].grid(axis="x", alpha=0.25)
    distributions = [
        panel.loc[panel["City"].eq(city), "Day"].dropna().to_numpy()
        for city in city_means.index
    ]
    axes[1].boxplot(distributions, vert=False, showmeans=True)
    axes[1].set_yticks(range(1, len(city_means.index) + 1), city_means.index)
    axes[1].set_xlabel("Day source value (definition unresolved)")
    axes[1].set_title("Day-reading distributions across source city labels")
    axes[1].grid(axis="x", alpha=0.25)
    fig.suptitle("Comparative India analysis; descriptive source values only")
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_india_relative_high_frequency(
    station_summary: pd.DataFrame, output_file: Path
) -> None:
    frequency = (
        station_summary.groupby(["city", "metric"])["relative_high_frequency"]
        .mean()
        .unstack("metric")
    )
    frequency = frequency.sort_values("Day", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    frequency.plot(kind="barh", ax=ax, color={"Day": "#d97706", "Night": "#2563eb"})
    ax.set_xlabel("Mean station-month share above pooled source-value top quartile")
    ax.set_ylabel("Source city label")
    ax.set_title("Relative high-reading frequency by city (exploratory, not regulatory)")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_forecasts(
    forecasts: pd.DataFrame,
    metric: str,
    methods: tuple[str, ...],
    output_file: Path,
) -> None:
    subset = forecasts.loc[
        forecasts["metric"].eq(metric) & forecasts["method"].isin(methods)
    ]
    actual = subset.drop_duplicates(["Station", "observation_period"])[
        ["observation_period", "actual"]
    ].groupby("observation_period")["actual"].mean()
    predicted = (
        subset.groupby(["method", "observation_period"])["predicted"]
        .mean()
        .unstack("method")
    )
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(actual.index, actual.values, color="#111827", linewidth=2, label="Actual station mean")
    for method in methods:
        if method in predicted.columns:
            ax.plot(predicted.index, predicted[method], linewidth=1.2, label=method)
    ax.set_title(f"{metric} one-step walk-forward backtest (2017–2018, not current)")
    ax.set_ylabel("Source value (measurement definition unresolved)")
    ax.set_xlabel("Historical monthly period")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_file, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for directory in (EDA_DIR, MODEL_DIR, COMPARISON_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    all_cities = load_monthly_panel(
        RAW_DIR / "station_month.csv", RAW_DIR / "stations.csv"
    )
    chennai = all_cities.loc[all_cities["City"].eq("Chennai")].copy()
    featured = build_lag_features(chennai)
    featured.to_csv(
        MODEL_DIR / "chennai_station_month_modeling_panel.csv", index=False
    )
    station_hotspots, monthly_hotspots = summarize_relative_hotspots(chennai)
    station_hotspots.to_csv(
        OUTPUT_DIR / "chennai_station_hotspot_behaviour.csv", index=False
    )
    monthly_hotspots.to_csv(
        OUTPUT_DIR / "chennai_monthly_hotspot_behaviour.csv", index=False
    )
    _plot_hotspots(
        station_hotspots,
        OUTPUT_DIR / "chennai_relative_hotspot_frequency.png",
    )
    _plot_station_means(chennai, OUTPUT_DIR / "chennai_station_comparison.png")

    forecasts, scores, per_station, splits = run_forecasting_backtest(featured)
    forecasts.to_csv(MODEL_DIR / "chennai_walk_forward_predictions.csv", index=False)
    scores.to_csv(MODEL_DIR / "chennai_walk_forward_metrics.csv", index=False)
    per_station.to_csv(
        MODEL_DIR / "chennai_walk_forward_metrics_by_station.csv", index=False
    )
    splits.to_csv(MODEL_DIR / "chennai_chronological_splits.csv", index=False)
    feature_importances = compute_feature_importances(featured)
    feature_importances.to_csv(
        MODEL_DIR / "chennai_feature_importances.csv", index=False
    )
    _plot_forecasts(
        forecasts,
        "Day",
        ("persistence", "seasonal_naive_12", "moving_average_3", "ridge_pooled"),
        MODEL_DIR / "chennai_day_actual_vs_predicted.png",
    )
    _plot_forecasts(
        forecasts,
        "Night",
        ("persistence", "seasonal_naive_12", "moving_average_3", "ridge_pooled"),
        MODEL_DIR / "chennai_night_actual_vs_predicted.png",
    )

    city_summary = (
        all_cities.groupby(["City", "State"], sort=True)
        .agg(
            station_count=("Station", "nunique"),
            monthly_records=("Station", "size"),
            first_year=("Year", "min"),
            last_year=("Year", "max"),
            day_mean=("Day", "mean"),
            day_median=("Day", "median"),
            day_minimum=("Day", "min"),
            day_maximum=("Day", "max"),
            night_mean=("Night", "mean"),
            night_median=("Night", "median"),
            night_minimum=("Night", "min"),
            night_maximum=("Night", "max"),
            mean_day_minus_night=("Day", lambda values: float((values - all_cities.loc[values.index, "Night"]).mean())),
        )
        .reset_index()
    )
    city_summary.to_csv(
        COMPARISON_DIR / "india_city_comparison.csv", index=False
    )
    india_station_hotspots, india_monthly_hotspots = summarize_relative_hotspots(
        all_cities
    )
    india_station_hotspots.to_csv(
        COMPARISON_DIR / "india_relative_high_readings_by_station.csv", index=False
    )
    india_monthly_hotspots.to_csv(
        COMPARISON_DIR / "india_relative_high_readings_by_city_month.csv", index=False
    )
    _plot_india_relative_high_frequency(
        india_station_hotspots,
        COMPARISON_DIR / "india_relative_high_reading_frequency.png",
    )
    station_summary = (
        all_cities.groupby(["City", "Station", "Name", "Type"], sort=True)
        .agg(
            monthly_records=("Day", "count"),
            day_mean=("Day", "mean"),
            night_mean=("Night", "mean"),
            mean_day_minus_night=("Day", lambda values: float((values - all_cities.loc[values.index, "Night"]).mean())),
            day_minimum=("Day", "min"),
            day_maximum=("Day", "max"),
            night_minimum=("Night", "min"),
            night_maximum=("Night", "max"),
        )
        .reset_index()
    )
    station_summary.to_csv(
        COMPARISON_DIR / "india_station_comparison.csv", index=False
    )
    yearly = (
        all_cities.groupby(["City", "Year"], sort=True)
        .agg(
            monthly_records=("Station", "size"),
            station_count=("Station", "nunique"),
            day_mean=("Day", "mean"),
            night_mean=("Night", "mean"),
        )
        .reset_index()
    )
    yearly.to_csv(COMPARISON_DIR / "india_city_yearly_trends.csv", index=False)
    _plot_india_comparison(
        all_cities, COMPARISON_DIR / "india_city_comparison.png"
    )

    replay_rows = write_historical_jsonl(
        all_cities, OUTPUT_DIR / "historical_monthly_noise_events.jsonl"
    )
    manifest = {
        "scope": "Chennai primary modelling; all-city comparison; SRMIST separate reference",
        "model_dataset_rows": len(chennai),
        "model_dataset_stations": int(chennai["Station"].nunique()),
        "comparative_city_labels": sorted(all_cities["City"].unique().tolist()),
        "comparative_station_count": int(all_cities["Station"].nunique()),
        "historical_replay_event_count": replay_rows,
        "observation_period_is_year_month_only": True,
        "synthetic_timestamps_created": False,
        "models_trained": ["Ridge", "RandomForestRegressor"],
        "backtest": "expanding-origin one-month-ahead walk-forward; test periods 2017-01 through 2018-12",
        "source_metric_warning": "Day/Night unit, weighting and aggregation remain unverified.",
        "live_measurements": False,
    }
    (OUTPUT_DIR / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(
        f"Created Chennai model/backtest outputs for {len(chennai)} station-month rows; "
        f"all-city comparison covers {all_cities['Station'].nunique()} stations and "
        f"{replay_rows} explicitly historical monthly replay events."
    )
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
