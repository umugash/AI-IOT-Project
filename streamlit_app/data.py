from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from urbannoise.india_data import load_monthly_panel


ROOT = Path(__file__).resolve().parents[1]
INDIA_DIR = ROOT / "data" / "external" / "india"
RAW_DIR = INDIA_DIR / "cpcb_noise_2011_2018"
OUTPUT_DIR = ROOT / "outputs" / "india"


class ApplicationDataError(RuntimeError):
    """Raised when an authoritative application dataset is missing or invalid."""


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise ApplicationDataError(f"Required data file is missing: {path}")
    return path


def read_csv(path: Path, required_columns: set[str] | None = None) -> pd.DataFrame:
    require_file(path)
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ApplicationDataError(f"Could not read CSV {path}: {error}") from error
    if frame.empty:
        raise ApplicationDataError(f"CSV has no records: {path}")
    if required_columns:
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ApplicationDataError(
                f"CSV {path} is missing required columns: {', '.join(sorted(missing))}"
            )
    return frame


def read_json(path: Path) -> dict[str, Any]:
    require_file(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ApplicationDataError(f"Could not read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise ApplicationDataError(f"Expected a JSON object in {path}")
    return value


def load_panel(city: str | None = None) -> pd.DataFrame:
    try:
        return load_monthly_panel(
            require_file(RAW_DIR / "station_month.csv"),
            require_file(RAW_DIR / "stations.csv"),
            city=city,
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        raise ApplicationDataError(f"Could not validate monthly noise data: {error}") from error


def filter_observations(
    panel: pd.DataFrame,
    stations: list[str] | None = None,
    years: list[int] | None = None,
    months: list[int] | None = None,
    categories: list[str] | None = None,
) -> pd.DataFrame:
    result = panel.copy()
    if stations:
        result = result.loc[result["Station"].isin(stations)]
    if years:
        result = result.loc[result["Year"].isin(years)]
    if months:
        result = result.loc[result["Month"].isin(months)]
    if categories:
        result = result.loc[result["Type"].isin(categories)]
    return result.reset_index(drop=True)


def metric_long_frame(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"Station", "Name", "City", "Year", "Month", "Day", "Night"}
    missing = required.difference(panel.columns)
    if missing:
        raise ApplicationDataError(
            "Cannot prepare chart data; missing fields: " + ", ".join(sorted(missing))
        )
    return panel.melt(
        id_vars=[
            column
            for column in (
                "Station",
                "Name",
                "City",
                "State",
                "Type",
                "Year",
                "Month",
                "observation_period",
                "month_index",
            )
            if column in panel.columns
        ],
        value_vars=["Day", "Night"],
        var_name="metric",
        value_name="source_value",
    ).dropna(subset=["source_value"])


def image_data_uri(path: Path) -> str:
    require_file(path)
    suffix = path.suffix.casefold()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"
    import base64

    return f"data:{media_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def csv_output(name: str) -> Path:
    allowed = {
        "metrics": OUTPUT_DIR / "models" / "chennai_walk_forward_metrics.csv",
        "predictions": OUTPUT_DIR / "models" / "chennai_walk_forward_predictions.csv",
        "station_scores": OUTPUT_DIR
        / "models"
        / "chennai_walk_forward_metrics_by_station.csv",
        "feature_importances": OUTPUT_DIR / "models" / "chennai_feature_importances.csv",
        "station_forecasts": OUTPUT_DIR / "system" / "next_month_station_forecasts.csv",
        "station_hotspots": OUTPUT_DIR / "chennai_station_hotspot_behaviour.csv",
        "monthly_hotspots": OUTPUT_DIR / "chennai_monthly_hotspot_behaviour.csv",
        "hotspot_timeline": OUTPUT_DIR / "system" / "chennai_hotspot_behaviour_timeline.csv",
        "recurring_patterns": OUTPUT_DIR
        / "system"
        / "chennai_recurring_hotspot_patterns.csv",
        "city_comparison": OUTPUT_DIR / "comparative" / "india_city_comparison.csv",
        "station_comparison": OUTPUT_DIR / "comparative" / "india_station_comparison.csv",
        "yearly_comparison": OUTPUT_DIR / "comparative" / "india_city_yearly_trends.csv",
        "city_hotspots": OUTPUT_DIR
        / "comparative"
        / "india_relative_high_readings_by_station.csv",
        "source_ledger": OUTPUT_DIR / "system" / "source_comparability.csv",
        "references": OUTPUT_DIR / "system" / "separate_reference_source_values.csv",
        "coverage": OUTPUT_DIR / "eda" / "chennai_station_coverage.csv",
        "descriptive": OUTPUT_DIR / "eda" / "chennai_station_descriptive_summary.csv",
    }
    if name not in allowed:
        raise ApplicationDataError(f"Unknown application output name: {name}")
    return require_file(allowed[name])


def app_document(name: str) -> Path:
    allowed = {
        "request": ROOT / "DATA_ACCESS_REQUEST_DRAFT.md",
        "architecture": ROOT / "ARCHITECTURE.md",
        "scope": ROOT / "FINAL_PROJECT_SCOPE.md",
        "forecasting": ROOT / "FORECASTING_BASELINE_REPORT.md",
        "audit": ROOT / "INDIA_DATA_AUDIT.md",
        "dictionary": ROOT / "DATA_DICTIONARY.md",
        "implementation": ROOT / "INDIA_IMPLEMENTATION_PLAN.md",
    }
    if name not in allowed:
        raise ApplicationDataError(f"Unknown application document name: {name}")
    return require_file(allowed[name])
