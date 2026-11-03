from __future__ import annotations

import pandas as pd

from .dataset import DatasetValidationError
from .spatial import assign_metric_grid


class TemporalDataUnavailableError(DatasetValidationError):
    """Raised when a time-dependent operation lacks genuine timestamps."""


def require_timestamps(frame: pd.DataFrame) -> pd.DataFrame:
    if "timestamp" not in frame.columns:
        raise TemporalDataUnavailableError(
            "Temporal analysis is unavailable: the supplied public Uganda "
            "metadata has no timestamp column. Do not infer or synthesize times."
        )
    parsed = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if parsed.isna().any():
        invalid = int(parsed.isna().sum())
        raise TemporalDataUnavailableError(
            f"Temporal analysis requires valid timestamps; found {invalid} invalid rows."
        )
    result = frame.copy()
    result["timestamp"] = parsed
    return result


def derive_hotspot_transitions(
    frame: pd.DataFrame,
    *,
    window: str = "1D",
    cell_size_m: int = 250,
    minimum_observations: int = 5,
    hotspot_quantile: float = 0.9,
    intensification_delta: float = 3.0,
) -> pd.DataFrame:
    timestamped = require_timestamps(frame)
    if window not in {"15min", "30min", "1h", "1D", "1W"}:
        raise ValueError("window must be one of: 15min, 30min, 1h, 1D, 1W.")
    if cell_size_m <= 0 or minimum_observations < 1:
        raise ValueError("cell_size_m and minimum_observations must be positive.")
    if not 0 < hotspot_quantile < 1:
        raise ValueError("hotspot_quantile must be between 0 and 1.")
    if intensification_delta < 0:
        raise ValueError("intensification_delta cannot be negative.")
    required = {"latitude", "longitude", "noise_measurement"}
    missing = required.difference(timestamped.columns)
    if missing:
        raise DatasetValidationError(
            "Temporal analysis requires columns: " + ", ".join(sorted(missing))
        )

    timestamped = assign_metric_grid(timestamped, cell_size_m)
    period = "W-SUN" if window == "1W" else window
    utc_naive = timestamped["timestamp"].dt.tz_localize(None)
    if window == "1W":
        window_start = utc_naive.dt.to_period(period).dt.start_time
    else:
        window_start = utc_naive.dt.floor(period)
    timestamped["_window"] = window_start.dt.tz_localize("UTC")
    aggregated = (
        timestamped.groupby(["_window", "_grid_x", "_grid_y"], as_index=False)
        .agg(
            observation_count=("noise_measurement", "size"),
            mean_reported_noise=("noise_measurement", "mean"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
    )
    eligible = aggregated["observation_count"] >= minimum_observations
    if not eligible.any():
        raise DatasetValidationError(
            "No time/location windows meet the minimum observation requirement."
        )
    threshold = float(
        aggregated.loc[eligible, "mean_reported_noise"].quantile(hotspot_quantile)
    )
    aggregated["is_hotspot"] = pd.Series(
        pd.NA,
        index=aggregated.index,
        dtype="boolean",
    )
    aggregated.loc[eligible, "is_hotspot"] = (
        aggregated.loc[eligible, "mean_reported_noise"] >= threshold
    )
    aggregated = aggregated.sort_values(
        ["_grid_x", "_grid_y", "_window"]
    ).reset_index(drop=True)
    eligible = aggregated["observation_count"] >= minimum_observations
    grouped = aggregated.groupby(["_grid_x", "_grid_y"], sort=False)
    aggregated["_previous_window"] = grouped["_window"].shift(1)
    aggregated["previous_count"] = grouped["observation_count"].shift(1)
    aggregated["previous_hotspot"] = grouped["is_hotspot"].shift(1)
    aggregated["previous_noise"] = grouped["mean_reported_noise"].shift(1)
    window_duration = {
        "15min": pd.Timedelta(minutes=15),
        "30min": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "1D": pd.Timedelta(days=1),
        "1W": pd.Timedelta(days=7),
    }[window]
    consecutive = (
        aggregated["_window"] - aggregated["_previous_window"] == window_duration
    )
    previous_eligible = aggregated["previous_count"] >= minimum_observations
    comparable = eligible & previous_eligible & consecutive
    aggregated["behaviour"] = "insufficient_data"
    aggregated.loc[eligible & ~comparable, "behaviour"] = (
        "insufficient_previous_window"
    )
    aggregated.loc[
        comparable
        & ~aggregated["previous_hotspot"].fillna(False)
        & aggregated["is_hotspot"].fillna(False),
        "behaviour",
    ] = "emergence"
    aggregated.loc[
        comparable
        & aggregated["previous_hotspot"].fillna(False)
        & aggregated["is_hotspot"].fillna(False),
        "behaviour",
    ] = "persistence"
    intensified = (
        comparable
        & aggregated["previous_hotspot"].fillna(False)
        & aggregated["is_hotspot"].fillna(False)
        & (
            aggregated["mean_reported_noise"] - aggregated["previous_noise"]
            >= intensification_delta
        )
    )
    aggregated.loc[intensified, "behaviour"] = "intensification"
    aggregated.loc[
        comparable
        & aggregated["previous_hotspot"].fillna(False)
        & ~aggregated["is_hotspot"].fillna(False),
        "behaviour",
    ] = "disappearance"

    for _, current in aggregated.groupby("_window"):
        disappeared = current.loc[
            current["behaviour"] == "disappearance", ["_grid_x", "_grid_y"]
        ].to_numpy()
        emerging = current.index[current["behaviour"] == "emergence"]
        if not len(disappeared) or not len(emerging):
            continue
        for index in emerging:
            x, y = aggregated.loc[index, ["_grid_x", "_grid_y"]]
            if any(
                max(abs(x - old_x), abs(y - old_y)) == 1
                for old_x, old_y in disappeared
            ):
                aggregated.loc[index, "behaviour"] = "shift"
    return aggregated.reset_index(drop=True)
