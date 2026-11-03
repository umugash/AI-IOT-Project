from __future__ import annotations

import math

import pandas as pd

from .dataset import DatasetValidationError


METERS_PER_LATITUDE_DEGREE = 111_320.0


def assign_metric_grid(frame: pd.DataFrame, cell_size_m: int) -> pd.DataFrame:
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive.")
    if frame.empty:
        raise DatasetValidationError("Cannot grid an empty observation frame.")
    required = {"latitude", "longitude"}
    missing = required.difference(frame.columns)
    if missing:
        raise DatasetValidationError(
            "Metric gridding requires columns: " + ", ".join(sorted(missing))
        )
    if frame[["latitude", "longitude"]].isna().any(axis=None):
        raise DatasetValidationError("Metric gridding requires valid GPS coordinates.")

    reference_latitude = float(frame["latitude"].mean())
    meters_per_longitude_degree = (
        METERS_PER_LATITUDE_DEGREE
        * math.cos(math.radians(reference_latitude))
    )
    if meters_per_longitude_degree <= 0:
        raise DatasetValidationError("Cannot project longitude at this latitude.")

    return frame.assign(
        _grid_x=(
            frame["longitude"] * meters_per_longitude_degree / cell_size_m
        ).apply(math.floor),
        _grid_y=(
            frame["latitude"] * METERS_PER_LATITUDE_DEGREE / cell_size_m
        ).apply(math.floor),
    )


def aggregate_spatial_cells(
    frame: pd.DataFrame,
    *,
    cell_size_m: int = 250,
    minimum_observations: int = 5,
    hotspot_quantile: float = 0.9,
    maximum_gps_accuracy_m: float = 100.0,
) -> pd.DataFrame:
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive.")
    if minimum_observations < 1:
        raise ValueError("minimum_observations must be at least 1.")
    if not 0 < hotspot_quantile < 1:
        raise ValueError("hotspot_quantile must be between 0 and 1.")
    if maximum_gps_accuracy_m <= 0:
        raise ValueError("maximum_gps_accuracy_m must be positive.")

    required = {"latitude", "longitude", "noise_measurement"}
    missing = required.difference(frame.columns)
    if missing:
        raise DatasetValidationError(
            "Spatial aggregation requires columns: " + ", ".join(sorted(missing))
        )
    if frame.empty:
        raise DatasetValidationError("Cannot aggregate an empty observation frame.")
    if frame[["latitude", "longitude", "noise_measurement"]].isna().any(axis=None):
        raise DatasetValidationError(
            "Spatial aggregation received missing coordinates or measurements."
        )

    valid = frame.copy()
    if "accuracy" in valid.columns:
        valid = valid.loc[
            valid["accuracy"].notna()
            & valid["accuracy"].between(0, maximum_gps_accuracy_m)
        ].copy()
        if valid.empty:
            raise DatasetValidationError(
                "No observations remain after GPS accuracy filtering."
            )
    valid = assign_metric_grid(valid, cell_size_m)
    groups = valid.groupby(["_grid_x", "_grid_y"], sort=True)
    cells = groups.agg(
        observation_count=("noise_measurement", "size"),
        mean_reported_noise=("noise_measurement", "mean"),
        median_reported_noise=("noise_measurement", "median"),
        p90_reported_noise=("noise_measurement", lambda values: values.quantile(0.9)),
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        mean_gps_accuracy_m=("accuracy", "mean")
        if "accuracy" in valid.columns
        else ("noise_measurement", lambda values: float("nan")),
    ).reset_index()
    cells = cells.loc[
        cells["observation_count"] >= minimum_observations
    ].copy()
    if cells.empty:
        cells["is_hotspot"] = pd.Series(dtype=bool)
        return cells

    threshold = float(cells["mean_reported_noise"].quantile(hotspot_quantile))
    cells["hotspot_threshold"] = threshold
    cells["is_hotspot"] = cells["mean_reported_noise"] >= threshold
    cells["cell_size_m"] = cell_size_m
    cells["maximum_gps_accuracy_m"] = (
        maximum_gps_accuracy_m if "accuracy" in frame.columns else float("nan")
    )
    return cells.sort_values(
        ["is_hotspot", "mean_reported_noise", "observation_count"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def summarize_categories_by_cell(
    frame: pd.DataFrame,
    *,
    cell_size_m: int = 250,
) -> pd.DataFrame:
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive.")
    required = {"latitude", "longitude", "class"}
    missing = required.difference(frame.columns)
    if missing:
        raise DatasetValidationError(
            "Category mapping requires columns: " + ", ".join(sorted(missing))
        )
    mapped = assign_metric_grid(frame, cell_size_m)
    counts = (
        mapped.groupby(["_grid_x", "_grid_y", "class"], as_index=False)
        .size()
        .rename(columns={"size": "observation_count"})
    )
    return counts
