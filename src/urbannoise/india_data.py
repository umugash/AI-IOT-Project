from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MONTHLY_REQUIRED = {
    "Station",
    "Year",
    "Month",
    "Day",
    "Night",
    "DayLimit",
    "NightLimit",
}
STATION_REQUIRED = {"Station", "Name", "City", "State", "Type"}
DATA_SOURCE = "CPCB-attributed Kaggle monthly noise dataset (2011-2018)"


class IndiaDatasetValidationError(ValueError):
    """Raised when an India noise dataset violates the expected source schema."""


def load_monthly_panel(
    monthly_path: str | Path, stations_path: str | Path, city: str | None = None
) -> pd.DataFrame:
    monthly_file = Path(monthly_path)
    stations_file = Path(stations_path)
    if not monthly_file.is_file():
        raise FileNotFoundError(f"Monthly source file not found: {monthly_file}")
    if not stations_file.is_file():
        raise FileNotFoundError(f"Station metadata file not found: {stations_file}")

    observations = pd.read_csv(monthly_file)
    stations = pd.read_csv(stations_file)
    if not MONTHLY_REQUIRED.issubset(observations.columns):
        raise IndiaDatasetValidationError(
            "Monthly file is missing required columns: "
            + ", ".join(sorted(MONTHLY_REQUIRED.difference(observations.columns)))
        )
    if not STATION_REQUIRED.issubset(stations.columns):
        raise IndiaDatasetValidationError(
            "Station file is missing required columns: "
            + ", ".join(sorted(STATION_REQUIRED.difference(stations.columns)))
        )
    if observations.empty or stations.empty:
        raise IndiaDatasetValidationError("India noise source files must not be empty.")
    if stations["Station"].duplicated().any():
        raise IndiaDatasetValidationError("Station metadata contains duplicate IDs.")

    observations["Station"] = observations["Station"].astype("string").str.strip()
    stations["Station"] = stations["Station"].astype("string").str.strip()
    if observations["Station"].isna().any() or observations["Station"].eq("").any():
        raise IndiaDatasetValidationError("Monthly data contain an empty station ID.")

    for column in ("Year", "Month", "Day", "Night", "DayLimit", "NightLimit"):
        observations[column] = pd.to_numeric(observations[column], errors="coerce")
    if observations[["Year", "Month"]].isna().any(axis=None):
        raise IndiaDatasetValidationError("Monthly data contain invalid year/month fields.")
    if not observations["Year"].mod(1).eq(0).all():
        raise IndiaDatasetValidationError("Year must contain whole years.")
    if not observations["Month"].between(1, 12).all():
        raise IndiaDatasetValidationError("Month must be between 1 and 12.")
    observations["Year"] = observations["Year"].astype(int)
    observations["Month"] = observations["Month"].astype(int)

    panel = observations.merge(
        stations, on="Station", how="left", validate="many_to_one", indicator=True
    )
    if panel["_merge"].ne("both").any():
        unknown = sorted(panel.loc[panel["_merge"].ne("both"), "Station"].unique())
        raise IndiaDatasetValidationError(
            "Monthly rows have no station metadata: " + ", ".join(unknown)
        )
    panel = panel.drop(columns="_merge")
    if panel.duplicated(["Station", "Year", "Month"]).any():
        raise IndiaDatasetValidationError("Duplicate station-month observations found.")

    if city is not None:
        panel = panel.loc[panel["City"].eq(city)].copy()
        if panel.empty:
            raise IndiaDatasetValidationError(
                f"No monthly observations found for city {city!r}."
            )

    panel["month_index"] = panel["Year"] * 12 + panel["Month"] - 1
    panel["observation_period"] = (
        panel["Year"].astype(str)
        + "-"
        + panel["Month"].astype(str).str.zfill(2)
    )
    panel["data_source"] = DATA_SOURCE
    return panel.sort_values(["month_index", "Station"]).reset_index(drop=True)


def build_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Create one-step-ahead features using only strictly earlier station rows."""
    data = panel.sort_values(["Station", "month_index"]).copy()
    month_indices = data.groupby("Station", sort=False)["month_index"]
    for metric in ("Day", "Night"):
        grouped = data.groupby("Station", sort=False)[metric]
        for lag in (1, 2, 3, 12):
            lagged = grouped.shift(lag)
            lagged_month = month_indices.shift(lag)
            data[f"{metric}_lag_{lag}"] = lagged.where(
                data["month_index"] - lagged_month == lag
            )
        shifted = grouped.shift(1)
        for window in (3, 6):
            rolling_mean = shifted.groupby(
                data["Station"], sort=False
            ).transform(lambda values: values.rolling(window, min_periods=window).mean())
            oldest_month = month_indices.shift(window)
            data[f"{metric}_rolling_mean_{window}"] = rolling_mean.where(
                data["month_index"] - oldest_month == window
            )

    angle = 2 * np.pi * (data["Month"] - 1) / 12
    data["month_sin"] = np.sin(angle)
    data["month_cos"] = np.cos(angle)
    return data.sort_values(["month_index", "Station"]).reset_index(drop=True)
