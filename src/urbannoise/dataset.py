from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd


REQUIRED_COLUMNS = {
    "filename",
    "class",
    "class_id",
    "noise_measurement",
    "latitude",
    "longitude",
    "altitude",
    "accuracy",
    "submitter_id",
    "region",
}
NUMERIC_COLUMNS = (
    "noise_measurement",
    "latitude",
    "longitude",
    "altitude",
    "accuracy",
    "class_id",
    "submitter_id",
)


class DatasetValidationError(ValueError):
    """Raised when source metadata is missing required or usable fields."""


@dataclass(frozen=True)
class DatasetAudit:
    rows: int
    columns: list[str]
    missing_by_column: dict[str, int]
    duplicate_filenames: int
    regions: dict[str, int]
    classes: dict[str, int]
    classes_by_region: dict[str, dict[str, int]]
    coordinate_pairs: int
    repeated_coordinate_pairs: int
    observations_at_repeated_coordinates: int
    timestamp_column_present: bool
    noise_measurement_min: float
    noise_measurement_median: float
    noise_measurement_max: float
    gps_accuracy_median: float
    gps_accuracy_p95: float
    gps_accuracy_max: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_metadata(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Dataset metadata file not found: {source}")

    frame = pd.read_csv(source)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise DatasetValidationError(
            "Dataset metadata is missing required columns: "
            + ", ".join(missing_columns)
        )
    if frame.empty:
        raise DatasetValidationError("Dataset metadata contains no observations.")

    frame["filename"] = frame["filename"].astype("string").str.strip()
    frame["class"] = frame["class"].astype("string").str.strip()
    frame["region"] = frame["region"].astype("string").str.strip()
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if frame["filename"].isna().any() or frame["filename"].eq("").any():
        raise DatasetValidationError("Dataset contains empty recording filenames.")
    if frame["class"].isna().any() or frame["region"].isna().any():
        raise DatasetValidationError("Dataset contains missing class or region labels.")
    if frame[["latitude", "longitude"]].isna().any(axis=None):
        raise DatasetValidationError("Dataset contains invalid GPS coordinates.")
    if not frame["latitude"].between(-90, 90).all():
        raise DatasetValidationError("Latitude values must be between -90 and 90.")
    if not frame["longitude"].between(-180, 180).all():
        raise DatasetValidationError("Longitude values must be between -180 and 180.")
    if frame["noise_measurement"].isna().any():
        raise DatasetValidationError("Dataset contains invalid noise measurements.")
    if frame["accuracy"].isna().any() or frame["accuracy"].lt(0).any():
        raise DatasetValidationError("GPS accuracy must be present and non-negative.")
    if frame["filename"].duplicated().any():
        duplicates = int(frame["filename"].duplicated().sum())
        raise DatasetValidationError(
            f"Dataset contains {duplicates} duplicate recording filenames."
        )

    if "timestamp" in frame.columns:
        parsed = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
        invalid = int(parsed.isna().sum())
        if invalid:
            raise DatasetValidationError(
                f"Timestamp column contains {invalid} invalid or missing values."
            )
        frame["timestamp"] = parsed

    return frame


def audit_metadata(frame: pd.DataFrame) -> DatasetAudit:
    if frame.empty:
        raise DatasetValidationError("Cannot audit an empty metadata frame.")

    coordinate_counts = frame.groupby(["latitude", "longitude"], dropna=False).size()
    noise = frame["noise_measurement"].astype(float)
    accuracy = frame["accuracy"].astype(float)
    return DatasetAudit(
        rows=len(frame),
        columns=list(frame.columns),
        missing_by_column={
            str(column): int(count)
            for column, count in frame.isna().sum().items()
        },
        duplicate_filenames=int(frame["filename"].duplicated().sum()),
        regions={
            str(label): int(count)
            for label, count in frame["region"].value_counts().items()
        },
        classes={
            str(label): int(count)
            for label, count in frame["class"].value_counts().items()
        },
        classes_by_region={
            str(region): {
                str(label): int(count)
                for label, count in region_frame["class"].value_counts().items()
            }
            for region, region_frame in frame.groupby("region")
        },
        coordinate_pairs=int(len(coordinate_counts)),
        repeated_coordinate_pairs=int((coordinate_counts > 1).sum()),
        observations_at_repeated_coordinates=int(
            coordinate_counts[coordinate_counts > 1].sum()
        ),
        timestamp_column_present="timestamp" in frame.columns,
        noise_measurement_min=float(noise.min()),
        noise_measurement_median=float(noise.median()),
        noise_measurement_max=float(noise.max()),
        gps_accuracy_median=float(accuracy.median()),
        gps_accuracy_p95=float(accuracy.quantile(0.95)),
        gps_accuracy_max=float(accuracy.max()),
    )
