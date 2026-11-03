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

from urbannoise.dataset import audit_metadata, load_metadata
from urbannoise.spatial import aggregate_spatial_cells, summarize_categories_by_cell


def main() -> None:
    metadata_path = ROOT / "data" / "raw" / "metadata.csv"
    output_directory = ROOT / "outputs"
    output_directory.mkdir(parents=True, exist_ok=True)

    frame = load_metadata(metadata_path)
    audit = audit_metadata(frame)
    with (output_directory / "dataset_audit.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(audit.to_dict(), stream, indent=2)

    class_counts = frame["class"].value_counts().sort_values()
    figure, axis = plt.subplots(figsize=(10, 7))
    class_counts.plot.barh(ax=axis, color="#4472C4")
    axis.set(title="Urban Noise Uganda: recordings by source category", xlabel="Rows")
    figure.tight_layout()
    figure.savefig(output_directory / "class_distribution.png", dpi=160)
    plt.close(figure)

    class_region_counts = pd.crosstab(frame["class"], frame["region"])
    figure, axis = plt.subplots(figsize=(11, 8))
    class_region_counts.plot.barh(stacked=True, ax=axis, colormap="tab20")
    axis.set(title="Source categories by region", xlabel="Recordings")
    figure.tight_layout()
    figure.savefig(output_directory / "class_distribution_by_region.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    frame["noise_measurement"].plot.hist(
        bins=30,
        ax=axis,
        color="#ED7D31",
        edgecolor="white",
    )
    axis.set(
        title="Dataset-reported noise measurement distribution",
        xlabel="noise_measurement (dataset units; confirm SPL calibration)",
        ylabel="Recordings",
    )
    figure.tight_layout()
    figure.savefig(output_directory / "noise_measurement_distribution.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 7))
    points = axis.scatter(
        frame["longitude"],
        frame["latitude"],
        c=frame["noise_measurement"],
        cmap="inferno",
        s=2,
        alpha=0.55,
    )
    figure.colorbar(
        points,
        ax=axis,
        label="Dataset-reported noise measurement",
    )
    axis.set(
        title="Geographic distribution of sampled recordings",
        xlabel="Longitude",
        ylabel="Latitude",
    )
    figure.tight_layout()
    figure.savefig(output_directory / "spatial_measurements.png", dpi=160)
    plt.close(figure)

    correlated_fields = [
        column
        for column in (
            "noise_measurement",
            "latitude",
            "longitude",
            "altitude",
            "accuracy",
        )
        if column in frame.columns
    ]
    correlation = frame[correlated_fields].corr()
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set(
        xticks=range(len(correlated_fields)),
        yticks=range(len(correlated_fields)),
        xticklabels=correlated_fields,
        yticklabels=correlated_fields,
        title="Numeric metadata correlation (descriptive only)",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")
    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    figure.savefig(output_directory / "metadata_correlations.png", dpi=160)
    plt.close(figure)

    cells = aggregate_spatial_cells(
        frame,
        cell_size_m=250,
        minimum_observations=5,
        maximum_gps_accuracy_m=100,
    )
    cells.to_csv(output_directory / "spatial_cells.csv", index=False)
    class_cells = summarize_categories_by_cell(frame, cell_size_m=250)
    class_cells.to_csv(output_directory / "class_counts_by_cell.csv", index=False)
    print(
        json.dumps(
            {
                "rows": audit.rows,
                "regions": audit.regions,
                "classes": len(audit.classes),
                "timestamp_column_present": audit.timestamp_column_present,
                "spatial_cells_with_minimum_count": len(cells),
                "candidate_hotspot_cells": int(cells["is_hotspot"].sum()),
                "outputs": str(output_directory),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
