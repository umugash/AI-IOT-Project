from __future__ import annotations

import argparse
import sys
from pathlib import Path

import folium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbannoise.dataset import load_metadata
from urbannoise.spatial import aggregate_spatial_cells


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an interactive map of sampled noise measurements."
    )
    parser.add_argument("--cell-size-m", type=int, default=250)
    parser.add_argument("--minimum-observations", type=int, default=5)
    parser.add_argument("--maximum-gps-accuracy-m", type=float, default=100.0)
    parser.add_argument(
        "--region",
        choices=("all", "Kampala", "Entebbe"),
        default="all",
    )
    parser.add_argument("--class", dest="source_class")
    arguments = parser.parse_args()

    frame = load_metadata(ROOT / "data" / "raw" / "metadata.csv")
    if arguments.region != "all":
        frame = frame.loc[frame["region"] == arguments.region].copy()
    if arguments.source_class is not None:
        if arguments.source_class not in set(frame["class"]):
            raise ValueError(f"Unknown source class: {arguments.source_class}")
        frame = frame.loc[frame["class"] == arguments.source_class].copy()
    cells = aggregate_spatial_cells(
        frame,
        cell_size_m=arguments.cell_size_m,
        minimum_observations=arguments.minimum_observations,
        maximum_gps_accuracy_m=arguments.maximum_gps_accuracy_m,
    )
    if cells.empty:
        raise RuntimeError(
            "No spatial cells meet the minimum count; lower the count threshold "
            "or increase the cell size."
        )

    center = [float(frame["latitude"].mean()), float(frame["longitude"].mean())]
    map_view = folium.Map(location=center, zoom_start=11, control_scale=True)
    for row in cells.itertuples(index=False):
        color = "#c62828" if row.is_hotspot else "#1565c0"
        popup = (
            f"Observations: {row.observation_count}<br>"
            f"Mean reported measurement: {row.mean_reported_noise:.2f}<br>"
            f"Median: {row.median_reported_noise:.2f}<br>"
            f"90th percentile: {row.p90_reported_noise:.2f}<br>"
            f"Mean GPS accuracy (m): {row.mean_gps_accuracy_m:.1f}"
        )
        folium.CircleMarker(
            location=[row.latitude, row.longitude],
            radius=min(12, 4 + row.observation_count**0.5),
            color=color,
            fill=True,
            fill_opacity=0.65,
            popup=folium.Popup(popup, max_width=300),
            tooltip=(
                "Candidate hotspot"
                if row.is_hotspot
                else "Observed spatial cell"
            ),
        ).add_to(map_view)

    map_view.get_root().html.add_child(
        folium.Element(
            "<div style='position:fixed;bottom:25px;left:25px;z-index:9999;"
            "background:white;padding:8px;border:1px solid #888'>"
            "<b>Red:</b> top-decile candidate cells by mean reported value; "
            "<b>blue:</b> other cells. Sampled data, not a population-wide "
            "exposure surface.</div>"
        )
    )
    output_directory = ROOT / "outputs"
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "hotspot_map.html"
    map_view.save(str(output_path))
    print(f"Saved {len(cells):,} aggregated cells to {output_path}")


if __name__ == "__main__":
    main()
