from __future__ import annotations

import unittest

import pandas as pd

from urbannoise.spatial import aggregate_spatial_cells, summarize_categories_by_cell


class SpatialTests(unittest.TestCase):
    def test_aggregates_measurements_and_retains_observation_counts(self) -> None:
        frame = pd.DataFrame(
            {
                "latitude": [0.3, 0.3001, 0.3, 0.3001],
                "longitude": [32.5, 32.5001, 32.5, 32.5001],
                "noise_measurement": [40, 50, 60, 70],
                "accuracy": [3, 5, 4, 500],
            }
        )
        cells = aggregate_spatial_cells(
            frame,
            cell_size_m=500,
            minimum_observations=2,
            hotspot_quantile=0.9,
        )

        self.assertEqual(len(cells), 1)
        self.assertEqual(cells.loc[0, "observation_count"], 3)
        self.assertEqual(cells.loc[0, "mean_reported_noise"], 50)
        self.assertEqual(cells.loc[0, "mean_gps_accuracy_m"], 4)
        self.assertTrue(bool(cells.loc[0, "is_hotspot"]))

    def test_drops_cells_below_the_minimum_sample_count(self) -> None:
        frame = pd.DataFrame(
            {
                "latitude": [0.3],
                "longitude": [32.5],
                "noise_measurement": [40],
            }
        )
        cells = aggregate_spatial_cells(frame, minimum_observations=2)
        self.assertTrue(cells.empty)

    def test_summarizes_source_categories_per_spatial_cell(self) -> None:
        frame = pd.DataFrame(
            {
                "latitude": [0.3, 0.3001, 0.3],
                "longitude": [32.5, 32.5001, 32.5],
                "class": ["traffic", "traffic", "vendor"],
            }
        )
        counts = summarize_categories_by_cell(frame, cell_size_m=500)
        self.assertEqual(set(counts["class"]), {"traffic", "vendor"})
        self.assertEqual(int(counts["observation_count"].sum()), 3)
