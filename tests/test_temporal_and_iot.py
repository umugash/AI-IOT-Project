from __future__ import annotations

import unittest

import pandas as pd

from urbannoise.iot import encode_event, iter_historical_events
from urbannoise.temporal import (
    TemporalDataUnavailableError,
    derive_hotspot_transitions,
)


class TemporalAndIoTTests(unittest.TestCase):
    def test_timestamp_dependent_analysis_refuses_public_metadata_shape(self) -> None:
        frame = pd.DataFrame(
            {
                "latitude": [0.3],
                "longitude": [32.5],
                "noise_measurement": [45],
            }
        )
        with self.assertRaisesRegex(TemporalDataUnavailableError, "no timestamp"):
            derive_hotspot_transitions(frame)
        with self.assertRaisesRegex(TemporalDataUnavailableError, "no timestamp"):
            list(iter_historical_events(frame))

    def test_simulated_replay_orders_real_timestamp_fields(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": ["2025-01-02T00:00:00Z", "2025-01-01T00:00:00Z"],
                "filename": ["later.ogg", "earlier.ogg"],
            }
        )
        events = list(iter_historical_events(frame))
        self.assertEqual(
            [event["filename"] for event in events],
            ["earlier.ogg", "later.ogg"],
        )
        self.assertTrue(
            all(
                event["ingestion_mode"] == "historical_dataset_replay"
                for event in events
            )
        )

    def test_event_encoding_requires_source_timestamp(self) -> None:
        with self.assertRaisesRegex(ValueError, "source timestamp"):
            encode_event({"filename": "sample.ogg"})

    def test_detects_hotspot_shift_between_adjacent_cells(self) -> None:
        rows: list[dict[str, object]] = []
        for timestamp, left, right in (
            ("2025-01-01T12:00:00Z", 80, 20),
            ("2025-01-02T12:00:00Z", 20, 80),
        ):
            for _ in range(5):
                rows.append(
                    {
                        "timestamp": timestamp,
                        "latitude": 0.3,
                        "longitude": 32.000,
                        "noise_measurement": left,
                    }
                )
                rows.append(
                    {
                        "timestamp": timestamp,
                        "latitude": 0.3,
                        "longitude": 32.009,
                        "noise_measurement": right,
                    }
                )
        result = derive_hotspot_transitions(
            pd.DataFrame(rows),
            window="1D",
            cell_size_m=1_000,
            minimum_observations=5,
            hotspot_quantile=0.5,
        )
        latest = result.loc[result["_window"] == result["_window"].max()]
        self.assertIn("disappearance", set(latest["behaviour"]))
        self.assertIn("shift", set(latest["behaviour"]))

    def test_detects_hotspot_intensification(self) -> None:
        rows: list[dict[str, object]] = []
        for timestamp, value in (
            ("2025-01-01T12:00:00Z", 70),
            ("2025-01-02T12:00:00Z", 78),
        ):
            for _ in range(5):
                rows.append(
                    {
                        "timestamp": timestamp,
                        "latitude": 0.3,
                        "longitude": 32.0,
                        "noise_measurement": value,
                    }
                )
                rows.append(
                    {
                        "timestamp": timestamp,
                        "latitude": 0.31,
                        "longitude": 32.01,
                        "noise_measurement": 10,
                    }
                )
        result = derive_hotspot_transitions(
            pd.DataFrame(rows),
            window="1D",
            minimum_observations=5,
            hotspot_quantile=0.1,
            intensification_delta=3,
        )
        self.assertIn("intensification", set(result["behaviour"]))

    def test_does_not_infer_persistence_across_missing_windows(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": [
                    "2025-01-01T12:00:00Z",
                    "2025-01-03T12:00:00Z",
                ],
                "latitude": [0.3, 0.3],
                "longitude": [32.0, 32.0],
                "noise_measurement": [70, 75],
            }
        )
        result = derive_hotspot_transitions(
            frame,
            window="1D",
            minimum_observations=1,
            hotspot_quantile=0.1,
        )
        self.assertEqual(
            set(result["behaviour"]),
            {"insufficient_previous_window"},
        )
