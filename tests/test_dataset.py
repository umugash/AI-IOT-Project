from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from urbannoise.dataset import (
    DatasetValidationError,
    audit_metadata,
    load_metadata,
)


def valid_row(**updates: object) -> dict[str, object]:
    row: dict[str, object] = {
        "filename": "sample.ogg",
        "class": "traffic",
        "class_id": 1,
        "noise_measurement": 50,
        "latitude": 0.3,
        "longitude": 32.5,
        "altitude": 1100,
        "accuracy": 4,
        "submitter_id": 2,
        "region": "Kampala",
    }
    row.update(updates)
    return row


class DatasetTests(unittest.TestCase):
    def test_loads_valid_metadata_without_inventing_timestamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            pd.DataFrame([valid_row()]).to_csv(path, index=False)
            frame = load_metadata(path)

        self.assertNotIn("timestamp", frame.columns)
        self.assertEqual(len(frame), 1)

    def test_rejects_missing_required_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            pd.DataFrame([{"filename": "sample.ogg"}]).to_csv(path, index=False)
            with self.assertRaisesRegex(DatasetValidationError, "missing required"):
                load_metadata(path)

    def test_audit_counts_duplicate_locations_without_treating_them_as_time(self) -> None:
        frame = pd.DataFrame(
            [
                valid_row(filename="one.ogg"),
                valid_row(filename="two.ogg"),
                valid_row(filename="three.ogg", longitude=32.51),
            ]
        )
        audit = audit_metadata(frame)

        self.assertEqual(audit.rows, 3)
        self.assertEqual(audit.coordinate_pairs, 2)
        self.assertEqual(audit.repeated_coordinate_pairs, 1)
        self.assertFalse(audit.timestamp_column_present)

    def test_rejects_duplicate_recording_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metadata.csv"
            pd.DataFrame([valid_row(), valid_row()]).to_csv(path, index=False)
            with self.assertRaisesRegex(DatasetValidationError, "duplicate"):
                load_metadata(path)
