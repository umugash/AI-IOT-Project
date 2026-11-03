from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import pandas as pd

from .temporal import require_timestamps


def iter_historical_events(frame: pd.DataFrame) -> Iterator[dict[str, Any]]:
    ordered = require_timestamps(frame).sort_values("timestamp", kind="stable")
    for record in ordered.to_dict(orient="records"):
        timestamp = record["timestamp"]
        record["timestamp"] = timestamp.isoformat()
        record["event_id"] = record["filename"]
        record["event_type"] = "historical_acoustic_observation"
        record["ingestion_mode"] = "historical_dataset_replay"
        yield record


def encode_event(event: dict[str, Any]) -> str:
    if "timestamp" not in event:
        raise ValueError("An IoT event must include its source timestamp.")
    return json.dumps(event, ensure_ascii=False, allow_nan=False)
