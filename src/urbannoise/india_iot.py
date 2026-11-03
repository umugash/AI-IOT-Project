from __future__ import annotations

import json
import math
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


_PERIOD_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
_REQUIRED_EVENT_FIELDS = {
    "station_id",
    "city",
    "observation_period",
    "day_measurement",
    "night_measurement",
    "data_source",
    "event_kind",
    "is_live_measurement",
}


def _validate_measurement(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number or null.")
    measurement = float(value)
    if not math.isfinite(measurement):
        raise ValueError(f"{field} must be a finite number or null.")
    return measurement


def validate_historical_event(event: Any) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError("Each replay event must be a JSON object.")
    if "timestamp" in event:
        raise ValueError(
            "Monthly source events use observation_period and must not add timestamps."
        )
    missing = _REQUIRED_EVENT_FIELDS.difference(event)
    if missing:
        raise ValueError("Replay event is missing fields: " + ", ".join(sorted(missing)))
    for field in ("station_id", "city", "data_source"):
        if not isinstance(event[field], str) or not event[field].strip():
            raise ValueError(f"{field} must be a non-empty string.")
    if not isinstance(event["observation_period"], str) or not _PERIOD_PATTERN.fullmatch(
        event["observation_period"]
    ):
        raise ValueError("observation_period must use a valid YYYY-MM period.")
    if event["event_kind"] != "historical_monthly_observation":
        raise ValueError("Only historical monthly observations are accepted.")
    if event["is_live_measurement"] is not False:
        raise ValueError("Historical replay events must not be labelled as live.")
    return {
        **event,
        "day_measurement": _validate_measurement(
            event["day_measurement"], "day_measurement"
        ),
        "night_measurement": _validate_measurement(
            event["night_measurement"], "night_measurement"
        ),
    }


def initialize_local_store(database_path: str | Path) -> None:
    database = Path(database_path)
    database.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(database)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS historical_noise_events (
                    station_id TEXT NOT NULL,
                    city TEXT NOT NULL,
                    observation_period TEXT NOT NULL,
                    day_measurement REAL,
                    night_measurement REAL,
                    data_source TEXT NOT NULL,
                    event_kind TEXT NOT NULL,
                    is_live_measurement INTEGER NOT NULL CHECK (is_live_measurement = 0),
                    raw_payload TEXT NOT NULL,
                    PRIMARY KEY (station_id, observation_period, data_source)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_historical_noise_period
                ON historical_noise_events (observation_period, station_id)
                """
            )


def ingest_historical_jsonl(
    event_path: str | Path,
    database_path: str | Path,
    limit: int | None = None,
) -> tuple[int, int]:
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least one when provided.")
    source = Path(event_path)
    if not source.is_file():
        raise FileNotFoundError(f"Historical event source not found: {source}")
    initialize_local_store(database_path)
    validated: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                event = validate_historical_event(json.loads(line))
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"Invalid historical event at line {line_number}: {error}"
                ) from error
            validated.append(event)
            if limit is not None and len(validated) >= limit:
                break

    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.executemany(
                """
                INSERT INTO historical_noise_events (
                    station_id, city, observation_period, day_measurement,
                    night_measurement, data_source, event_kind,
                    is_live_measurement, raw_payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)
                ON CONFLICT (station_id, observation_period, data_source)
                DO UPDATE SET
                    city = excluded.city,
                    day_measurement = excluded.day_measurement,
                    night_measurement = excluded.night_measurement,
                    event_kind = excluded.event_kind,
                    is_live_measurement = 0,
                    raw_payload = excluded.raw_payload
                """,
                [
                    (
                        event["station_id"],
                        event["city"],
                        event["observation_period"],
                        event["day_measurement"],
                        event["night_measurement"],
                        event["data_source"],
                        event["event_kind"],
                        json.dumps(event, sort_keys=True),
                    )
                    for event in validated
                ],
            )
            stored_rows = int(
                connection.execute(
                    "SELECT COUNT(*) FROM historical_noise_events"
                ).fetchone()[0]
            )
    return len(validated), stored_rows


def read_local_events(database_path: str | Path) -> list[dict[str, Any]]:
    database = Path(database_path)
    if not database.is_file():
        raise FileNotFoundError(f"Local event store not found: {database}")
    with closing(sqlite3.connect(database)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT station_id, city, observation_period, day_measurement,
                   night_measurement, data_source, event_kind, is_live_measurement
            FROM historical_noise_events
            ORDER BY observation_period, station_id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def local_store_stats(database_path: str | Path) -> dict[str, int | str | None]:
    database = Path(database_path)
    if not database.is_file():
        return {
            "row_count": 0,
            "live_row_count": 0,
            "first_period": None,
            "last_period": None,
        }
    with closing(sqlite3.connect(database)) as connection:
        row = connection.execute(
            """
            SELECT COUNT(*), SUM(is_live_measurement),
                   MIN(observation_period), MAX(observation_period)
            FROM historical_noise_events
            """
        ).fetchone()
    return {
        "row_count": int(row[0]),
        "live_row_count": int(row[1] or 0),
        "first_period": row[2],
        "last_period": row[3],
    }
