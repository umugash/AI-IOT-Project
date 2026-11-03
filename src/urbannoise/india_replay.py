from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def write_historical_jsonl(panel: pd.DataFrame, output_path: str | Path) -> int:
    required = {
        "Station",
        "City",
        "Year",
        "Month",
        "Day",
        "Night",
        "data_source",
    }
    if not required.issubset(panel.columns):
        raise ValueError(
            "Replay panel is missing columns: "
            + ", ".join(sorted(required.difference(panel.columns)))
        )
    ordered = panel.sort_values(["Year", "Month", "Station"])
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8", newline="\n") as output:
        for row in ordered.itertuples(index=False):
            event = {
                "station_id": row.Station,
                "city": row.City,
                "observation_period": f"{int(row.Year):04d}-{int(row.Month):02d}",
                "day_measurement": None if pd.isna(row.Day) else float(row.Day),
                "night_measurement": None if pd.isna(row.Night) else float(row.Night),
                "data_source": row.data_source,
                "event_kind": "historical_monthly_observation",
                "is_live_measurement": False,
            }
            output.write(json.dumps(event, sort_keys=True) + "\n")
            count += 1
    return count
