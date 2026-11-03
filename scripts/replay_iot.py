from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbannoise.dataset import load_metadata
from urbannoise.iot import encode_event, iter_historical_events
from urbannoise.temporal import TemporalDataUnavailableError, require_timestamps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay real timestamped records as a local IoT simulation."
    )
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/raw/metadata.csv")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "replayed_events.jsonl",
    )
    arguments = parser.parse_args()
    if arguments.limit is not None and arguments.limit < 1:
        parser.error("--limit must be positive.")

    metadata = load_metadata(arguments.metadata)
    try:
        timestamped = require_timestamps(metadata)
    except TemporalDataUnavailableError as error:
        parser.error(str(error))
    events = iter_historical_events(timestamped)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with arguments.output.open("w", encoding="utf-8") as destination:
        for event in events:
            if arguments.limit is not None and count >= arguments.limit:
                break
            destination.write(encode_event(event) + "\n")
            count += 1
    print(json.dumps({"events_replayed": count, "output": str(arguments.output)}))


if __name__ == "__main__":
    main()
