from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbannoise.training import predict_recording


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify one OGG recording.")
    parser.add_argument("audio", type=Path)
    parser.add_argument(
        "--model",
        type=Path,
        default=ROOT / "models" / "urban_noise_cnn.pt",
    )
    arguments = parser.parse_args()
    print(json.dumps(predict_recording(arguments.model, arguments.audio), indent=2))


if __name__ == "__main__":
    main()
