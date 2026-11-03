from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from urbannoise.dataset import load_metadata
from urbannoise.training import train_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Uganda audio CNN.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/raw/metadata.csv")
    parser.add_argument(
        "--audio-directory",
        type=Path,
        default=ROOT / "data/raw/audio_files",
    )
    parser.add_argument("--output-directory", type=Path, default=ROOT / "models")
    arguments = parser.parse_args()

    metadata = load_metadata(arguments.metadata)
    results = train_classifier(
        metadata,
        arguments.audio_directory,
        output_directory=arguments.output_directory,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        max_samples=arguments.max_samples,
        random_seed=arguments.seed,
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
