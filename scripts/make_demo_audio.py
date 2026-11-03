"""Build a 60-second demo 'street' recording from ESC-50 clips (for pipeline demos only).

The clips are ESC-50 training data (CC BY-NC 3.0, K. J. Piczak), so classifying this file
demonstrates the pipeline; it is NOT an accuracy test and NOT a field recording.

    python scripts/make_demo_audio.py --esc50 data/external/ESC-50
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

import _bootstrap  # noqa: F401
from noisescope import audio
from noisescope.config import DATA_DIR, EXTERNAL_DIR

SEQUENCE = ["engine", "car_horn", "engine", "siren", "car_horn", "chainsaw",
            "footsteps", "laughing", "car_horn", "church_bells", "dog", "engine"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50", type=Path, default=EXTERNAL_DIR / "ESC-50")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "demo" / "demo_street_mix.ogg")
    args = ap.parse_args()
    meta = pd.read_csv(args.esc50 / "meta" / "esc50.csv")
    meta = meta[meta["fold"] == 5]
    rng = np.random.default_rng(3)
    parts = []
    for cat in SEQUENCE:
        f = meta[meta["category"] == cat].sample(1, random_state=int(rng.integers(1e6)))["filename"].iloc[0]
        y = audio.load_audio(str(args.esc50 / "audio" / f))
        y = y / (np.max(np.abs(y)) + 1e-9) * rng.uniform(0.3, 0.8)
        parts.append(np.pad(y, (0, int(audio.CLIP_SECONDS * audio.SR) - len(y))))
    mix = np.concatenate(parts)
    bed = audio.load_audio(str(args.esc50 / "audio" / meta[meta["category"] == "engine"]["filename"].iloc[0]))
    mix = mix + 0.03 * np.resize(bed / (np.max(np.abs(bed)) + 1e-9), mix.shape)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.out, mix / np.max(np.abs(mix)) * 0.9, audio.SR, format="OGG", subtype="VORBIS")
    print(f"wrote {args.out} ({len(mix) / audio.SR:.0f} s): {', '.join(SEQUENCE)}")


if __name__ == "__main__":
    main()
