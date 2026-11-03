"""Train the urban sound-source classifier on ESC-50 and report 5-fold CV metrics.

    python scripts/train_sound_classifier.py                 # downloads ESC-50 (~600 MB) via git
    python scripts/train_sound_classifier.py --esc50 PATH    # use an existing copy
"""
import argparse
import json
import subprocess
from pathlib import Path

import _bootstrap  # noqa: F401
from noisescope import audio
from noisescope.config import EXTERNAL_DIR, OUTPUTS_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--esc50", type=Path, default=EXTERNAL_DIR / "ESC-50")
    args = ap.parse_args()
    if not (args.esc50 / "meta" / "esc50.csv").exists():
        print(f"Downloading ESC-50 into {args.esc50} ...")
        args.esc50.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/karolpiczak/ESC-50.git", str(args.esc50)], check=True)
    clf = audio.train(args.esc50)
    r = clf.report
    print(f"5-fold CV on {r['n_clips']} clips: accuracy {r['accuracy']:.3f}, "
          f"balanced accuracy {r['balanced_accuracy']:.3f}, macro-F1 {r['macro_f1']:.3f}")
    for c in r["classes"]:
        pc = r["per_class"][c]
        print(f"  {c:<26} precision {pc['precision']:.2f}  recall {pc['recall']:.2f}  f1 {pc['f1-score']:.2f}")
    OUTPUTS_DIR.mkdir(exist_ok=True)
    (OUTPUTS_DIR / "sound_classifier_report.json").write_text(json.dumps(r, indent=1))
    print(f"Model saved to {audio.MODEL_PATH}")


if __name__ == "__main__":
    main()
