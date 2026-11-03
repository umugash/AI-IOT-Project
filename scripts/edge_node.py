"""Run an AIoT edge node on an audio file or a live microphone.

Examples
    # analyse a phone recording made at SRM main gate, starting at a known time
    python scripts/edge_node.py --node-id srm-main-gate --lat 12.8231 --lon 80.0442 \
        --audio recordings/main_gate.wav --start "2026-11-03 08:30:00"

    # listen to the microphone for 10 minutes (needs `pip install sounddevice`)
    python scripts/edge_node.py --node-id my-desk --mic --minutes 10

Options --calibration-db converts dBFS to dB SPL. Measure the offset once by recording next
to a sound level meter: offset = meter_dB - reported_dBFS.
"""
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401
from noisescope import audio
from noisescope.config import OUTPUTS_DIR
from noisescope.edge import EdgeNode, publish


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node-id", required=True)
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", type=Path, help="audio file (wav/ogg/flac/mp3)")
    src.add_argument("--mic", action="store_true", help="record from the default microphone")
    ap.add_argument("--minutes", type=float, default=5.0, help="microphone duration")
    ap.add_argument("--start", help="IST start time of the recording 'YYYY-MM-DD HH:MM:SS' (default: now)")
    ap.add_argument("--calibration-db", type=float, help="offset added to dBFS to obtain dB SPL")
    ap.add_argument("--window", type=float, default=audio.CLIP_SECONDS, help="analysis window (s)")
    ap.add_argument("--mqtt-host", help="also publish events to this MQTT broker")
    args = ap.parse_args()

    clf = audio.load_classifier()
    node = EdgeNode(args.node_id, args.lat, args.lon, args.calibration_db, args.window)
    log = OUTPUTS_DIR / "edge_events.jsonl"

    if args.audio:
        start = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S") if args.start else datetime.now()
        y = audio.load_audio(str(args.audio))
        events = node.events_from_signal(y, start, args.audio.name, clf)
        n = publish(events, log, mqtt_host=args.mqtt_host)
        print(f"{n} events from {args.audio.name}")
        for e in events:
            print(f"  {e['ts']}  {e['level_db']:6.1f} dB  {e['source_class']:<26} {e['confidence']}")
        return

    import sounddevice as sd  # optional dependency for live capture
    chunk = int(args.window * audio.SR)
    end = datetime.now().timestamp() + args.minutes * 60
    print(f"Listening on the default microphone for {args.minutes} min (Ctrl+C to stop)")
    while datetime.now().timestamp() < end:
        start = datetime.now()
        rec = sd.rec(chunk, samplerate=audio.SR, channels=1, dtype="float32")
        sd.wait()
        events = node.events_from_signal(np.asarray(rec[:, 0]), start, "microphone", clf)
        publish(events, log, mqtt_host=args.mqtt_host)
        for e in events:
            print(f"  {e['ts']}  {e['level_db']:6.1f} dB  {e['source_class']:<26} {e['confidence']}")


if __name__ == "__main__":
    main()
