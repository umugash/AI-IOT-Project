"""Poll the CPCB NANMN portal for the latest Chennai readings and store them.

Run on a machine that can reach http://cpcbnoise.com (e.g. your laptop):

    python scripts/poll_cpcb_live.py --once            # one poll
    python scripts/poll_cpcb_live.py --interval 600    # every 10 minutes until stopped

Each poll updates the live status table and appends new 10-minute readings, so the
dashboard and the hotspot forecaster work on current data.
"""
import argparse
import time
from datetime import datetime

import _bootstrap  # noqa: F401
from noisescope import cpcb, store
from noisescope.config import CPCB_SNAPSHOT_DIR, STATIONS


def poll_once(save_raw: bool) -> None:
    records = cpcb.poll_live(save_raw_dir=CPCB_SNAPSHOT_DIR / "raw_xml" if save_raw else None)
    good = cpcb.good_records(records)
    store.upsert_live_status(good)
    readings = cpcb.live_records_to_readings(good)
    store.upsert_readings(readings)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {len(good)}/{len(records)} stations OK, stored {len(readings)} readings")
    for r in records:
        name = STATIONS[r["station_id"]].label
        if r.get("status") == "ok":
            print(f"   {name:<20} {r['observed_ts'] or '-':<20} LAF={r['lins']}")
        else:
            print(f"   {name:<20} {r['status'].upper()}: {r['detail'][:120]}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true", help="poll a single time and exit")
    ap.add_argument("--interval", type=int, default=600, help="seconds between polls (default 600)")
    ap.add_argument("--save-raw", action="store_true", help="keep raw XML responses for debugging")
    args = ap.parse_args()
    while True:
        try:
            poll_once(args.save_raw)
        except Exception as exc:
            print(f"poll failed: {exc}")
        if args.once:
            break
        time.sleep(max(args.interval, 120))


if __name__ == "__main__":
    main()
