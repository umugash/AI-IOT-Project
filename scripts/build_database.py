"""Load CPCB CSV exports (data/cpcb/raw) and saved live snapshots into the SQLite store.

    python scripts/build_database.py
"""
import _bootstrap  # noqa: F401
from noisescope.config import CPCB_SNAPSHOT_DIR, DB_PATH
from noisescope import cpcb, store
from noisescope.analytics import quality_summary


def main() -> None:
    readings = cpcb.load_all_exports()
    n = store.upsert_readings(readings)
    print(f"Stored {n:,} 10-minute readings from CSV exports in {DB_PATH}")
    for snap in sorted(CPCB_SNAPSHOT_DIR.glob("*.json")):
        recs = cpcb.snapshot_json_to_records(snap)
        store.upsert_live_status(recs)
        print(f"Loaded live snapshot {snap.name} ({len(recs)} stations)")
    print(quality_summary(store.load_readings(valid_only=False)).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
