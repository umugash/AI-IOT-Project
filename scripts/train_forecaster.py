"""Backtest the hotspot forecaster against baselines, then train it on all data.

    python scripts/train_forecaster.py
"""
import json

import _bootstrap  # noqa: F401
from noisescope import analytics, forecasting, store
from noisescope.config import OUTPUTS_DIR


def main() -> None:
    hourly = analytics.hourly_levels(analytics.clean_readings(store.load_readings(valid_only=False)))
    print(f"{len(hourly):,} station-hours from {hourly['station_id'].nunique()} stations")
    ev = forecasting.evaluate(hourly)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    print("\nMAE (dB) by forecast horizon (hours), chronological hold-out:")
    print(ev["regression"].pivot(index="method", columns="horizon_h", values="MAE_dB").round(2).to_string())
    print("\nHotspot classification:")
    print(ev["classification"].round(3).to_string(index=False))
    print("\n80% range coverage:", ev["interval_coverage_80"].set_index("horizon_h")["coverage"].round(2).to_dict())
    for k in ["regression", "classification", "interval_coverage_80", "per_station", "feature_importance"]:
        ev[k].to_csv(OUTPUTS_DIR / f"forecast_{k}.csv", index=False)
    forecasting.save_evaluation(ev)
    models = forecasting.train_final(hourly)
    print(f"\nFinal model trained on {models.trained_on['rows']:,} rows -> {forecasting.MODEL_DIR}")
    (OUTPUTS_DIR / "forecast_summary.json").write_text(json.dumps(
        {"n_train": ev["n_train"], "n_test": ev["n_test"], **models.trained_on}, indent=1))


if __name__ == "__main__":
    main()
