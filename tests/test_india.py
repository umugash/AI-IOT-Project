from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from urbannoise.india_data import build_lag_features, load_monthly_panel
from urbannoise.india_forecasting import chronological_split
from urbannoise.india_forecast_product import forecast_next_month_persistence
from urbannoise.india_hotspots import (
    build_hotspot_behaviour,
    summarize_relative_hotspots,
)
from urbannoise.india_iot import ingest_historical_jsonl, read_local_events
from urbannoise.india_replay import write_historical_jsonl


ROOT = Path(__file__).resolve().parents[1]
INDIA_DIR = ROOT / "data" / "external" / "india"


class IndiaDataTests(unittest.TestCase):
    def test_loads_chennai_and_builds_only_prior_month_features(self) -> None:
        panel = load_monthly_panel(
            INDIA_DIR / "cpcb_noise_2011_2018" / "station_month.csv",
            INDIA_DIR / "cpcb_noise_2011_2018" / "stations.csv",
            city="Chennai",
        )
        self.assertEqual(len(panel), 715)
        self.assertEqual(panel["Station"].nunique(), 10)
        featured = build_lag_features(panel)
        row = featured.loc[
            featured["Station"].eq("CHE01")
            & featured["observation_period"].eq("2011-03")
        ].iloc[0]
        previous = panel.loc[
            panel["Station"].eq("CHE01")
            & panel["observation_period"].eq("2011-02")
        ].iloc[0]
        self.assertEqual(row["Day_lag_1"], previous["Day"])
        self.assertEqual(row["Night_lag_1"], previous["Night"])
        self.assertTrue(pd.isna(row["Day_lag_12"]))
        self.assertNotIn("timestamp", featured.columns)

    def test_calendar_lags_do_not_cross_missing_months(self) -> None:
        panel = pd.DataFrame(
            {
                "Station": ["A", "A", "A", "A"],
                "month_index": [24133, 24134, 24136, 24137],
                "Year": [2011, 2011, 2011, 2011],
                "Month": [2, 3, 5, 6],
                "Day": [10.0, 20.0, 50.0, 60.0],
                "Night": [1.0, 2.0, 5.0, 6.0],
            }
        )
        featured = build_lag_features(panel)
        may = featured.loc[featured["Month"].eq(5)].iloc[0]
        june = featured.loc[featured["Month"].eq(6)].iloc[0]
        self.assertTrue(pd.isna(may["Day_lag_1"]))
        self.assertTrue(pd.isna(may["Day_rolling_mean_3"]))
        self.assertEqual(june["Day_lag_1"], 50.0)

    def test_relative_hotspots_are_nonregulatory_and_station_level(self) -> None:
        panel = load_monthly_panel(
            INDIA_DIR / "cpcb_noise_2011_2018" / "station_month.csv",
            INDIA_DIR / "cpcb_noise_2011_2018" / "stations.csv",
            city="Chennai",
        )
        station, monthly = summarize_relative_hotspots(panel)
        self.assertEqual(len(station), 20)
        self.assertIn("relative_high_threshold", station.columns)
        self.assertTrue(
            station["threshold_basis"].str.contains("not a regulatory standard").all()
        )
        self.assertFalse(monthly.empty)
        self.assertIn("relative_high_frequency", monthly.columns)
        self.assertTrue(monthly["relative_high_frequency"].between(0, 1).all())

    def test_historical_replay_uses_period_not_fabricated_timestamp(self) -> None:
        panel = load_monthly_panel(
            INDIA_DIR / "cpcb_noise_2011_2018" / "station_month.csv",
            INDIA_DIR / "cpcb_noise_2011_2018" / "stations.csv",
            city="Chennai",
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "events.jsonl"
            count = write_historical_jsonl(panel, output)
            events = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(count, 715)
        self.assertEqual(events[0]["observation_period"], "2011-02")
        self.assertNotIn("timestamp", events[0])
        self.assertFalse(events[0]["is_live_measurement"])
        self.assertEqual(events[0]["event_kind"], "historical_monthly_observation")

    def test_chronological_split_never_uses_future_rows_for_training(self) -> None:
        panel = load_monthly_panel(
            INDIA_DIR / "cpcb_noise_2011_2018" / "station_month.csv",
            INDIA_DIR / "cpcb_noise_2011_2018" / "stations.csv",
            city="Chennai",
        )
        featured = build_lag_features(panel).dropna(subset=["Day_lag_12"])
        train, validation, test = chronological_split(featured)
        self.assertLess(train["Year"].max(), validation["Year"].min())
        self.assertEqual(validation["Year"].unique().tolist(), [2016])
        self.assertGreaterEqual(test["Year"].min(), 2017)

    def test_hotspot_behaviour_labels_emergence_persistence_and_reduction(self) -> None:
        panel = pd.DataFrame(
            {
                "Station": ["A"] * 6,
                "Name": ["Station A"] * 6,
                "City": ["Chennai"] * 6,
                "Year": [2020] * 6,
                "Month": [1, 2, 3, 4, 5, 6],
                "month_index": list(range(24240, 24246)),
                "observation_period": [f"2020-{month:02d}" for month in range(1, 7)],
                "Day": [0.0, 0.0, 10.0, 10.0, 0.0, 0.0],
                "Night": [0.0, 0.0, 10.0, 10.0, 0.0, 0.0],
            }
        )
        behaviour = build_hotspot_behaviour(panel, quantile=0.5)
        day = behaviour.loc[behaviour["metric"].eq("Day")]
        self.assertIn("emergence", day["relative_hotspot_behaviour"].tolist())
        self.assertIn("persistence", day["relative_hotspot_behaviour"].tolist())
        self.assertIn("reduction", day["relative_hotspot_behaviour"].tolist())
        self.assertIn("intensity_behaviour", day.columns)

    def test_local_replay_store_is_validated_ordered_and_idempotent(self) -> None:
        event = {
            "station_id": "CHE01",
            "city": "Chennai",
            "observation_period": "2018-12",
            "day_measurement": 64.0,
            "night_measurement": 55.0,
            "data_source": "CPCB historical test",
            "event_kind": "historical_monthly_observation",
            "is_live_measurement": False,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "events.jsonl"
            store = path / "events.sqlite3"
            source.write_text(json.dumps(event) + "\n", encoding="utf-8")
            self.assertEqual(ingest_historical_jsonl(source, store), (1, 1))
            self.assertEqual(ingest_historical_jsonl(source, store), (1, 1))
            stored = read_local_events(store)
        self.assertEqual(stored[0]["observation_period"], "2018-12")
        self.assertEqual(stored[0]["is_live_measurement"], 0)

    def test_local_replay_can_process_a_limited_ordered_batch(self) -> None:
        events = [
            {
                "station_id": "CHE01",
                "city": "Chennai",
                "observation_period": period,
                "day_measurement": 64.0,
                "night_measurement": 55.0,
                "data_source": "CPCB historical test",
                "event_kind": "historical_monthly_observation",
                "is_live_measurement": False,
            }
            for period in ("2018-10", "2018-11", "2018-12")
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            source = path / "events.jsonl"
            store = path / "events.sqlite3"
            source.write_text(
                "\n".join(json.dumps(event) for event in events) + "\n",
                encoding="utf-8",
            )
            self.assertEqual(ingest_historical_jsonl(source, store, limit=2), (2, 2))
            stored = read_local_events(store)
        self.assertEqual(
            [row["observation_period"] for row in stored], ["2018-10", "2018-11"]
        )

    def test_local_replay_rejects_live_and_malformed_events(self) -> None:
        event = {
            "station_id": "CHE01",
            "city": "Chennai",
            "observation_period": "2018-12",
            "day_measurement": 64.0,
            "night_measurement": 55.0,
            "data_source": "CPCB historical test",
            "event_kind": "historical_monthly_observation",
            "is_live_measurement": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "events.jsonl"
            store = Path(directory) / "events.sqlite3"
            source.write_text(json.dumps(event) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not be labelled as live"):
                ingest_historical_jsonl(source, store)
            event["is_live_measurement"] = False
            event["observation_period"] = "2018-13"
            source.write_text(json.dumps(event) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "valid YYYY-MM"):
                ingest_historical_jsonl(source, store)
            event["observation_period"] = "2018-12"
            event["day_measurement"] = float("nan")
            source.write_text(json.dumps(event) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "finite number or null"):
                ingest_historical_jsonl(source, store)
            event["day_measurement"] = 64.0
            event["timestamp"] = "2018-12-01T00:00:00Z"
            source.write_text(json.dumps(event) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must not add timestamps"):
                ingest_historical_jsonl(source, store)

    def test_next_month_forecast_is_historical_and_uses_empirical_error_ranges(self) -> None:
        month_index = 2018 * 12 + 11
        panel = pd.DataFrame(
            [
                {
                    "Station": station,
                    "Name": f"Station {station}",
                    "City": "Chennai",
                    "Year": 2018,
                    "Month": 12,
                    "month_index": month_index,
                    "observation_period": "2018-12",
                    "Day": day,
                    "Night": night,
                }
                for station, day, night in (("A", 60.0, 50.0), ("B", 75.0, 65.0))
            ]
        )
        backtest = pd.DataFrame(
            [
                {"Station": station, "metric": metric, "method": "persistence", "error": error}
                for station in ("A", "B")
                for metric in ("Day", "Night")
                for error in (-2.0, 0.0, 2.0)
            ]
        )
        forecast = forecast_next_month_persistence(panel, backtest)
        self.assertEqual(set(forecast["forecast_period"]), {"2019-01"})
        self.assertFalse(forecast["is_current_forecast"].any())
        self.assertTrue(
            (forecast["prediction_lower_heuristic"] <= forecast["predicted_value"]).all()
        )
        self.assertTrue(
            (forecast["predicted_value"] <= forecast["prediction_upper_heuristic"]).all()
        )
        self.assertTrue(
            forecast["hotspot_probability_estimate"].between(0, 1).all()
        )


if __name__ == "__main__":
    unittest.main()
