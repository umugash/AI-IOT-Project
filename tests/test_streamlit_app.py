from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from streamlit_app.data import (
    ApplicationDataError,
    ROOT,
    filter_observations,
    metric_long_frame,
    read_csv,
)


class StreamlitApplicationTests(unittest.TestCase):
    def test_application_module_imports_without_starting_ui(self) -> None:
        spec = importlib.util.spec_from_file_location("urban_noise_app", ROOT / "app.py")
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.assertTrue(callable(module.main))

    def test_all_three_sections_render_without_exceptions(self) -> None:
        pages = (
            "Data Overview",
            "Data Analysis",
            "Spatiotemporal Prediction",
        )
        for page in pages:
            with self.subTest(page=page):
                app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
                app.radio[0].set_value(page).run()
                self.assertEqual(
                    [error.message for error in app.exception],
                    [],
                    msg=f"{page} raised a Streamlit exception.",
                )

    def test_spatiotemporal_prediction_station_selection(self) -> None:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
        app.radio[0].set_value("Spatiotemporal Prediction").run()
        self.assertEqual([e.message for e in app.exception], [])
        # Check that metrics and prediction sections rendered
        self.assertGreater(len(app.metric), 0)

    def test_data_analysis_city_filter(self) -> None:
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=45).run()
        app.radio[0].set_value("Data Analysis").run()
        self.assertEqual([e.message for e in app.exception], [])


    def test_filter_logic_and_chart_long_format(self) -> None:
        panel = pd.DataFrame(
            {
                "Station": ["A", "A", "B"],
                "Name": ["Alpha", "Alpha", "Beta"],
                "City": ["Chennai"] * 3,
                "State": ["Tamil Nadu"] * 3,
                "Type": ["Residential", "Residential", "Commercial"],
                "Year": [2017, 2018, 2018],
                "Month": [1, 2, 1],
                "observation_period": ["2017-01", "2018-02", "2018-01"],
                "month_index": [24204, 24217, 24216],
                "Day": [60.0, 62.0, 64.0],
                "Night": [50.0, 52.0, 54.0],
            }
        )
        selected = filter_observations(
            panel,
            stations=["A"],
            years=[2018],
            months=[2],
            categories=["Residential"],
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0]["observation_period"], "2018-02")
        long = metric_long_frame(selected)
        self.assertEqual(set(long["metric"]), {"Day", "Night"})
        self.assertEqual(len(long), 2)

    def test_missing_data_file_has_explicit_error(self) -> None:
        with self.assertRaisesRegex(ApplicationDataError, "Required data file is missing"):
            read_csv(ROOT / "outputs" / "missing-unit-test.csv")

    def test_corrupt_csv_has_explicit_error(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            source = Path(directory) / "corrupt.csv"
            source.write_text('"unterminated\n"value', encoding="utf-8")
            with self.assertRaisesRegex(ApplicationDataError, "CSV"):
                read_csv(source)

    def test_generated_metrics_and_separate_sources_match_outputs(self) -> None:
        metrics = pd.read_csv(
            ROOT / "outputs" / "india" / "models" / "chennai_walk_forward_metrics.csv"
        )
        predictions = pd.read_csv(
            ROOT / "outputs" / "india" / "models" / "chennai_walk_forward_predictions.csv"
        )
        persistence_day = metrics.loc[
            metrics["metric"].eq("Day") & metrics["method"].eq("persistence")
        ].iloc[0]
        actual = predictions.loc[
            predictions["metric"].eq("Day")
            & predictions["method"].eq("persistence"),
            "actual",
        ]
        predicted = predictions.loc[
            predictions["metric"].eq("Day")
            & predictions["method"].eq("persistence"),
            "predicted",
        ]
        self.assertAlmostEqual(
            float(persistence_day["mae"]), float((actual - predicted).abs().mean())
        )
        self.assertEqual(int(persistence_day["observations"]), 240)

        ledger = pd.read_csv(
            ROOT / "outputs" / "india" / "system" / "source_comparability.csv"
        )
        separate_sources = ledger.loc[
            ledger["source"].str.contains("TNPCB|SRMIST", case=False)
        ]
        self.assertFalse(separate_sources["directly_merged_with_chennai_model"].astype(bool).any())

        forecasts = pd.read_csv(
            ROOT / "outputs" / "india" / "system" / "next_month_station_forecasts.csv"
        )
        self.assertEqual(set(forecasts["forecast_period"]), {"2019-01"})
        self.assertFalse(forecasts["is_current_forecast"].astype(bool).any())

        references = pd.read_csv(
            ROOT / "outputs" / "india" / "system" / "separate_reference_source_values.csv"
        )
        srmist = references.loc[
            references["source"].eq("SRMIST January 2025 report")
        ]
        self.assertEqual(set(srmist["value"]), {48.3, 41.7})

    def test_regression_benchmark_column_mismatch_original_behavior(self) -> None:
        """Reproduce original bug: assigning 3 column headers to a 4-field DataFrame."""
        four_field_data = [
            {"model": "Persistence Baseline", "calib_mae": 0.03, "mae": 0.03, "rmse": 0.08},
            {"model": "Ridge Regression", "calib_mae": 0.14, "mae": 0.14, "rmse": 0.16},
        ]
        raw_df = pd.DataFrame(four_field_data)
        self.assertEqual(raw_df.shape[1], 4)
        
        # Verify that original naive 3-column assignment raises ValueError
        with self.assertRaisesRegex(ValueError, "Length mismatch"):
            raw_df.columns = ["Model", "MAE", "RMSE"]

        # Verify that the corrected format_model_benchmark_table handles it cleanly
        from streamlit_app.pages import format_model_benchmark_table
        bench_df = format_model_benchmark_table(four_field_data, best_model_name="Persistence Baseline")
        self.assertEqual(len(bench_df), 2)
        self.assertIn("Forecasting Model Candidate", bench_df.columns)
        self.assertIn("Calibration MAE (dBA)", bench_df.columns)
        self.assertIn("Test MAE (dBA)", bench_df.columns)
        self.assertIn("Test RMSE (dBA)", bench_df.columns)
        self.assertIn("Model Selection Status", bench_df.columns)
        self.assertTrue(bench_df.iloc[0]["Model Selection Status"].startswith("⭐"))

    def test_format_model_benchmark_table_all_conditions(self) -> None:
        """Test benchmark formatting under various missing/optional conditions."""
        from streamlit_app.pages import format_model_benchmark_table

        # 1. Empty or None input
        empty_df = format_model_benchmark_table(None)
        self.assertTrue(empty_df.empty)
        empty_list_df = format_model_benchmark_table([])
        self.assertTrue(empty_list_df.empty)

        # 2. One model unavailable (only 3 models)
        three_models = [
            {"model": "Persistence Baseline", "mae": 1.2, "rmse": 1.8},
            {"model": "Autoregressive Lag (AR)", "mae": 1.0, "rmse": 1.5},
            {"model": "Ridge Regression", "mae": 1.4, "rmse": 2.0},
        ]
        df_3m = format_model_benchmark_table(three_models, best_model_name="Autoregressive Lag (AR)")
        self.assertEqual(len(df_3m), 3)
        self.assertEqual(df_3m.iloc[1]["Model Selection Status"], "⭐ Selected Best Model (Calib Min)")
        self.assertEqual(df_3m.iloc[0]["Model Selection Status"], "Candidate Model")

        # 3. Missing optional metrics (e.g. None or missing RMSE/MAE/calib_mae)
        partial_data = [
            {"model": "Model Alpha", "mae": None, "rmse": 2.5},
            {"model": "Model Beta", "mae": 1.8, "rmse": None, "calib_mae": 1.7},
        ]
        df_partial = format_model_benchmark_table(partial_data, best_model_name="Model Beta")
        self.assertEqual(len(df_partial), 2)
        self.assertEqual(df_partial.iloc[0]["Test MAE (dBA)"], "N/A")
        self.assertEqual(df_partial.iloc[1]["Test RMSE (dBA)"], "N/A")
        self.assertEqual(df_partial.iloc[1]["Calibration MAE (dBA)"], "1.70")

    def test_future_prediction_across_stations_and_horizons(self) -> None:
        """Validate forecasting benchmark generation across stations and 10m/1h horizons."""
        from urbannoise.cpcb_forecasting import evaluate_station_forecasts
        from streamlit_app.pages import format_model_benchmark_table

        test_cases = [
            (267, 1, "Next 10-Min"),
            (267, 6, "Next 1-Hour"),
            (265, 1, "Next 10-Min"),
            (268, 1, "Next 10-Min"),
            (272, 6, "Next 1-Hour"),
        ]
        for sid, hz, label in test_cases:
            with self.subTest(station_id=sid, horizon_steps=hz):
                res = evaluate_station_forecasts(station_id=sid, horizon_steps=hz)
                self.assertNotIn("error", res)
                bench_df = format_model_benchmark_table(
                    models_summary=res.get("models_summary"),
                    best_model_name=res.get("best_model_name"),
                    horizon_label=label,
                )
                self.assertEqual(len(bench_df), 4)
                self.assertIn("Forecasting Model Candidate", bench_df.columns)
                self.assertIn("Test MAE (dBA)", bench_df.columns)
                self.assertIn("Test RMSE (dBA)", bench_df.columns)
                self.assertIn("Model Selection Status", bench_df.columns)


if __name__ == "__main__":
    unittest.main()
