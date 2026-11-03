from __future__ import annotations

import pandas as pd


def forecast_next_month_persistence(
    panel: pd.DataFrame,
    backtest_predictions: pd.DataFrame,
    hotspot_quantile: float = 0.75,
    history_months: int = 24,
) -> pd.DataFrame:
    if not 0 < hotspot_quantile < 1:
        raise ValueError("hotspot_quantile must be between 0 and 1.")
    if history_months < 1:
        raise ValueError("history_months must be at least one.")
    required_panel = {
        "Station",
        "Name",
        "City",
        "Year",
        "Month",
        "month_index",
        "observation_period",
        "Day",
        "Night",
    }
    if not required_panel.issubset(panel.columns):
        raise ValueError(
            "Panel is missing required fields: "
            + ", ".join(sorted(required_panel.difference(panel.columns)))
        )
    required_predictions = {"Station", "metric", "method", "error"}
    if not required_predictions.issubset(backtest_predictions.columns):
        raise ValueError(
            "Backtest predictions are missing required fields: "
            + ", ".join(
                sorted(required_predictions.difference(backtest_predictions.columns))
            )
        )
    if panel.empty:
        raise ValueError("Cannot forecast from an empty station-month panel.")

    ordered = panel.sort_values(["Station", "month_index"]).copy()
    results: list[dict[str, object]] = []
    for metric in ("Day", "Night"):
        valid = ordered.dropna(subset=[metric])
        if valid.empty:
            raise ValueError(f"No valid {metric} observations are available.")
        threshold = float(valid[metric].quantile(hotspot_quantile))
        station_latest = valid.groupby("Station", sort=True).tail(1)
        station_errors = backtest_predictions.loc[
            backtest_predictions["metric"].eq(metric)
            & backtest_predictions["method"].eq("persistence")
        ]
        pooled_errors = station_errors["error"].dropna()
        if pooled_errors.empty:
            raise ValueError(f"No persistence backtest errors are available for {metric}.")
        pooled_lower = float(pooled_errors.quantile(0.10))
        pooled_upper = float(pooled_errors.quantile(0.90))

        for latest in station_latest.itertuples(index=False):
            next_index = int(latest.month_index) + 1
            next_year, month_zero_based = divmod(next_index, 12)
            next_month = month_zero_based + 1
            station_history = valid.loc[
                valid["Station"].eq(latest.Station)
                & valid["month_index"].le(int(latest.month_index))
            ].tail(history_months)
            recent_frequency = float(station_history[metric].gt(threshold).mean())
            residuals = station_errors.loc[
                station_errors["Station"].eq(latest.Station), "error"
            ].dropna()
            lower_error = (
                float(residuals.quantile(0.10)) if not residuals.empty else pooled_lower
            )
            upper_error = (
                float(residuals.quantile(0.90)) if not residuals.empty else pooled_upper
            )
            prediction = float(getattr(latest, metric))
            results.append(
                {
                    "Station": latest.Station,
                    "Name": latest.Name,
                    "City": latest.City,
                    "metric": metric,
                    "forecast_period": f"{next_year:04d}-{next_month:02d}",
                    "last_observed_period": latest.observation_period,
                    "last_observed_value": prediction,
                    "predicted_value": prediction,
                    "forecast_method": "persistence",
                    "historical_relative_hotspot_threshold": threshold,
                    "hotspot_probability_estimate": recent_frequency,
                    "probability_basis": (
                        f"Empirical high-reading frequency in latest "
                        f"{len(station_history)} available station-months; "
                        "uncalibrated historical proxy, not a learned probability."
                    ),
                    "prediction_lower_heuristic": prediction + lower_error,
                    "prediction_upper_heuristic": prediction + upper_error,
                    "interval_basis": (
                        "Station persistence residual 10th–90th percentiles from "
                        "2017–2018 chronological backtest; heuristic range, not a "
                        "calibrated confidence interval."
                    ),
                    "is_current_forecast": False,
                    "source_metric_warning": (
                        "Day/Night unit, weighting, and aggregation are unverified."
                    ),
                }
            )
    return pd.DataFrame(results).sort_values(["metric", "Station"]).reset_index(drop=True)
