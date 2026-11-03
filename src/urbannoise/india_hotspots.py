from __future__ import annotations

import numpy as np
import pandas as pd


def summarize_relative_hotspots(
    panel: pd.DataFrame, quantile: float = 0.75
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1.")
    station_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []

    for metric in ("Day", "Night"):
        observations = panel.dropna(subset=[metric]).copy()
        threshold = float(observations[metric].quantile(quantile))
        observations["relative_high"] = observations[metric].gt(threshold)
        station_stats = (
            observations.groupby(["Station", "Name", "City"], sort=True)
            .agg(
                observation_count=(metric, "count"),
                mean_value=(metric, "mean"),
                median_value=(metric, "median"),
                minimum_value=(metric, "min"),
                maximum_value=(metric, "max"),
                relative_high_count=("relative_high", "sum"),
                relative_high_frequency=("relative_high", "mean"),
            )
            .reset_index()
        )

        for row in station_stats.itertuples(index=False):
            station_series = observations.loc[
                observations["Station"].eq(row.Station)
            ].sort_values("month_index")
            hot_indices = station_series.loc[
                station_series["relative_high"], "month_index"
            ].astype(int)
            longest_run = current_run = 0
            previous_index: int | None = None
            for index in hot_indices:
                current_run = current_run + 1 if previous_index == index - 1 else 1
                longest_run = max(longest_run, current_run)
                previous_index = index

            elapsed_years = (
                station_series["month_index"].astype(float)
                - float(station_series["month_index"].min())
            ) / 12
            slope = float(
                pd.Series(station_series[metric].to_numpy(dtype=float))
                .pipe(lambda values: _linear_slope(elapsed_years, values))
            )
            station_rows.append(
                {
                    "city": row.City,
                    "station_id": row.Station,
                    "station_name": row.Name,
                    "metric": metric,
                    "observations": int(row.observation_count),
                    "mean_value": float(row.mean_value),
                    "median_value": float(row.median_value),
                    "minimum_value": float(row.minimum_value),
                    "maximum_value": float(row.maximum_value),
                    "relative_high_threshold": threshold,
                    "relative_high_count": int(row.relative_high_count),
                    "relative_high_frequency": float(row.relative_high_frequency),
                    "longest_consecutive_relative_high_run_months": longest_run,
                    "descriptive_linear_slope_per_year": slope,
                    "trend_direction": (
                        "increasing"
                        if slope > 0
                        else "decreasing"
                        if slope < 0
                        else "flat"
                    ),
                    "threshold_basis": (
                        f"pooled {quantile:.0%} quantile across source readings; "
                        "exploratory, not a regulatory standard"
                    ),
                }
            )

        monthly = (
            observations.groupby(["City", "Year", "Month", "observation_period"])
            .agg(
                station_count=("Station", "nunique"),
                relative_high_station_count=("relative_high", "sum"),
            )
            .reset_index()
        )
        monthly["relative_high_frequency"] = (
            monthly["relative_high_station_count"] / monthly["station_count"]
        )
        monthly["metric"] = metric
        monthly["relative_high_threshold"] = threshold
        monthly_rows.extend(monthly.to_dict(orient="records"))

    station_summary = pd.DataFrame(station_rows).sort_values(
        ["metric", "city", "mean_value"], ascending=[True, True, False]
    )
    monthly_summary = pd.DataFrame(monthly_rows).sort_values(
        ["metric", "City", "Year", "Month"]
    )
    return station_summary, monthly_summary


def build_hotspot_behaviour(
    panel: pd.DataFrame, quantile: float = 0.75
) -> pd.DataFrame:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1.")
    required = {"Station", "Name", "City", "Year", "Month", "month_index"}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError("Panel is missing required fields: " + ", ".join(sorted(missing)))

    output: list[pd.DataFrame] = []
    ordered = panel.sort_values(["Station", "month_index"]).copy()
    for metric in ("Day", "Night"):
        observations = ordered.dropna(subset=[metric]).copy()
        if observations.empty:
            continue
        threshold = float(observations[metric].quantile(quantile))
        station_groups = observations.groupby("Station", sort=False)
        observations["relative_high_threshold"] = threshold
        observations["relative_high"] = observations[metric].gt(threshold)
        observations["previous_value"] = station_groups[metric].shift(1)
        observations["previous_month_index"] = station_groups["month_index"].shift(1)
        observations["previous_relative_high"] = (
            station_groups["relative_high"].shift(1).fillna(False).astype(bool)
        )
        consecutive = (
            observations["month_index"] - observations["previous_month_index"]
        ).eq(1)

        observations["consecutive_month"] = consecutive
        cond_gap = ~consecutive
        cond_persist = consecutive & observations["relative_high"] & observations["previous_relative_high"]
        cond_emerge = consecutive & observations["relative_high"] & ~observations["previous_relative_high"]
        cond_reduce = consecutive & ~observations["relative_high"] & observations["previous_relative_high"]

        conditions = [
            cond_gap & observations["relative_high"],
            cond_gap & ~observations["relative_high"],
            cond_persist,
            cond_emerge,
            cond_reduce,
        ]
        choices = [
            "new_or_reappearing_after_gap",
            "no_comparison_after_gap",
            "persistence",
            "emergence",
            "reduction",
        ]
        observations["relative_hotspot_behaviour"] = np.select(
            conditions, choices, default="below_relative_threshold"
        )
        current_rolling = station_groups[metric].transform(
            lambda values: values.rolling(3, min_periods=3).mean()
        )
        previous_rolling = station_groups[metric].transform(
            lambda values: values.shift(1).rolling(3, min_periods=3).mean()
        )
        consecutive_rolling_window = station_groups["month_index"].transform(
            lambda months: months.diff()
            .eq(1)
            .rolling(3, min_periods=3)
            .sum()
            .eq(3)
        )
        observations["rolling_3_month_mean"] = current_rolling.where(
            consecutive_rolling_window
        )
        observations["previous_rolling_3_month_mean"] = previous_rolling.where(
            consecutive_rolling_window
        )
        observations["rolling_change"] = (
            observations["rolling_3_month_mean"]
            - observations["previous_rolling_3_month_mean"]
        )
        observations["intensity_behaviour"] = "insufficient_history"
        compared = observations["rolling_change"].notna()
        observations.loc[compared & observations["rolling_change"].gt(0), "intensity_behaviour"] = "intensification"
        observations.loc[compared & observations["rolling_change"].lt(0), "intensity_behaviour"] = "reduction"
        observations.loc[compared & observations["rolling_change"].eq(0), "intensity_behaviour"] = "stable"
        observations["metric"] = metric
        observations["threshold_basis"] = (
            f"pooled {quantile:.0%} quantile; exploratory source-value rank, "
            "not a regulatory standard"
        )
        output.append(observations)

    if not output:
        raise ValueError("No non-missing Day or Night values are available.")
    return pd.concat(output, ignore_index=True).sort_values(
        ["metric", "Station", "month_index"]
    ).reset_index(drop=True)


def summarize_recurring_hotspot_patterns(
    behaviour: pd.DataFrame,
) -> pd.DataFrame:
    required = {"Station", "Name", "City", "Month", "metric", "relative_high"}
    missing = required.difference(behaviour.columns)
    if missing:
        raise ValueError(
            "Behaviour data is missing required fields: " + ", ".join(sorted(missing))
        )
    return (
        behaviour.groupby(
            ["City", "Station", "Name", "metric", "Month"], sort=True
        )
        .agg(
            observations=("relative_high", "size"),
            relative_high_count=("relative_high", "sum"),
            recurring_high_frequency=("relative_high", "mean"),
        )
        .reset_index()
        .rename(columns={"Month": "calendar_month"})
    )


def _linear_slope(x: pd.Series, y: pd.Series) -> float:
    import numpy as np

    if len(x) < 2:
        return 0.0
    return float(np.polyfit(x.to_numpy(dtype=float), y.to_numpy(dtype=float), 1)[0])
