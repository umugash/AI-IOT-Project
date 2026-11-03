from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


NUMERIC_FEATURES = (
    "Year",
    "month_sin",
    "month_cos",
    "Day_lag_1",
    "Day_lag_2",
    "Day_lag_3",
    "Day_lag_12",
    "Day_rolling_mean_3",
    "Day_rolling_mean_6",
    "Night_lag_1",
    "Night_lag_2",
    "Night_lag_3",
    "Night_lag_12",
    "Night_rolling_mean_3",
    "Night_rolling_mean_6",
)


def chronological_split(
    data: pd.DataFrame,
    validation_year: int = 2016,
    test_start_year: int = 2017,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = data.loc[data["Year"] < validation_year]
    validation = data.loc[data["Year"].eq(validation_year)]
    test = data.loc[data["Year"].ge(test_start_year)]
    if train.empty or validation.empty or test.empty:
        raise ValueError("Chronological train/validation/test partitions must be non-empty.")
    return train.copy(), validation.copy(), test.copy()


def _metrics(actual: pd.Series, predicted: np.ndarray) -> tuple[float, float]:
    errors = actual.to_numpy(dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.abs(errors).mean()), float(np.sqrt(np.square(errors).mean()))


def _learned_model(kind: str) -> object:
    categorical = ["Station"]
    numeric = ["Year", "month_sin", "month_cos"]
    numeric += [feature for feature in NUMERIC_FEATURES if feature.startswith(("Day_", "Night_"))]
    preprocessor = ColumnTransformer(
        [
            (
                "station",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    if kind == "ridge":
        estimator = Ridge(alpha=10.0)
    elif kind == "random_forest":
        estimator = RandomForestRegressor(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=1,
        )
    else:
        raise ValueError(f"Unsupported learned model: {kind}")
    return make_pipeline(preprocessor, estimator)


def _predict_rule(
    history: pd.DataFrame, targets: pd.DataFrame, metric: str, method: str
) -> np.ndarray:
    if method == "persistence":
        return targets[f"{metric}_lag_1"].to_numpy(dtype=float)
    if method == "seasonal_naive_12":
        return targets[f"{metric}_lag_12"].to_numpy(dtype=float)
    if method == "moving_average_3":
        return targets[f"{metric}_rolling_mean_3"].to_numpy(dtype=float)
    raise ValueError(f"Unsupported forecast rule: {method}")


def _one_step_backtest(
    eligible: pd.DataFrame,
    metric: str,
    methods: tuple[str, ...],
    fit: Callable[[pd.DataFrame, pd.DataFrame, str], np.ndarray],
) -> pd.DataFrame:
    predictions: list[pd.DataFrame] = []
    for month_index in sorted(eligible.loc[eligible["Year"].ge(2017), "month_index"].unique()):
        targets = eligible.loc[eligible["month_index"].eq(month_index)].copy()
        history = eligible.loc[eligible["month_index"].lt(month_index)]
        if history.empty:
            continue
        for method in methods:
            if method == "persistence":
                predicted = _predict_rule(history, targets, metric, method)
            elif method == "seasonal_naive_12":
                predicted = _predict_rule(history, targets, metric, method)
            elif method == "moving_average_3":
                predicted = _predict_rule(history, targets, metric, method)
            else:
                predicted = fit(history, targets, method)
            predictions.append(
                pd.DataFrame(
                    {
                        "Station": targets["Station"].to_numpy(),
                        "Name": targets["Name"].to_numpy(),
                        "City": targets["City"].to_numpy(),
                        "Year": targets["Year"].to_numpy(),
                        "Month": targets["Month"].to_numpy(),
                        "observation_period": targets["observation_period"].to_numpy(),
                        "metric": metric,
                        "method": method,
                        "actual": targets[metric].to_numpy(dtype=float),
                        "predicted": predicted,
                    }
                )
            )
    if not predictions:
        raise ValueError("No one-step-ahead backtest predictions were generated.")
    return pd.concat(predictions, ignore_index=True)


def run_forecasting_backtest(
    featured: pd.DataFrame,
    validation_year: int = 2016,
    test_start_year: int = 2017,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_features = [
        "Station",
        "Name",
        "City",
        "Year",
        "Month",
        "month_index",
        "observation_period",
        "Day",
        "Night",
        "month_sin",
        "month_cos",
        *NUMERIC_FEATURES,
    ]
    missing_columns = sorted(set(required_features).difference(featured.columns))
    if missing_columns:
        raise ValueError("Featured panel is missing columns: " + ", ".join(missing_columns))

    metric_predictions: list[pd.DataFrame] = []
    split_rows: list[dict[str, object]] = []
    methods = (
        "persistence",
        "seasonal_naive_12",
        "moving_average_3",
        "ridge_pooled",
        "ridge_station",
        "random_forest_pooled",
    )

    for metric in ("Day", "Night"):
        feature_columns = list(NUMERIC_FEATURES)
        eligible = featured.dropna(subset=[metric, *feature_columns]).copy()
        train, validation, test = chronological_split(
            eligible, validation_year, test_start_year
        )
        split_rows.extend(
            [
                {"metric": metric, "partition": "train", "start_year": int(train["Year"].min()), "end_year": int(train["Year"].max()), "rows": len(train), "stations": train["Station"].nunique()},
                {"metric": metric, "partition": "validation", "start_year": int(validation["Year"].min()), "end_year": int(validation["Year"].max()), "rows": len(validation), "stations": validation["Station"].nunique()},
                {"metric": metric, "partition": "test", "start_year": int(test["Year"].min()), "end_year": int(test["Year"].max()), "rows": len(test), "stations": test["Station"].nunique()},
            ]
        )

        def fit_for_origin(
            history: pd.DataFrame, target: pd.DataFrame, method: str
        ) -> np.ndarray:
            x_train = history[["Station", *NUMERIC_FEATURES]]
            y_train = history[metric].astype(float)
            x_target = target[["Station", *NUMERIC_FEATURES]]
            if method == "ridge_pooled":
                model = _learned_model("ridge")
                model.fit(x_train, y_train)
                return np.asarray(model.predict(x_target), dtype=float)
            if method == "random_forest_pooled":
                model = _learned_model("random_forest")
                model.fit(x_train, y_train)
                return np.asarray(model.predict(x_target), dtype=float)
            if method == "ridge_station":
                output = np.empty(len(target), dtype=float)
                for station_id, indices in target.groupby("Station").groups.items():
                    train_station = history.loc[history["Station"].eq(station_id)]
                    if train_station.empty:
                        raise ValueError(
                            f"No earlier training history for station {station_id}."
                        )
                    station_model = Ridge(alpha=10.0)
                    station_model.fit(
                        train_station[["month_index", *feature_columns]],
                        train_station[metric].astype(float),
                    )
                    positions = target.index.get_indexer(indices)
                    output[positions] = station_model.predict(
                        target.loc[indices, ["month_index", *feature_columns]]
                    )
                return output
            raise ValueError(f"Unsupported backtest model: {method}")

        predictions = _one_step_backtest(
            eligible,
            metric,
            methods,
            fit_for_origin,
        )
        metric_predictions.append(predictions)

    forecasts = pd.concat(metric_predictions, ignore_index=True)
    forecasts["error"] = forecasts["actual"] - forecasts["predicted"]
    forecasts["absolute_error"] = forecasts["error"].abs()
    forecasts["squared_error"] = forecasts["error"] ** 2
    aggregate = (
        forecasts.groupby(["metric", "method"], sort=True)
        .agg(
            observations=("actual", "size"),
            mae=("absolute_error", "mean"),
            rmse=("squared_error", lambda values: float(np.sqrt(values.mean()))),
        )
        .reset_index()
    )
    per_station = (
        forecasts.groupby(["metric", "method", "Station", "Name"], sort=True)
        .agg(
            observations=("actual", "size"),
            mae=("absolute_error", "mean"),
            rmse=("squared_error", lambda values: float(np.sqrt(values.mean()))),
        )
        .reset_index()
    )
    splits = pd.DataFrame(split_rows)
    return forecasts, aggregate, per_station, splits


def compute_feature_importances(featured: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in ("Day", "Night"):
        eligible = featured.dropna(subset=[metric, "Year", "Station", *NUMERIC_FEATURES]).copy()
        if eligible.empty:
            continue
        x_features = ["Station", *NUMERIC_FEATURES]
        x_train = eligible[x_features]
        y_train = eligible[metric].astype(float)

        # Random Forest importances
        rf = _learned_model("random_forest")
        rf.fit(x_train, y_train)
        ohe = rf.named_steps["columntransformer"].named_transformers_["station"]
        feature_names = (
            list(ohe.get_feature_names_out(["Station"]))
            + list(rf.named_steps["columntransformer"].transformers_[1][2])
        )
        rf_importances = rf.named_steps["randomforestregressor"].feature_importances_
        for feat, imp in zip(feature_names, rf_importances):
            rows.append({
                "metric": metric,
                "model": "Random Forest",
                "feature": feat,
                "importance": round(float(imp), 4),
                "coefficient": np.nan,
            })

        # Ridge Regression coefficients
        ridge = _learned_model("ridge")
        ridge.fit(x_train, y_train)
        coefs = ridge.named_steps["ridge"].coef_
        abs_coefs = np.abs(coefs)
        total_abs = abs_coefs.sum()
        rel_imp = abs_coefs / total_abs if total_abs > 0 else abs_coefs
        for feat, imp, coef in zip(feature_names, rel_imp, coefs):
            rows.append({
                "metric": metric,
                "model": "Ridge Regression",
                "feature": feat,
                "importance": round(float(imp), 4),
                "coefficient": round(float(coef), 4),
            })

    result = pd.DataFrame(rows)
    if not result.empty:
        result["rank"] = result.groupby(["metric", "model"])["importance"].rank(
            ascending=False, method="min"
        ).astype(int)
    return result
