from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "external" / "india" / "cpcb_noise" / "cleaned" / "cpcb_chennai_10min_unified.csv"


def load_cpcb_10min_data() -> pd.DataFrame:
    """Load the cleaned 10-minute unified dataset with correct typing."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Missing unified 10-minute dataset at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


def evaluate_station_forecasts(
    station_id: int,
    horizon_steps: int = 1,
    nominal_coverage: float = 0.95,
) -> dict[str, Any]:
    """Train, calibrate, and evaluate forecasting models using leak-free chronological partitioning.
    
    Rigorous Scientific Safeguards:
      1. Zero Future Leakage / Zero Backfilling:
         - Past features use information strictly known at or before prediction time t.
         - No .bfill() or future interpolation across boundaries.
      2. Exact Timestamp Target Alignment:
         - For horizon_steps = 1: target timestamp is exactly t + 10 minutes.
         - For horizon_steps = 6: target timestamp is exactly t + 60 minutes.
         - If the exact timestamp is missing, that observation is excluded from evaluation.
      3. Strict 3-Way Chronological Separation:
         - Training Set: First 70% of chronological observations.
         - Calibration Set: Next 15% of observations (model selection & conformal quantile).
         - Test Set: Final 15% of observations (held-out out-of-sample evaluation).
         - Zero test-set leakage: the best model is chosen exclusively on the calibration set.
      4. Split-Conformal Prediction Interval:
         - Absolute residuals on the calibration set determine the non-conformity threshold q_conf.
         - Evaluated on the held-out test set to report actual achieved coverage.
    """
    full_df = load_cpcb_10min_data()
    st_df = full_df.loc[full_df["station_id"] == station_id].sort_values("timestamp").reset_index(drop=True).copy()
    if st_df.empty:
        raise ValueError(f"Station {station_id} not found in CPCB dataset.")
        
    station_name = st_df["display_name"].iloc[0] if "display_name" in st_df.columns else st_df["station_name"].iloc[0]
    zone_category = st_df["zone_category"].iloc[0]
    is_leq_compat = bool(st_df["is_leq_compatible"].iloc[0]) if "is_leq_compatible" in st_df.columns else True
    metric_type = st_df["metric_type"].iloc[0] if "metric_type" in st_df.columns else "LAF_avg"
    
    # Filter physical extremes (keep clean observations, no bfill)
    st_df.loc[(st_df["noise_value"] < 20) | (st_df["noise_value"] > 120), "noise_value"] = np.nan
    st_df["noise_value"] = st_df["noise_value"].ffill(limit=1)
    
    # Build exact timestamp lookup dictionary
    val_map = dict(zip(st_df["timestamp"], st_df["noise_value"]))
    
    # Target via EXACT timestamp matching
    delta_mins = 10 * horizon_steps
    st_df["target_timestamp"] = st_df["timestamp"] + pd.Timedelta(minutes=delta_mins)
    st_df["target"] = st_df["target_timestamp"].map(val_map)
    
    # Information available at current time t:
    val = st_df["noise_value"]
    st_df["current_level"] = val
    st_df["lag_1"] = (st_df["timestamp"] - pd.Timedelta(minutes=10)).map(val_map)
    st_df["lag_2"] = (st_df["timestamp"] - pd.Timedelta(minutes=20)).map(val_map)
    st_df["lag_3"] = (st_df["timestamp"] - pd.Timedelta(minutes=30)).map(val_map)
    st_df["lag_6"] = (st_df["timestamp"] - pd.Timedelta(minutes=60)).map(val_map)
    
    # Rolling statistics strictly from past observations up to current time t
    st_df["rolling_mean_3"] = val.rolling(window=3, min_periods=1).mean()
    st_df["rolling_mean_6"] = val.rolling(window=6, min_periods=1).mean()
    st_df["rolling_std_6"] = val.rolling(window=6, min_periods=1).std().fillna(0)
    st_df["delta_1"] = st_df["current_level"] - st_df["lag_1"]
    
    # Cyclical diurnal time encodings
    hours = st_df["timestamp"].dt.hour + st_df["timestamp"].dt.minute / 60.0
    st_df["hour_sin"] = np.sin(2 * np.pi * hours / 24.0)
    st_df["hour_cos"] = np.cos(2 * np.pi * hours / 24.0)
    st_df["day_of_week"] = st_df["timestamp"].dt.day_of_week
    st_df["is_weekend"] = (st_df["day_of_week"] >= 5).astype(int)
    
    feature_cols = [
        "current_level", "lag_1", "lag_2", "lag_3", "lag_6",
        "rolling_mean_3", "rolling_mean_6", "rolling_std_6", "delta_1",
        "hour_sin", "hour_cos", "is_weekend"
    ]
    
    # Valid mask: drop any row where target or features are missing
    valid_mask = st_df[feature_cols + ["target"]].notna().all(axis=1)
    clean_df = st_df.loc[valid_mask].reset_index(drop=True)
    
    n = len(clean_df)
    if n < 500:
        return {"error": f"Insufficient valid observations for station {station_id} ({n} valid records < 500 required threshold)."}
        
    # Chronological 70% Train / 15% Calibration / 15% Test Split
    n_train = int(n * 0.70)
    n_calib = int(n * 0.15)
    
    train_data = clean_df.iloc[:n_train]
    calib_data = clean_df.iloc[n_train:n_train + n_calib]
    test_data = clean_df.iloc[n_train + n_calib:]
    
    X_train = train_data[feature_cols].values
    y_train = train_data["target"].values
    
    X_calib = calib_data[feature_cols].values
    y_calib = calib_data["target"].values
    
    X_test = test_data[feature_cols].values
    y_test = test_data["target"].values
    test_times = test_data["target_timestamp"].values
    
    # ─── 1. Persistence Baseline: y_hat = y_t (current reading) ───
    y_pred_persist_test = test_data["current_level"].values
    y_pred_persist_calib = calib_data["current_level"].values
    mae_persist = float(np.mean(np.abs(y_test - y_pred_persist_test)))
    rmse_persist = float(np.sqrt(np.mean((y_test - y_pred_persist_test) ** 2)))
    
    # ─── 2. Autoregressive Lag Model ───
    y_pred_ar_test = (
        0.5 * test_data["current_level"].values +
        0.3 * test_data["lag_1"].values +
        0.2 * test_data["lag_2"].values
    )
    y_pred_ar_calib = (
        0.5 * calib_data["current_level"].values +
        0.3 * calib_data["lag_1"].values +
        0.2 * calib_data["lag_2"].values
    )
    mae_ar = float(np.mean(np.abs(y_test - y_pred_ar_test)))
    rmse_ar = float(np.sqrt(np.mean((y_test - y_pred_ar_test) ** 2)))
    
    # ─── 3. Ridge Regression Model (with scaling) ───
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    ridge.fit(X_train, y_train)
    y_pred_ridge_test = ridge.predict(X_test)
    y_pred_ridge_calib = ridge.predict(X_calib)
    mae_ridge = float(np.mean(np.abs(y_test - y_pred_ridge_test)))
    rmse_ridge = float(np.sqrt(np.mean((y_test - y_pred_ridge_test) ** 2)))
    
    # ─── 4. Random Forest Regressor ───
    rf = RandomForestRegressor(n_estimators=35, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf_test = rf.predict(X_test)
    y_pred_rf_calib = rf.predict(X_calib)
    mae_rf = float(np.mean(np.abs(y_test - y_pred_rf_test)))
    rmse_rf = float(np.sqrt(np.mean((y_test - y_pred_rf_test) ** 2)))
    
    # Feature importances from Random Forest
    feat_importances = dict(zip(feature_cols, [round(float(v), 4) for v in rf.feature_importances_]))
    
    # ─── Model Selection on CALIBRATION SET ONLY (Zero Test Set Leakage) ───
    models_summary = [
        {"model": "Persistence Baseline", "calib_mae": float(np.mean(np.abs(y_calib - y_pred_persist_calib))), "mae": round(mae_persist, 2), "rmse": round(rmse_persist, 2), "preds": y_pred_persist_test, "calib_preds": y_pred_persist_calib},
        {"model": "Autoregressive Lag (AR)", "calib_mae": float(np.mean(np.abs(y_calib - y_pred_ar_calib))), "mae": round(mae_ar, 2), "rmse": round(rmse_ar, 2), "preds": y_pred_ar_test, "calib_preds": y_pred_ar_calib},
        {"model": "Ridge Regression", "calib_mae": float(np.mean(np.abs(y_calib - y_pred_ridge_calib))), "mae": round(mae_ridge, 2), "rmse": round(rmse_ridge, 2), "preds": y_pred_ridge_test, "calib_preds": y_pred_ridge_calib},
        {"model": "Random Forest Regressor", "calib_mae": float(np.mean(np.abs(y_calib - y_pred_rf_calib))), "mae": round(mae_rf, 2), "rmse": round(rmse_rf, 2), "preds": y_pred_rf_test, "calib_preds": y_pred_rf_calib},
    ]
    # Pick optimal candidate exclusively on calibration performance
    models_summary.sort(key=lambda m: m["calib_mae"])
    best_model = models_summary[0]
    
    # ─── Split-Conformal Prediction Interval Calibration ───
    # Non-conformity score on held-out calibration set: absolute residual
    calib_scores = np.abs(y_calib - best_model["calib_preds"])
    q_conformal = float(np.quantile(calib_scores, nominal_coverage))
    
    # Evaluate empirical coverage on the held-out test set
    y_pred_best_test = best_model["preds"]
    test_lower = y_pred_best_test - q_conformal
    test_upper = y_pred_best_test + q_conformal
    empirical_coverage = float(np.mean((y_test >= test_lower) & (y_test <= test_upper)) * 100.0)
    mpiw_dba = float(2.0 * q_conformal)
    
    # Temporal stability of coverage: evaluate on first vs second half of test partition
    mid_idx = len(y_test) // 2
    cov_early = float(np.mean((y_test[:mid_idx] >= test_lower[:mid_idx]) & (y_test[:mid_idx] <= test_upper[:mid_idx])) * 100.0)
    cov_late = float(np.mean((y_test[mid_idx:] >= test_lower[mid_idx:]) & (y_test[mid_idx:] <= test_upper[mid_idx:])) * 100.0)
    
    # ─── Forward Prediction From Most Recent Historical Record ───
    latest_row = clean_df.iloc[-1:]
    latest_X = latest_row[feature_cols].values
    latest_time = latest_row["timestamp"].iloc[0]
    forecast_time = latest_time + pd.Timedelta(minutes=10 * horizon_steps)
    
    if best_model["model"] == "Random Forest Regressor":
        latest_pred = float(rf.predict(latest_X)[0])
    elif best_model["model"] == "Ridge Regression":
        latest_pred = float(ridge.predict(latest_X)[0])
    elif best_model["model"] == "Autoregressive Lag (AR)":
        latest_pred = float(
            0.5 * latest_row["current_level"].iloc[0] +
            0.3 * latest_row["lag_1"].iloc[0] +
            0.2 * latest_row["lag_2"].iloc[0]
        )
    else:
        latest_pred = float(latest_row["current_level"].iloc[0])
        
    forecast_lower = round(latest_pred - q_conformal, 2)
    forecast_upper = round(latest_pred + q_conformal, 2)
    
    plot_points = min(120, len(y_test))
    
    return {
        "station_id": station_id,
        "station_name": station_name,
        "zone": zone_category,
        "is_leq_compatible": is_leq_compat,
        "metric_type": metric_type,
        "horizon_steps": horizon_steps,
        "horizon_label": "Next 10-Min" if horizon_steps == 1 else ("Next 1-Hour" if horizon_steps == 6 else "Next 24-Hour"),
        "total_valid_samples": n,
        "train_samples": len(train_data),
        "calibration_samples": len(calib_data),
        "test_samples": len(test_data),
        "test_times": test_times[-plot_points:],
        "y_true": y_test[-plot_points:],
        "y_pred_best": best_model["preds"][-plot_points:],
        "best_test_lower": test_lower[-plot_points:],
        "best_test_upper": test_upper[-plot_points:],
        "best_model_name": best_model["model"],
        "best_mae": best_model["mae"],
        "best_rmse": best_model["rmse"],
        "models_summary": [{k: v for k, v in m.items() if k not in ("preds", "calib_preds")} for m in models_summary],
        "feature_importances": feat_importances,
        "latest_observation_time": latest_time.strftime("%Y-%m-%d %H:%M:%S"),
        "latest_observation_val": round(float(latest_row["current_level"].iloc[0]), 2),
        "forecast_time": forecast_time.strftime("%Y-%m-%d %H:%M:%S"),
        "forecast_value": round(latest_pred, 2),
        "forecast_lower": forecast_lower,
        "forecast_upper": forecast_upper,
        "confidence_lower": forecast_lower,
        "confidence_upper": forecast_upper,
        "conformal_quantile_dba": round(q_conformal, 2),
        "empirical_coverage_pct": round(empirical_coverage, 1),
        "cov_early_pct": round(cov_early, 1),
        "cov_late_pct": round(cov_late, 1),
        "mpiw_dba": round(mpiw_dba, 2),
        "calibration_method": "Split-Conformal Absolute Residual Quantile",
        "nominal_coverage_pct": round(nominal_coverage * 100.0, 1),
        "metric_unit": "dBA (LAF_avg / 10-min Leq)" if is_leq_compat else "dBA (LAF_ins instantaneous)"
    }
