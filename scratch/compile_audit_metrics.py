"""
Compile all exact tables and metrics for final_scientific_audit.md
"""
import sys
import os
import pandas as pd
import numpy as np

sys.path.insert(0, r"c:\Users\umuga\New folder")
from src.urbannoise.cpcb_forecasting import (
    evaluate_station_forecasts,
    load_cpcb_10min_data
)
from src.urbannoise.cpcb_intelligence import get_monitoring_network_status

DATA_DIR = r"c:\Users\umuga\New folder\data\external\india\cpcb_noise\cleaned"
RAW_DIR = r"c:\Users\umuga\New folder\data\external\india\cpcb_noise\raw"

print("--- 1. RAW MEASUREMENT DISTRIBUTION AUDIT ---")
df_10m = pd.read_csv(os.path.join(DATA_DIR, "cpcb_chennai_10min_unified.csv"))
print("Total rows in unified 10m:", len(df_10m))
print("Valid rows (valid):", (df_10m["quality_flag"] == "valid").sum())
print("Extreme acoustic spikes:", (df_10m["quality_flag"] == "extreme_acoustic_spike").sum())
print("Sensor dropout or zero:", (df_10m["quality_flag"] == "sensor_dropout_or_zero").sum())
print("Negative readings:", (df_10m["noise_value"] < 0).sum())
print("Zero readings:", (df_10m["noise_value"] == 0).sum())
print("Null readings:", df_10m["noise_value"].isna().sum())

# Station-wise raw distributions
stats = []
for sid, group in df_10m.groupby("station_id"):
    sname = group["station_name"].iloc[0]
    metric = group["metric_type"].iloc[0]
    is_leq = group["is_leq_compatible"].iloc[0]
    vals = group["noise_value"].dropna()
    p1, p5, p25, p50, p75, p95, p99, p99_9 = np.percentile(vals, [1, 5, 25, 50, 75, 95, 99, 99.9])
    
    # Raw energetic mean
    raw_energetic = 10.0 * np.log10(np.mean(10.0 ** (vals / 10.0)))
    arith_mean = vals.mean()
    
    # Filtered energetic mean (<120 dBA)
    clean_vals = vals[vals <= 120.0]
    clean_energetic = 10.0 * np.log10(np.mean(10.0 ** (clean_vals / 10.0)))
    
    stats.append({
        "station_id": sid,
        "station_name": sname,
        "metric": metric,
        "is_leq": is_leq,
        "count": len(vals),
        "min": vals.min(),
        "p1": p1, "p5": p5, "p25": p25, "p50": p50, "p75": p75, "p95": p95, "p99": p99, "p99_9": p99_9,
        "max": vals.max(),
        "arith_mean": arith_mean,
        "raw_energetic": raw_energetic,
        "clean_energetic": clean_energetic,
        "spikes_gt_120": (vals > 120.0).sum()
    })

df_dist = pd.DataFrame(stats)
print(df_dist.to_string())

# Network-wide percentiles (all valid rows)
all_vals = df_10m["noise_value"].dropna()
p1, p5, p25, p50, p75, p95, p99, p99_9 = np.percentile(all_vals, [1, 5, 25, 50, 75, 95, 99, 99.9])
print("\nNETWORK ALL ROWS:")
print(f"Min: {all_vals.min():.2f}, P1: {p1:.2f}, P5: {p5:.2f}, P25: {p25:.2f}, P50: {p50:.2f}, P75: {p75:.2f}, P95: {p95:.2f}, P99: {p99:.2f}, P99.9: {p99_9:.2f}, Max: {all_vals.max():.2f}")
print(f"Arithmetic Mean: {all_vals.mean():.2f} dBA")
raw_net_energetic = 10.0 * np.log10(np.mean(10.0 ** (all_vals / 10.0)))
print(f"Raw Network Energetic Mean: {raw_net_energetic:.2f} dBA")

# True Leq subset
leq_subset = df_10m[df_10m["is_leq_compatible"] == True]["noise_value"].dropna()
clean_leq_subset = df_10m[(df_10m["is_leq_compatible"] == True) & (df_10m["noise_value"] <= 120.0)]["noise_value"].dropna()
print(f"Clean Leq Subset Count: {len(clean_leq_subset)}")
print(f"Clean Leq Arithmetic: {clean_leq_subset.mean():.2f} dBA")
clean_net_energetic = 10.0 * np.log10(np.mean(10.0 ** (clean_leq_subset / 10.0)))
print(f"Clean Leq Network Energetic Mean: {clean_net_energetic:.2f} dBA")
print(f"Clean Delta: {clean_net_energetic - clean_leq_subset.mean():.2f} dB")

print("\n--- 2. FORECASTING VALIDATION (ALL STATIONS, 10m & 1h) ---")
STATIONS = [264, 265, 266, 267, 268, 269, 270, 271, 272, 273]
forecast_results = []
for sid in STATIONS:
    for hz in [1, 6]:
        res = evaluate_station_forecasts(station_id=sid, horizon_steps=hz)
        if "error" in res:
            print(f"Station {sid} hz={hz} error: {res['error']}")
            continue
        best_m = res["best_model_name"]
        cal_n = res["calibration_samples"]
        test_n = res["test_samples"]
        ci_w = res["mpiw_dba"]
        test_cov = res["empirical_coverage_pct"]
        early_cov = res["cov_early_pct"]
        late_cov = res["cov_late_pct"]
        delta_cov = round(late_cov - early_cov, 1)
        
        # persistence vs best
        pers_res = next(m for m in res["models_summary"] if "Persistence" in m["model"])
        best_res = next(m for m in res["models_summary"] if m["model"] == best_m)
        
        forecast_results.append({
            "station_id": sid,
            "station_name": res["station_name"],
            "horizon": "10-min" if hz == 1 else "1-hour",
            "horizon_steps": hz,
            "cal_samples": cal_n,
            "test_samples": test_n,
            "best_model": best_m,
            "pers_mae": pers_res["mae"],
            "best_mae": best_res["mae"],
            "best_rmse": best_res["rmse"],
            "conformal_q": res["conformal_quantile_dba"],
            "interval_width": ci_w,
            "test_cov": test_cov,
            "early_cov": early_cov,
            "late_cov": late_cov,
            "delta_cov": delta_cov,
            "is_leq": res["is_leq_compatible"]
        })

df_fc = pd.DataFrame(forecast_results)
print(df_fc[["station_name", "horizon", "cal_samples", "test_samples", "best_model", "best_mae", "interval_width", "test_cov", "delta_cov"]].to_string())

print("\n--- 3. REGULATORY COMPLIANCE TABLE ---")
df_daily = pd.read_csv(os.path.join(DATA_DIR, "cpcb_chennai_daily_unified.csv"))
print("Total daily rows:", len(df_daily))
status_df = get_monitoring_network_status()
print(status_df[["Station ID", "Station Name", "Acoustic Zone", "Day Leq (dBA)", "Night Leq (dBA)", "Day Limit", "Night Limit", "Data Sufficiency Status"]].to_string())
