"""Multi-horizon hourly noise level and hotspot forecasting.

One direct model covers all horizons (horizon is a feature). From an issue hour t with
observations up to and including t, the model predicts hourly LAeq at t+h, h in HORIZONS.
Quantile models give an 80% interval and a classifier gives P(hotspot at t+h).

Evaluation is strictly chronological per station: the first TRAIN_FRACTION of each
station's hours trains the models, the remainder tests them. The station hour-of-day
profile feature is computed from the training portion only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (brier_score_loss, f1_score, mean_absolute_error,
                             roc_auc_score)

from .analytics import HOTSPOT_MARGIN_DB
from .config import MODELS_DIR, STATIONS, ZONE_LIMITS, limit_for

HORIZONS = [1, 2, 3, 6, 12, 24]
# Quantiles for the "80% range". Wider-than-nominal quantiles are used because test periods
# drift from training periods; the 5/95 pair gives close to 80% empirical coverage in backtest.
INTERVAL_ALPHAS = (0.05, 0.95)
TRAIN_FRACTION = 0.7
ZONE_CODE = {z: i for i, z in enumerate(ZONE_LIMITS)}
# Level features are expressed relative to the latest hour (lag0), and the regressors
# predict the change from lag0. This makes "no change" (persistence) the model's
# natural starting point and makes it robust to station level shifts.
# Day-of-week and same-hour-last-week were dropped: with ~4 weeks per station they overfit
# (chronological backtest MAE got worse with them).
FEATURES = ["horizon", "lag0", "d_lag1", "d_lag2", "d_mean3", "d_mean24", "slope3",
            "d_yday", "d_profile", "profile_std", "d_limit",
            "target_hour", "target_night", "zone_code"]
MODEL_DIR = MODELS_DIR / "forecaster"
EVAL_DIR = MODEL_DIR / "evaluation"
_BOOSTERS = ("point", "q_lo", "q_hi", "hotspot")

# n_jobs is capped: in shared cloud containers OpenMP can see the host's cores and
# oversubscribe the few CPUs actually available, which makes training crawl.
_LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=100,
                   subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=7,
                   n_jobs=2)


def data_signature(hourly: pd.DataFrame) -> str:
    """Identifies the data a model was trained on (row count + latest hour)."""
    return f"{len(hourly)}|{hourly['hour_start'].max()}"


def _rolling_leq(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Rolling energy average, vectorised (NaN hours are skipped)."""
    power = np.power(10.0, s / 10.0)
    return 10.0 * np.log10(power.rolling(window, min_periods=min_periods).mean())


def _regular_series(hourly: pd.DataFrame, sid: int) -> pd.Series:
    g = hourly[hourly["station_id"] == sid].set_index("hour_start")["laeq"].sort_index()
    return g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))


def station_profiles(hourly: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """Hour-of-day mean and std of LAeq per station."""
    out = {}
    for sid, g in hourly.groupby("station_id"):
        out[sid] = g.groupby("hour")["laeq"].agg(["mean", "std"]).reindex(range(24))
    return out


def build_features(series: pd.Series, sid: int, profile: pd.DataFrame,
                   issue_times: pd.DatetimeIndex | None = None,
                   horizons: list[int] = HORIZONS, with_target: bool = True) -> pd.DataFrame:
    """Feature rows for every (issue time, horizon) pair of one station's hourly series."""
    s = series
    zone = STATIONS[sid].zone
    lagk = {k: s.shift(k) for k in range(3)}
    mean3 = _rolling_leq(s, 3, 2)
    mean24 = _rolling_leq(s, 24, 12)
    base = pd.DataFrame({"lag0": lagk[0], "lag1": lagk[1], "lag2": lagk[2],
                         "mean3": mean3, "mean24": mean24, "slope3": (s - s.shift(2)) / 2})
    if issue_times is None:
        issue_times = s.index[s.notna()]
    base = base.reindex(issue_times)
    base = base[base["lag0"].notna()]
    frames = []
    for h in horizons:
        f = base.copy()
        tgt_time = f.index + pd.Timedelta(hours=h)
        f["issue_time"] = f.index
        f["target_time"] = tgt_time
        f["horizon"] = h
        f["same_hour_yday"] = s.reindex(tgt_time - pd.Timedelta(hours=24)).to_numpy()
        f["same_hour_lastweek"] = s.reindex(tgt_time - pd.Timedelta(hours=168)).to_numpy()
        hrs = tgt_time.hour
        f["profile"] = profile["mean"].reindex(hrs).to_numpy()
        f["profile_std"] = profile["std"].reindex(hrs).to_numpy()
        f["target_hour"] = hrs
        f["target_dow"] = tgt_time.dayofweek
        f["target_weekend"] = (tgt_time.dayofweek >= 5).astype(int)
        f["target_night"] = ((hrs < 6) | (hrs >= 22)).astype(int)
        f["target_limit"] = [limit_for(zone, x) for x in hrs]
        f["zone_code"] = ZONE_CODE[zone]
        for col, src in [("d_lag1", "lag1"), ("d_lag2", "lag2"), ("d_mean3", "mean3"),
                         ("d_mean24", "mean24"), ("d_yday", "same_hour_yday"),
                         ("d_lastweek", "same_hour_lastweek"), ("d_profile", "profile"),
                         ("d_limit", "target_limit")]:
            f[col] = f[src] - f["lag0"]
        if with_target:
            f["target"] = s.reindex(tgt_time).to_numpy()
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out.insert(0, "station_id", sid)
    if with_target:
        out = out[out["target"].notna()].copy()
        out["target_hotspot"] = (out["target"] - out["target_limit"] > HOTSPOT_MARGIN_DB).astype(int)
    return out


def _split_station(hourly: pd.DataFrame, sid: int) -> pd.Timestamp:
    g = hourly.loc[hourly["station_id"] == sid, "hour_start"].sort_values()
    return g.iloc[int(len(g) * TRAIN_FRACTION)]


@dataclass
class ForecastModels:
    """Four LightGBM boosters: change-from-now point and quantile models, hotspot classifier."""
    point: lgb.Booster
    q_lo: lgb.Booster
    q_hi: lgb.Booster
    hotspot: lgb.Booster
    profiles: dict[int, pd.DataFrame]
    trained_on: dict = field(default_factory=dict)

    def predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        X = feats[FEATURES].to_numpy(dtype=float)
        out = feats[["station_id", "issue_time", "target_time", "horizon", "target_limit"]].copy()
        base = feats["lag0"].to_numpy()
        out["pred_laeq"] = base + self.point.predict(X)
        lo, hi = base + self.q_lo.predict(X), base + self.q_hi.predict(X)
        out["pred_lo"] = np.minimum(lo, out["pred_laeq"])
        out["pred_hi"] = np.maximum(hi, out["pred_laeq"])
        out["p_hotspot"] = self.hotspot.predict(X)  # binary booster returns P(class 1)
        out["pred_exceedance"] = out["pred_laeq"] - out["target_limit"]
        return out


def _fit(train: pd.DataFrame, profiles: dict) -> ForecastModels:
    X, y = train[FEATURES], train["target"] - train["lag0"]
    point = lgb.LGBMRegressor(objective="l1", **_LGB_PARAMS).fit(X, y)
    q_lo = lgb.LGBMRegressor(objective="quantile", alpha=INTERVAL_ALPHAS[0], **_LGB_PARAMS).fit(X, y)
    q_hi = lgb.LGBMRegressor(objective="quantile", alpha=INTERVAL_ALPHAS[1], **_LGB_PARAMS).fit(X, y)
    clf = lgb.LGBMClassifier(**_LGB_PARAMS).fit(X, train["target_hotspot"])
    return ForecastModels(point.booster_, q_lo.booster_, q_hi.booster_, clf.booster_, profiles)


def make_dataset(hourly: pd.DataFrame, split: bool = True):
    """Return (train, test, train_profiles) or (all_rows, profiles) if split=False."""
    train_parts, test_parts = [], []
    profiles = {}
    for sid in sorted(hourly["station_id"].unique()):
        s = _regular_series(hourly, sid)
        if split:
            cut = _split_station(hourly, sid)
            hs = hourly[hourly["station_id"] == sid]
            prof = station_profiles(hs[hs["hour_start"] < cut])[sid]
            profiles[sid] = prof
            f = build_features(s, sid, prof)
            # Train rows must have their *target* inside the training period (no leakage).
            train_parts.append(f[f["target_time"] < cut])
            test_parts.append(f[f["issue_time"] >= cut])
        else:
            prof = station_profiles(hourly[hourly["station_id"] == sid])[sid]
            profiles[sid] = prof
            train_parts.append(build_features(s, sid, prof))
    if split:
        return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True), profiles
    return pd.concat(train_parts, ignore_index=True), profiles


def evaluate(hourly: pd.DataFrame) -> dict:
    """Chronological backtest of the model against three baselines, per horizon."""
    train, test, profiles = make_dataset(hourly, split=True)
    models = _fit(train, profiles)
    pred = models.predict(test)
    test = test.reset_index(drop=True)
    pred = pred.reset_index(drop=True)
    y = test["target"]
    baselines = {
        "Persistence (last hour)": test["lag0"],
        "Same hour yesterday": test["same_hour_yday"].fillna(test["lag0"]),
        "Station hourly profile": test["profile"].fillna(test["lag0"]),
        "NoiseScope LightGBM": pred["pred_laeq"],
    }
    rows = []
    for h in HORIZONS:
        m = test["horizon"] == h
        for name, yhat in baselines.items():
            rows.append({"horizon_h": h, "method": name,
                         "MAE_dB": mean_absolute_error(y[m], yhat[m]),
                         "RMSE_dB": float(np.sqrt(np.mean((y[m] - yhat[m]) ** 2)))})
    reg = pd.DataFrame(rows)

    cls_rows = []
    for h in HORIZONS:
        m = test["horizon"] == h
        yt = test.loc[m, "target_hotspot"]
        p = pred.loc[m, "p_hotspot"]
        persist = (test.loc[m, "lag0"] - test.loc[m, "target_limit"] > HOTSPOT_MARGIN_DB).astype(int)
        cls_rows.append({
            "horizon_h": h, "test_rows": int(m.sum()), "hotspot_rate": float(yt.mean()),
            "AUC": roc_auc_score(yt, p) if yt.nunique() > 1 else np.nan,
            "Brier": brier_score_loss(yt, p),
            "F1_model": f1_score(yt, (p >= 0.5).astype(int), zero_division=0),
            "F1_persistence": f1_score(yt, persist, zero_division=0),
        })
    cls = pd.DataFrame(cls_rows)
    inside = (y >= pred["pred_lo"]) & (y <= pred["pred_hi"])
    coverage = pd.DataFrame({"horizon_h": test["horizon"], "inside": inside}).groupby("horizon_h")["inside"].mean()
    per_station = (pd.DataFrame({"station_id": test["station_id"], "horizon": test["horizon"],
                                 "abs_err": (y - pred["pred_laeq"]).abs(),
                                 "abs_err_persist": (y - test["lag0"]).abs()})
                   .groupby(["station_id", "horizon"]).mean().reset_index())
    importance = pd.Series(models.point.feature_importance("gain"), index=FEATURES)
    return {
        "regression": reg, "classification": cls,
        "interval_coverage_80": coverage.rename("coverage").reset_index(),
        "per_station": per_station,
        "feature_importance": (importance / importance.sum()).sort_values(ascending=False).rename("gain_share").rename_axis("feature").reset_index(),
        "n_train": len(train), "n_test": len(test), "signature": data_signature(hourly),
        "test_predictions": pd.concat([test[["station_id", "issue_time", "target_time", "horizon", "target"]],
                                       pred[["pred_laeq", "pred_lo", "pred_hi", "p_hotspot"]]], axis=1),
    }


_EVAL_TABLES = ["regression", "classification", "interval_coverage_80", "per_station",
                "feature_importance", "test_predictions"]


def save_evaluation(ev: dict, path: Path = EVAL_DIR) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for k in _EVAL_TABLES:
        ev[k].to_csv(path / f"{k}.csv", index=False)
    (path / "summary.json").write_text(json.dumps(
        {"n_train": ev["n_train"], "n_test": ev["n_test"], "signature": ev["signature"]}, indent=1))


def load_evaluation(path: Path = EVAL_DIR) -> dict:
    ev = json.loads((path / "summary.json").read_text())
    for k in _EVAL_TABLES:
        ev[k] = pd.read_csv(path / f"{k}.csv")
    tp = ev["test_predictions"]
    tp["issue_time"] = pd.to_datetime(tp["issue_time"])
    tp["target_time"] = pd.to_datetime(tp["target_time"])
    return ev


def train_final(hourly: pd.DataFrame, path: Path | None = MODEL_DIR) -> ForecastModels:
    rows, profiles = make_dataset(hourly, split=False)
    models = _fit(rows, profiles)
    models.trained_on = {"rows": len(rows), "stations": sorted(int(s) for s in hourly["station_id"].unique()),
                         "from": str(hourly["hour_start"].min()), "to": str(hourly["hour_start"].max()),
                         "signature": data_signature(hourly)}
    if path is not None:
        save_models(models, path)
    return models


def save_models(models: ForecastModels, path: Path = MODEL_DIR) -> None:
    """LightGBM text + JSON, so the files load across library versions."""
    path.mkdir(parents=True, exist_ok=True)
    for name in _BOOSTERS:
        (path / f"{name}.txt").write_text(getattr(models, name).model_to_string())
    profiles = {str(sid): {"mean": p["mean"].tolist(), "std": p["std"].tolist()}
                for sid, p in models.profiles.items()}
    (path / "meta.json").write_text(json.dumps({"trained_on": models.trained_on, "profiles": profiles}))


def load_models(path: Path = MODEL_DIR) -> ForecastModels:
    meta = json.loads((path / "meta.json").read_text())
    boosters = {n: lgb.Booster(model_str=(path / f"{n}.txt").read_text()) for n in _BOOSTERS}
    profiles = {int(sid): pd.DataFrame(p, index=range(24)) for sid, p in meta["profiles"].items()}
    return ForecastModels(**boosters, profiles=profiles, trained_on=meta["trained_on"])


def forecast_latest(models: ForecastModels, hourly: pd.DataFrame,
                    horizons: list[int] | None = None) -> pd.DataFrame:
    """Forecast every station from its latest observed hour, for horizons 1..24."""
    horizons = horizons or list(range(1, 25))
    out = []
    for sid in sorted(hourly["station_id"].unique()):
        s = _regular_series(hourly, sid)
        last = s.last_valid_index()
        prof = models.profiles.get(sid)
        if prof is None:
            prof = station_profiles(hourly[hourly["station_id"] == sid])[sid]
        f = build_features(s, sid, prof, issue_times=pd.DatetimeIndex([last]),
                           horizons=horizons, with_target=False)
        if not f.empty:
            out.append(models.predict(f))
    res = pd.concat(out, ignore_index=True)
    res["station"] = res["station_id"].map(lambda x: STATIONS[x].label)
    res["status"] = np.select([res["pred_exceedance"] <= 0, res["pred_exceedance"] <= HOTSPOT_MARGIN_DB],
                              ["Compliant", "Exceeding"], "Hotspot")
    return res
