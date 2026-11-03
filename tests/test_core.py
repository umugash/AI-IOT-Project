import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from noisescope import analytics, audio, cpcb, edge, forecasting, store  # noqa: E402
from noisescope.acoustics import leq  # noqa: E402
from noisescope.config import CPCB_RAW_DIR, limit_for  # noqa: E402


def test_leq_is_energy_average():
    assert leq([60, 60]) == pytest.approx(60)
    assert leq([60, 70]) == pytest.approx(10 * np.log10((10**6 + 10**7) / 2))
    assert np.isnan(leq([]))


def test_limits_day_night():
    assert limit_for("Silence", 10) == 50 and limit_for("Silence", 23) == 40
    assert limit_for("Industrial", 5) == 70 and limit_for("Industrial", 6) == 75


def _write_export(path, header, rows):
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def test_parse_export_with_and_without_average(tmp_path):
    hdr = '"Date/Time","3. LAF #1 Noise Level (Ins.)dBA","4. LAF #1 Noise Level (Avg.)dBA","5. LAF #1 Noise Level (Max)dBA","6. LAF #1 Noise Level (Min)dBA"'
    f = tmp_path / "chennai_x_267_jan2026.csv"
    _write_export(f, hdr, ["01/02/2026 00:10:00,60,61,70,50,", "01/02/2026 00:20:00,62,130,70,-1,"])
    df = cpcb.parse_export_csv(f)
    assert df["station_id"].unique().tolist() == [267]
    assert df["ts"].iloc[0] == pd.Timestamp("2026-01-02 00:10:00")  # month/day/year
    assert df["metric"].iloc[0] == "LAF_avg" and df["laeq"].iloc[0] == 61
    assert df["quality"].tolist() == ["valid", "above_range_spike"]
    assert np.isnan(df["lmin"].iloc[1])

    hdr2 = '"Date/Time","3. LAF #1 Noise Level (Ins.)dBA","4. LAF #1 Noise Level (Max)dBA","5. LAF #1 Noise Level (Min)dBA","6. LCF #1 Noise Level (Ins.)dBC"'
    f2 = tmp_path / "chennai_y_268_may2025.csv"
    _write_export(f2, hdr2, ["05/01/2025 00:10:00,55,70,45,60,"])
    d2 = cpcb.parse_export_csv(f2)
    assert d2["metric"].iloc[0] == "LAF_ins" and d2["laeq"].iloc[0] == 55 and d2["lmax"].iloc[0] == 70


def test_bundled_exports_load():
    df = cpcb.load_all_exports(CPCB_RAW_DIR)
    assert df["station_id"].nunique() == 10
    assert not df.duplicated(["station_id", "ts"]).any()


def test_flatline_detection():
    ts = pd.date_range("2026-01-01", periods=60, freq="10min")
    vals = np.r_[np.random.default_rng(0).normal(65, 3, 20), np.full(40, 62.6)]
    df = pd.DataFrame({"station_id": 265, "ts": ts, "laeq": vals, "quality": "valid"})
    out = cpcb.flag_flatlines(df)
    assert (out["quality"].iloc[35:] == "flatline_suspect").all()
    assert (out["quality"].iloc[:5] == "valid").all()


def test_hourly_levels_end_labelled_and_status():
    ts = pd.date_range("2026-01-05 10:10", periods=6, freq="10min")  # covers 10:00-11:00
    df = pd.DataFrame({"station_id": 264, "ts": ts, "laeq": 65.0, "lmax": 70.0, "quality": "valid"})
    h = analytics.hourly_levels(df)
    assert len(h) == 1 and h["hour"].iloc[0] == 10
    assert h["limit"].iloc[0] == 50 and h["exceedance"].iloc[0] == pytest.approx(15)
    assert h["status"].iloc[0] == "Hotspot"
    assert analytics.status_for(-1) == "Compliant" and analytics.status_for(5) == "Exceeding"


def test_parse_channels_xml_layouts():
    attr = '<datos><canal id="3" alias="dBA" valor="54.8" fecha="1793622600"/><canal id="7" alias="dBC" valor="51.2" fecha="1793622600"/></datos>'
    ch = cpcb.parse_channels_xml(attr)
    assert ch["3"]["value"] == 54.8
    assert ch["3"]["observed_ist"] == datetime(2026, 11, 2, 12, 30)
    child = "<r><canal><id>3</id><valor>60.1</valor><fecha>1793622600</fecha></canal></r>"
    assert cpcb.parse_channels_xml(child)["3"]["value"] == 60.1
    rec = cpcb.channels_to_record(272, ch, "2026-11-02T07:08:00+00:00")
    assert rec["lins"] == 54.8 and rec["observed_ts"] == "2026-11-02 12:30:00"


def test_store_roundtrip(tmp_path):
    db = tmp_path / "t.db"
    df = pd.DataFrame({"station_id": [264], "ts": [pd.Timestamp("2026-01-01 00:10")], "laeq": [60.0],
                       "lmax": [np.nan], "lmin": [np.nan], "lins": [59.0], "metric": ["LAF_avg"],
                       "quality": ["valid"], "source": ["test"]})
    assert store.upsert_readings(df, db) == 1
    assert store.upsert_readings(df, db) == 1  # idempotent
    out = store.load_readings(db)
    assert len(out) == 1 and out["laeq"].iloc[0] == 60.0


def _synthetic_hourly(days=30):
    rows = []
    rng = np.random.default_rng(1)
    for sid, base in [(264, 62), (267, 80)]:
        for t in pd.date_range("2026-01-01", periods=24 * days, freq="h"):
            lvl = base + 6 * np.sin((t.hour - 6) / 24 * 2 * np.pi) + rng.normal(0, 1.5)
            rows.append({"station_id": sid, "ts": t + pd.Timedelta(minutes=10), "laeq": lvl,
                         "lmax": np.nan, "quality": "valid"})
    df = pd.DataFrame(rows)
    # hourly_levels needs >= 3 readings per hour: replicate each hourly value 6 times
    df = pd.concat([df.assign(ts=df["ts"] + pd.Timedelta(minutes=10 * k)) for k in range(6)])
    return analytics.hourly_levels(df)


def test_forecast_has_no_leakage_and_runs():
    h = _synthetic_hourly()
    train, test, _ = forecasting.make_dataset(h, split=True)
    for sid in h["station_id"].unique():
        cut = forecasting._split_station(h, sid)
        assert (train.loc[train["station_id"] == sid, "target_time"] < cut).all()
        assert (test.loc[test["station_id"] == sid, "issue_time"] >= cut).all()
    ev = forecasting.evaluate(h)
    reg = ev["regression"]
    model_mae = reg[(reg["method"] == "NoiseScope LightGBM") & (reg["horizon_h"] == 6)]["MAE_dB"].iloc[0]
    persist = reg[(reg["method"] == "Persistence (last hour)") & (reg["horizon_h"] == 6)]["MAE_dB"].iloc[0]
    assert model_mae < persist  # a daily cycle should be learnable
    models = forecasting._fit(*forecasting.make_dataset(h, split=False))
    fc = forecasting.forecast_latest(models, h)
    assert set(fc["horizon"]) == set(range(1, 25)) and fc["p_hotspot"].between(0, 1).all()


def test_audio_features_fixed_length():
    sr = audio.SR
    t = np.arange(sr * 2) / sr
    a = audio.extract_features(np.sin(2 * np.pi * 440 * t).astype(np.float32))
    b = audio.extract_features(np.random.default_rng(0).normal(0, 0.1, sr * 5).astype(np.float32))
    assert a.shape == b.shape and np.isfinite(a).all()


def test_a_weighted_level_reference():
    sr = audio.SR
    t = np.arange(sr * 2) / sr
    full = audio.a_weighted_level_dbfs(np.sin(2 * np.pi * 1000 * t).astype(np.float32))
    half = audio.a_weighted_level_dbfs(0.5 * np.sin(2 * np.pi * 1000 * t).astype(np.float32))
    assert full - half == pytest.approx(6.02, abs=0.1)


@pytest.mark.skipif(not audio.MODEL_PATH.exists(), reason="classifier not trained")
def test_edge_node_events_and_mix():
    clf = audio.load_classifier()
    y = np.random.default_rng(0).normal(0, 0.05, audio.SR * 13).astype(np.float32)
    node = edge.EdgeNode("test", 13.0, 80.2, calibration_db=90.0)
    ev = node.events_from_signal(y, datetime(2026, 1, 1, 8), "synthetic", clf)
    assert len(ev) == 3 and all(e["calibrated"] for e in ev)
    df = pd.DataFrame(ev).assign(ts=lambda d: pd.to_datetime(d["ts"]))
    mix = edge.source_mix(df)
    assert mix["energy_share"].sum() == pytest.approx(1.0)
