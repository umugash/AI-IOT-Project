"""Urban sound source classification and acoustic level estimation.

Training data: ESC-50 (Piczak 2015, CC BY-NC 3.0), 2,000 labelled 5-second clips. Its
fine-grained classes are grouped into the urban source categories of this project.
Features: log-mel / MFCC / spectral statistics. Model: gradient-boosted trees.
Evaluation: ESC-50's official 5 folds (clips from the same recording share a fold).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import json

import numpy as np
import pandas as pd

from .config import MODELS_DIR

SR = 22050
CLIP_SECONDS = 5.0
MODEL_PATH = MODELS_DIR / "sound_source_classifier.txt"

# ESC-50 class -> urban source category. ESC-50 classes not listed are not used.
URBAN_CATEGORIES: dict[str, list[str]] = {
    "Vehicle horn": ["car_horn"],
    "Traffic & engines": ["engine", "train", "airplane", "helicopter"],
    "Emergency siren": ["siren"],
    "Construction & machinery": ["chainsaw", "hand_saw", "vacuum_cleaner"],
    "Human activity": ["crying_baby", "laughing", "coughing", "footsteps", "clapping", "sneezing"],
    "Bells & alarms": ["church_bells", "clock_alarm"],
    "Animals": ["dog", "rooster", "crow", "cat", "hen", "cow"],
    "Nature & weather": ["rain", "wind", "thunderstorm", "sea_waves", "crickets", "chirping_birds"],
}
ESC_TO_CATEGORY = {esc: cat for cat, escs in URBAN_CATEGORIES.items() for esc in escs}
CATEGORY_COLORS = {
    "Vehicle horn": "#e45756", "Traffic & engines": "#f58518", "Emergency siren": "#b279a2",
    "Construction & machinery": "#9d755d", "Human activity": "#4c78a8", "Bells & alarms": "#eeca3b",
    "Animals": "#72b7b2", "Nature & weather": "#54a24b",
}
# Categories treated as man-made urban noise sources for hotspot attribution.
ANTHROPOGENIC = {"Vehicle horn", "Traffic & engines", "Emergency siren",
                 "Construction & machinery", "Human activity", "Bells & alarms"}


def load_audio(path_or_buffer, sr: int = SR) -> np.ndarray:
    import librosa
    y, _ = librosa.load(path_or_buffer, sr=sr, mono=True)
    return y.astype(np.float32)


def level_dbfs(y: np.ndarray) -> float:
    """Equivalent level of a signal in dB relative to digital full scale."""
    rms = float(np.sqrt(np.mean(np.square(y, dtype=np.float64)))) if y.size else 0.0
    return 20.0 * np.log10(max(rms, 1e-10))


def a_weighted_level_dbfs(y: np.ndarray, sr: int = SR) -> float:
    """A-weighted equivalent level (dBFS) computed in the frequency domain."""
    import librosa
    if y.size == 0:
        return float("nan")
    n_fft = 2048
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    weights = librosa.A_weighting(np.maximum(freqs, 1.0))
    power = (S * (10.0 ** (weights / 10.0))[:, None]).sum(axis=0)
    # Normalise so that a full-scale 1 kHz sine reads ~0 dBFS (Parseval for Hann window).
    norm = (n_fft * np.sum(np.hanning(n_fft) ** 2) / n_fft) * n_fft / 2.0
    mean_power = float(np.mean(power)) / norm
    return 10.0 * np.log10(max(mean_power, 1e-20))


def extract_features(y: np.ndarray, sr: int = SR) -> np.ndarray:
    """Fixed-length feature vector for a clip (~280 values)."""
    import librosa
    if y.size < sr // 4:
        y = np.pad(y, (0, sr // 4 - y.size))
    peak = np.max(np.abs(y))
    if peak > 0:
        y = y / peak  # classification should not depend on recording gain
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=512), ref=np.max)
    mfcc = librosa.feature.mfcc(S=mel, n_mfcc=20)
    dmfcc = librosa.feature.delta(mfcc)
    S = np.abs(librosa.stft(y, hop_length=512))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
    flatness = librosa.feature.spectral_flatness(S=S)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y, hop_length=512)
    rms = librosa.feature.rms(S=S)
    onset_env = librosa.onset.onset_strength(S=mel, sr=sr)
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)

    def stats(m):
        return np.concatenate([m.mean(axis=1), m.std(axis=1)])

    rms_db = 20 * np.log10(np.maximum(rms[0], 1e-6))
    extra = np.array([
        onset_env.mean(), onset_env.std(), np.percentile(onset_env, 90),
        rms_db.std(), np.percentile(rms_db, 90) - np.percentile(rms_db, 10),
        (rms_db > rms_db.max() - 10).mean(),  # fraction of loud frames
    ])
    return np.concatenate([
        stats(mel), stats(mfcc), stats(dmfcc),
        stats(np.vstack([centroid, bandwidth, rolloff, flatness, zcr])),
        stats(contrast), stats(chroma), extra,
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Training (ESC-50)
# ---------------------------------------------------------------------------

def esc50_table(esc_dir: Path) -> pd.DataFrame:
    meta = pd.read_csv(Path(esc_dir) / "meta" / "esc50.csv")
    meta = meta[meta["category"].isin(ESC_TO_CATEGORY)].copy()
    meta["urban_category"] = meta["category"].map(ESC_TO_CATEGORY)
    meta["path"] = meta["filename"].map(lambda f: str(Path(esc_dir) / "audio" / f))
    return meta.reset_index(drop=True)


def _feat_from_path(path: str) -> np.ndarray:
    return extract_features(load_audio(path))


def featurize_paths(paths: list[str], n_jobs: int = -1) -> np.ndarray:
    from joblib import Parallel, delayed
    return np.vstack(Parallel(n_jobs=n_jobs)(delayed(_feat_from_path)(p) for p in paths))


# Street-like backgrounds mixed into augmented copies (engine, rain, wind, footsteps).
_AUG_BACKGROUND = ["engine", "rain", "wind", "footsteps"]


def _augmented_features(path: str, bg_path: str, seed: int) -> np.ndarray:
    """Pitch shift +-2 semitones, random circular time shift, background mix at 5-20 dB SNR."""
    import warnings
    import librosa
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y = librosa.effects.pitch_shift(load_audio(path), sr=SR, n_steps=rng.uniform(-2, 2))
    y = np.roll(y, rng.integers(0, len(y)))
    bg = load_audio(bg_path)[: len(y)]
    bg = np.pad(bg, (0, len(y) - len(bg)))
    snr_db = rng.uniform(5, 20)
    gain = np.sqrt((np.mean(y ** 2) + 1e-12) / ((np.mean(bg ** 2) + 1e-12) * 10 ** (snr_db / 10)))
    return extract_features(y + gain * bg)


def augment_paths(meta: pd.DataFrame, n_jobs: int = -1) -> np.ndarray:
    from joblib import Parallel, delayed
    bg = meta.loc[meta["category"].isin(_AUG_BACKGROUND), "path"].tolist()
    rng = np.random.default_rng(42)
    picks = rng.integers(0, len(bg), size=len(meta))
    return np.vstack(Parallel(n_jobs=n_jobs)(
        delayed(_augmented_features)(p, bg[k], i) for i, (p, k) in enumerate(zip(meta["path"], picks))))


@dataclass
class SoundClassifier:
    booster: object          # lightgbm.Booster
    classes: list[str]
    report: dict

    def predict_proba_features(self, feats: np.ndarray) -> np.ndarray:
        p = self.booster.predict(np.atleast_2d(feats), num_threads=1)
        return np.atleast_2d(p)

    def predict_proba(self, y: np.ndarray, sr: int = SR) -> pd.Series:
        p = self.predict_proba_features(extract_features(y, sr))[0]
        return pd.Series(p, index=self.classes).sort_values(ascending=False)


def _make_model():
    import lightgbm as lgb
    return lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=15,
                              min_child_samples=10, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.5, class_weight="balanced",
                              verbose=-1, random_state=7)


def cross_validate(X: np.ndarray, meta: pd.DataFrame, X_aug: np.ndarray | None = None) -> dict:
    """ESC-50 official 5-fold cross-validation on the urban categories.

    Augmented copies (if given) are added to the training folds only; scoring always uses
    the original, unaugmented held-out clips.
    """
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 classification_report, confusion_matrix, f1_score)
    y = meta["urban_category"].to_numpy()
    classes = sorted(URBAN_CATEGORIES)
    preds = np.empty_like(y)
    fold_acc = []
    for fold in sorted(meta["fold"].unique()):
        te = meta["fold"].to_numpy() == fold
        Xtr, ytr = X[~te], y[~te]
        if X_aug is not None:
            Xtr, ytr = np.vstack([Xtr, X_aug[~te]]), np.concatenate([ytr, ytr])
        m = _make_model().fit(Xtr, ytr)
        preds[te] = m.predict(X[te])
        fold_acc.append(accuracy_score(y[te], preds[te]))
    return {
        "accuracy": accuracy_score(y, preds),
        "balanced_accuracy": balanced_accuracy_score(y, preds),
        "macro_f1": f1_score(y, preds, average="macro"),
        "fold_accuracy": fold_acc,
        "per_class": classification_report(y, preds, labels=classes, output_dict=True, zero_division=0),
        "confusion": confusion_matrix(y, preds, labels=classes).tolist(),
        "classes": classes,
        "n_clips": int(len(y)),
        "augmented": X_aug is not None,
    }


def train(esc_dir: Path, path: Path = MODEL_PATH) -> SoundClassifier:
    meta = esc50_table(esc_dir)
    X = featurize_paths(meta["path"].tolist())
    X_aug = augment_paths(meta)
    report = cross_validate(X, meta, X_aug)
    y = meta["urban_category"].to_numpy()
    model = _make_model().fit(np.vstack([X, X_aug]), np.concatenate([y, y]))
    clf = SoundClassifier(booster=model.booster_, classes=[str(c) for c in model.classes_], report=report)
    save_classifier(clf, path)
    return clf


def save_classifier(clf: SoundClassifier, path: Path = MODEL_PATH) -> None:
    """Stored as LightGBM text + JSON so it loads across library versions."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".txt").write_text(clf.booster.model_to_string())
    path.with_suffix(".json").write_text(json.dumps({"classes": clf.classes, "report": clf.report}, indent=1))


def load_classifier(path: Path = MODEL_PATH) -> SoundClassifier:
    import lightgbm as lgb
    path = Path(path)
    booster = lgb.Booster(model_str=path.with_suffix(".txt").read_text())
    meta = json.loads(path.with_suffix(".json").read_text())
    return SoundClassifier(booster=booster, classes=meta["classes"], report=meta["report"])


# ---------------------------------------------------------------------------
# Windowed analysis (used by the edge node and the dashboard)
# ---------------------------------------------------------------------------

def analyse_signal(y: np.ndarray, clf: SoundClassifier, sr: int = SR, window_s: float = CLIP_SECONDS,
                   calibration_db: float | None = None, min_level_dbfs: float = -70.0) -> pd.DataFrame:
    """Split a signal into windows; per window return level and predicted source."""
    win = int(window_s * sr)
    rows = []
    starts = range(0, max(len(y) - win // 2, 1), win)
    for i, start in enumerate(starts):
        seg = y[start:start + win]
        if seg.size < sr // 2:
            continue
        lvl = a_weighted_level_dbfs(seg, sr)
        row = {"offset_s": start / sr, "duration_s": seg.size / sr, "level_dbfs": lvl,
               "level_db": lvl + calibration_db if calibration_db is not None else lvl,
               "calibrated": calibration_db is not None}
        if lvl < min_level_dbfs:
            row.update(source_class="Quiet / background", confidence=np.nan, top3=[])
        else:
            p = clf.predict_proba(seg, sr)
            row.update(source_class=p.index[0], confidence=float(p.iloc[0]),
                       top3=[[k, round(float(v), 3)] for k, v in p.head(3).items()])
        rows.append(row)
    return pd.DataFrame(rows)
