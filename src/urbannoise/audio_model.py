from __future__ import annotations

from pathlib import Path
from typing import Sequence

import librosa
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset


SAMPLE_RATE = 16_000
CLIP_SECONDS = 5.0
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256
EXPECTED_FRAMES = int(np.ceil(SAMPLE_RATE * CLIP_SECONDS / HOP_LENGTH))


def load_log_mel(path: str | Path) -> np.ndarray:
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio recording not found: {audio_path}")
    signal, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=CLIP_SECONDS,
    )
    if signal.size == 0:
        raise ValueError(f"Audio recording is empty: {audio_path}")
    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    if log_mel.shape[1] < EXPECTED_FRAMES:
        log_mel = np.pad(
            log_mel,
            ((0, 0), (0, EXPECTED_FRAMES - log_mel.shape[1])),
            mode="constant",
        )
    else:
        log_mel = log_mel[:, :EXPECTED_FRAMES]
    return log_mel


def extract_mfcc_summary(path: str | Path) -> np.ndarray:
    audio_path = Path(path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio recording not found: {audio_path}")
    signal, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True,
        duration=CLIP_SECONDS,
    )
    if signal.size == 0:
        raise ValueError(f"Audio recording is empty: {audio_path}")
    coefficients = librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=20,
    )
    return np.concatenate(
        [coefficients.mean(axis=1), coefficients.std(axis=1)]
    ).astype(np.float32)


class UrbanAudioDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        metadata: pd.DataFrame,
        audio_directory: str | Path,
        labels: Sequence[str],
    ) -> None:
        self.metadata = metadata.reset_index(drop=True)
        self.audio_directory = Path(audio_directory)
        self.label_to_index = {label: index for index, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.metadata.iloc[index]
        filename = Path(str(row["filename"])).name
        if filename != str(row["filename"]):
            raise ValueError(f"Unsafe audio filename in dataset: {row['filename']}")
        features = load_log_mel(self.audio_directory / filename)
        label = self.label_to_index[str(row["class"])]
        return (
            torch.from_numpy(features).unsqueeze(0),
            torch.tensor(label, dtype=torch.long),
        )


class UrbanNoiseCNN(nn.Module):
    def __init__(self, number_of_classes: int) -> None:
        super().__init__()
        if number_of_classes < 2:
            raise ValueError("At least two classes are required for classification.")
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(64, number_of_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))
