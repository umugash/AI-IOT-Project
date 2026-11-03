from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import DataLoader, Subset

from .audio_model import (
    UrbanAudioDataset,
    UrbanNoiseCNN,
    extract_mfcc_summary,
)
from .dataset import DatasetValidationError


def train_classifier(
    metadata: pd.DataFrame,
    audio_directory: str | Path,
    *,
    output_directory: str | Path,
    epochs: int = 5,
    batch_size: int = 32,
    max_samples: int | None = None,
    random_seed: int = 42,
) -> dict[str, Any]:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive.")
    if max_samples is not None and max_samples < 10:
        raise ValueError("max_samples must be at least 10.")

    labeled = metadata.loc[metadata["region"].str.casefold() == "kampala"].copy()
    if labeled["class"].nunique() < 2:
        raise DatasetValidationError(
            "At least two Kampala source classes are required for training."
        )
    audio_directory = Path(audio_directory)
    missing_files = [
        str(value)
        for value in labeled["filename"]
        if not (audio_directory / Path(str(value)).name).is_file()
    ]
    if missing_files:
        examples = ", ".join(missing_files[:5])
        raise FileNotFoundError(
            f"{len(missing_files)} Kampala recordings are missing from "
            f"{audio_directory}; extract the full archive first. Examples: {examples}"
        )
    labeled = labeled.reset_index(drop=True)
    if max_samples is not None and len(labeled) > max_samples:
        labeled = labeled.sample(n=max_samples, random_state=random_seed)
    if labeled["class"].nunique() < 2:
        raise DatasetValidationError(
            "The available Kampala audio files contain fewer than two classes."
        )

    labels = sorted(labeled["class"].astype(str).unique())
    groups = (
        labeled["latitude"].round(3).astype(str)
        + ":"
        + labeled["longitude"].round(3).astype(str)
    )
    if groups.nunique() < 3:
        raise DatasetValidationError(
            "At least three geographic groups are required for train, "
            "validation, and test splits."
        )
    outer_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=random_seed,
    )
    train_validation_indices, test_indices = next(
        outer_splitter.split(labeled, labeled["class"], groups=groups)
    )
    train_validation = labeled.iloc[train_validation_indices]
    train_validation_groups = groups.iloc[train_validation_indices]
    inner_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=random_seed + 1,
    )
    train_relative, validation_relative = next(
        inner_splitter.split(
            train_validation,
            train_validation["class"],
            groups=train_validation_groups,
        )
    )
    train_indices = train_validation_indices[train_relative]
    validation_indices = train_validation_indices[validation_relative]
    train_subset = Subset(
        UrbanAudioDataset(labeled, audio_directory, labels),
        train_indices.tolist(),
    )
    validation_subset = Subset(
        UrbanAudioDataset(labeled, audio_directory, labels),
        validation_indices.tolist(),
    )
    test_subset = Subset(
        UrbanAudioDataset(labeled, audio_directory, labels),
        test_indices.tolist(),
    )
    generator = torch.Generator().manual_seed(random_seed)
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        validation_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UrbanNoiseCNN(len(labels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_targets = (
        labeled.iloc[train_indices]["class"].map(
            {label: index for index, label in enumerate(labels)}
        ).to_numpy(dtype=np.int64)
    )
    class_counts = np.bincount(train_targets, minlength=len(labels))
    class_weights = np.zeros(len(labels), dtype=np.float32)
    present = class_counts > 0
    class_weights[present] = len(train_targets) / (
        int(present.sum()) * class_counts[present]
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    training_loss_history: list[float] = []
    validation_loss_history: list[float] = []
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(targets)
            total_examples += len(targets)
        training_loss_history.append(total_loss / total_examples)

        model.eval()
        validation_loss = 0.0
        validation_examples = 0
        with torch.no_grad():
            for features, targets in validation_loader:
                features = features.to(device)
                targets = targets.to(device)
                loss = criterion(model(features), targets)
                validation_loss += float(loss.item()) * len(targets)
                validation_examples += len(targets)
        mean_validation_loss = validation_loss / validation_examples
        validation_loss_history.append(mean_validation_loss)
        if mean_validation_loss < best_validation_loss:
            best_validation_loss = mean_validation_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("Training completed without a valid model checkpoint.")
    model.load_state_dict(best_state)

    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for features, targets in test_loader:
            predictions = model(features.to(device)).argmax(dim=1).cpu()
            y_true.extend(targets.tolist())
            y_pred.extend(predictions.tolist())

    observed_labels = list(range(len(labels)))
    report = classification_report(
        y_true,
        y_pred,
        labels=observed_labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=observed_labels).tolist()
    label_encoder = LabelEncoder().fit(labels)
    train_features = np.stack(
        [
            extract_mfcc_summary(audio_directory / Path(str(name)).name)
            for name in labeled.iloc[train_indices]["filename"]
        ]
    )
    test_features = np.stack(
        [
            extract_mfcc_summary(audio_directory / Path(str(name)).name)
            for name in labeled.iloc[test_indices]["filename"]
        ]
    )
    train_class_targets = label_encoder.transform(
        labeled.iloc[train_indices]["class"].astype(str)
    )
    test_class_targets = label_encoder.transform(
        labeled.iloc[test_indices]["class"].astype(str)
    )
    random_forest = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        random_state=random_seed,
        n_jobs=-1,
    )
    random_forest.fit(train_features, train_class_targets)
    forest_predictions = random_forest.predict(test_features)
    baseline_report = classification_report(
        test_class_targets,
        forest_predictions,
        labels=observed_labels,
        target_names=labels,
        output_dict=True,
        zero_division=0,
    )
    baseline_matrix = confusion_matrix(
        test_class_targets,
        forest_predictions,
        labels=observed_labels,
    ).tolist()
    results: dict[str, Any] = {
        "training_rows": len(train_indices),
        "validation_rows": len(validation_indices),
        "test_rows": len(test_indices),
        "classes": labels,
        "label_provenance": "Kampala subset only",
        "split_method": (
            "Nested GroupShuffleSplit on coordinates rounded to 0.001 degrees"
        ),
        "class_weights": {
            labels[index]: float(class_weights[index])
            for index in range(len(labels))
        },
        "device": str(device),
        "training_loss": training_loss_history,
        "validation_loss": validation_loss_history,
        "best_validation_loss": best_validation_loss,
        "classification_report": report,
        "confusion_matrix": matrix,
        "confusion_matrix_labels": [labels[index] for index in observed_labels],
        "mfcc_random_forest_baseline": {
            "features": "20 MFCC coefficients, mean and standard deviation",
            "n_estimators": 200,
            "classification_report": baseline_report,
            "confusion_matrix": baseline_matrix,
        },
    }

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "urban_noise_cnn.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": labels,
            "sample_rate": 16_000,
            "n_mels": 64,
            "clip_seconds": 5.0,
        },
        model_path,
    )
    import joblib

    forest_path = output_path / "mfcc_random_forest.joblib"
    joblib.dump(random_forest, forest_path)
    results["model_path"] = str(model_path)
    results["baseline_model_path"] = str(forest_path)
    with (output_path / "classification_metrics.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(results, stream, indent=2)

    figure, axis = plt.subplots(figsize=(8, 5))
    epochs_axis = np.arange(1, epochs + 1)
    axis.plot(epochs_axis, training_loss_history, label="Training")
    axis.plot(epochs_axis, validation_loss_history, label="Validation")
    axis.set(
        xlabel="Epoch",
        ylabel="Weighted cross-entropy loss",
        title="Urban noise source CNN learning curves",
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path / "training_curves.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis, label="Recordings")
    axis.set(
        xticks=np.arange(len(observed_labels)),
        yticks=np.arange(len(observed_labels)),
        xticklabels=[labels[index] for index in observed_labels],
        yticklabels=[labels[index] for index in observed_labels],
        xlabel="Predicted class",
        ylabel="True class",
        title="Geographic holdout confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=90, ha="right")
    figure.tight_layout()
    figure.savefig(output_path / "confusion_matrix.png", dpi=160)
    plt.close(figure)
    return results


def predict_recording(
    model_path: str | Path,
    audio_path: str | Path,
) -> dict[str, Any]:
    from .audio_model import load_log_mel

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    labels = checkpoint["classes"]
    model = UrbanNoiseCNN(len(labels))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    features = torch.from_numpy(load_log_mel(audio_path)).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        probabilities = torch.softmax(model(features), dim=1)[0].numpy()
    order = np.argsort(probabilities)[::-1]
    return {
        "recording": str(audio_path),
        "top_class": labels[int(order[0])],
        "probabilities": {
            labels[int(index)]: float(probabilities[index]) for index in order
        },
    }
