"""Reproducibility, device selection, and metric helpers."""

from __future__ import annotations

import random
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def set_seed(seed: int) -> None:
    """Seeds python, numpy, and all available torch backends for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str = "auto") -> torch.device:
    """Resolves 'auto' to the best available backend: CUDA > MPS > CPU."""
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_class_weights(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss, normalized to mean 1."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0  # avoid div-by-zero for classes absent in this split
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_sample_weights(labels: Sequence[int], class_weights: torch.Tensor) -> list[float]:
    """Maps each training example to its class weight, for WeightedRandomSampler.

    Feeding these into WeightedRandomSampler oversamples minority classes at the
    batch level, complementing (not replacing) loss-level class weighting.
    """
    return [float(class_weights[label]) for label in labels]


def compute_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int
) -> dict[str, float]:
    """Overall accuracy and macro-averaged F1 across all `num_classes` labels."""
    labels = list(range(num_classes))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
    }
