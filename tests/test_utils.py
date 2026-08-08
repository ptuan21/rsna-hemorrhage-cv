"""Tests for seeding, device selection, class weights, and metrics helpers."""

from __future__ import annotations

import torch

from src.utils import compute_class_weights, compute_metrics, get_device, set_seed


def test_set_seed_reproducibility() -> None:
    set_seed(123)
    a = torch.rand(10)
    set_seed(123)
    b = torch.rand(10)
    assert torch.equal(a, b)


def test_get_device_explicit_cpu() -> None:
    assert get_device("cpu") == torch.device("cpu")


def test_compute_class_weights_favors_rare_classes() -> None:
    # class 0 appears 3x, class 1 appears 1x -> class 1 should get a larger weight
    labels = [0, 0, 0, 1]
    weights = compute_class_weights(labels, num_classes=2)

    assert weights[1] > weights[0]
    assert torch.isclose(weights[1] / weights[0], torch.tensor(3.0), atol=1e-4)


def test_compute_class_weights_handles_absent_class() -> None:
    """Edge case: a class with zero occurrences must not raise or produce inf/nan."""
    labels = [0, 0, 1]
    weights = compute_class_weights(labels, num_classes=3)  # class 2 absent

    assert weights.shape == (3,)
    assert torch.isfinite(weights).all()


def test_compute_metrics_perfect_predictions() -> None:
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    metrics = compute_metrics(y_true, y_pred, num_classes=5)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_compute_metrics_all_wrong_predictions() -> None:
    y_true = [0, 0, 0]
    y_pred = [1, 1, 1]
    metrics = compute_metrics(y_true, y_pred, num_classes=2)

    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0


def test_compute_metrics_handles_class_absent_from_batch() -> None:
    """Edge case: num_classes=5 but only 2 classes present in this batch/split."""
    y_true = [0, 0, 1]
    y_pred = [0, 1, 1]
    metrics = compute_metrics(y_true, y_pred, num_classes=5)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
