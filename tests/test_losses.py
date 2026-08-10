"""Tests for FocalLoss."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.losses import FocalLoss


def test_focal_loss_is_lower_for_easy_confident_correct_predictions() -> None:
    loss_fn = FocalLoss(gamma=2.0)

    easy_logits = torch.tensor([[10.0, -10.0, -10.0]])  # very confident, correct
    hard_logits = torch.tensor([[0.1, 0.05, -0.05]])  # barely correct
    target = torch.tensor([0])

    easy_loss = loss_fn(easy_logits, target)
    hard_loss = loss_fn(hard_logits, target)

    assert easy_loss.item() < hard_loss.item()


def test_focal_loss_with_gamma_zero_matches_weighted_cross_entropy() -> None:
    torch.manual_seed(0)
    alpha = torch.tensor([1.0, 2.0, 0.5])
    logits = torch.randn(8, 3)
    targets = torch.randint(0, 3, (8,))

    focal = FocalLoss(gamma=0.0, alpha=alpha)
    ce = torch.nn.CrossEntropyLoss(weight=alpha)

    assert torch.allclose(focal(logits, targets), ce(logits, targets), atol=1e-5)


def test_focal_loss_without_alpha_runs() -> None:
    loss_fn = FocalLoss(gamma=2.0, alpha=None)
    logits = torch.randn(4, 5)
    targets = torch.randint(0, 5, (4,))

    loss = loss_fn(logits, targets)
    assert loss.dim() == 0
    assert torch.isfinite(loss)


def test_focal_loss_alpha_upweights_rare_class_within_a_batch() -> None:
    """Within one batch, the rare class (higher alpha) must pull gradients more
    than an equally-hard common-class example, even though the loss is
    normalized by the batch's total alpha (so isolated single-sample batches
    can't reveal this — the effect only shows up in relative contribution)."""
    alpha = torch.tensor([1.0, 5.0])  # class 1 is rare, weighted 5x
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha)

    logits = torch.zeros(2, 2, requires_grad=True)  # identical, ambiguous predictions
    targets = torch.tensor([0, 1])  # sample 0 -> common class, sample 1 -> rare class

    loss_fn(logits, targets).backward()

    grad_common = logits.grad[0].norm()
    grad_rare = logits.grad[1].norm()
    assert grad_rare > grad_common


def test_focal_loss_single_sample_batch() -> None:
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.randn(1, 5)
    targets = torch.tensor([2])

    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)
