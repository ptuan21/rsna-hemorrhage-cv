"""Focal loss for severely imbalanced multi-class classification."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss (Lin et al., 2017).

    Down-weights easy, already well-classified examples so gradient signal
    concentrates on hard/rare examples — e.g. the epidural class, which makes
    up under 3% of this dataset and is otherwise drowned out by the majority
    classes even under class-weighted cross-entropy.
    """

    def __init__(self, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", alpha)
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_probs = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        target_probs = target_log_probs.exp()

        focal_weight = (1.0 - target_probs) ** self.gamma
        loss = -focal_weight * target_log_probs

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            # Normalize by sum of weights (matching nn.CrossEntropyLoss's weighted
            # mean convention) so gamma=0 reduces exactly to weighted CE.
            return (alpha_t * loss).sum() / alpha_t.sum()

        return loss.mean()
