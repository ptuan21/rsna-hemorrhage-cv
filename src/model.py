"""CNN encoder + attention pooling model for variable-length CT slice sequences."""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision


class SliceEncoder(nn.Module):
    """Shared 2D CNN backbone that embeds each CT slice independently."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        self.features = nn.Sequential(*list(backbone.children())[:-1])  # drop fc layer
        self.out_dim: int = 512

    def forward(self, slices: torch.Tensor) -> torch.Tensor:
        """slices: (N, C, H, W) -> (N, out_dim)."""
        features = self.features(slices)
        return features.flatten(1)


class AttentionPooling(nn.Module):
    """Masked additive attention pooling over the slice (sequence) dimension."""

    def __init__(self, in_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """embeddings: (B, T, D), mask: (B, T) bool, True = real slice -> (B, D)."""
        scores = self.score(embeddings).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=1)  # (B, T)
        pooled = torch.bmm(weights.unsqueeze(1), embeddings).squeeze(1)  # (B, D)
        return pooled


class HemorrhageSequenceClassifier(nn.Module):
    """Per-slice CNN encoder -> attention pooling -> linear classifier head."""

    def __init__(self, num_classes: int, pretrained: bool = True) -> None:
        super().__init__()
        self.encoder = SliceEncoder(pretrained=pretrained)
        self.pooling = AttentionPooling(in_dim=self.encoder.out_dim)
        self.classifier = nn.Linear(self.encoder.out_dim, num_classes)

    def forward(self, slices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """slices: (B, T, C, H, W), mask: (B, T) -> logits (B, num_classes)."""
        b, t, c, h, w = slices.shape
        flat = slices.view(b * t, c, h, w)
        embeddings = self.encoder(flat).view(b, t, -1)
        pooled = self.pooling(embeddings, mask)
        return self.classifier(pooled)
