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


class SequenceContextEncoder(nn.Module):
    """Bidirectional GRU that contextualizes each slice's embedding using its
    neighbors before pooling.

    Plain attention pooling is permutation-invariant, so it otherwise ignores
    slice order entirely — but a hemorrhage finding typically spans several
    consecutive slices, and this lets each slice's representation pick up
    signal from adjacent slices before the sequence is collapsed to one vector.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.out_dim: int = hidden_dim * 2

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """embeddings: (B, T, in_dim), mask: (B, T) -> (B, T, out_dim)."""
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings, lengths, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=embeddings.size(1)
        )
        return out


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
    """Per-slice CNN encoder -> GRU sequence context -> attention pooling -> classifier."""

    def __init__(
        self, num_classes: int, pretrained: bool = True, gru_hidden_dim: int = 256
    ) -> None:
        super().__init__()
        self.encoder = SliceEncoder(pretrained=pretrained)
        self.sequence_encoder = SequenceContextEncoder(
            in_dim=self.encoder.out_dim, hidden_dim=gru_hidden_dim
        )
        self.pooling = AttentionPooling(in_dim=self.sequence_encoder.out_dim)
        self.classifier = nn.Linear(self.sequence_encoder.out_dim, num_classes)

    def forward(self, slices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """slices: (B, T, C, H, W), mask: (B, T) -> logits (B, num_classes).

        Only real (non-padded) slices are run through the CNN encoder — padded
        positions are skipped entirely, since batches routinely mix short and
        long sequences and full-batch padding otherwise wastes both compute
        and, at 384x384 resolution, GPU memory.
        """
        b, t, c, h, w = slices.shape
        flat = slices.view(b * t, c, h, w)
        flat_mask = mask.view(b * t)

        valid_embeddings = self.encoder(flat[flat_mask])  # (N_valid, out_dim)
        embeddings = flat.new_zeros(b * t, self.encoder.out_dim)
        embeddings[flat_mask] = valid_embeddings
        embeddings = embeddings.view(b, t, -1)

        context = self.sequence_encoder(embeddings, mask)
        pooled = self.pooling(context, mask)
        return self.classifier(pooled)
