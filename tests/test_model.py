"""Tests for SliceEncoder, AttentionPooling, and HemorrhageSequenceClassifier."""

from __future__ import annotations

import torch

from src.model import AttentionPooling, HemorrhageSequenceClassifier, SliceEncoder


def test_slice_encoder_output_shape() -> None:
    encoder = SliceEncoder(pretrained=False)
    encoder.eval()

    x = torch.randn(6, 3, 64, 64)  # (N, C, H, W)
    with torch.no_grad():
        out = encoder(x)

    assert out.shape == (6, encoder.out_dim)


def test_attention_pooling_ignores_padded_positions() -> None:
    """Changing a masked-out embedding must not change the pooled output."""
    torch.manual_seed(0)
    pooling = AttentionPooling(in_dim=4)
    pooling.eval()

    embeddings = torch.randn(1, 3, 4)
    mask = torch.tensor([[True, True, False]])

    with torch.no_grad():
        out_original = pooling(embeddings, mask)

        embeddings_perturbed = embeddings.clone()
        embeddings_perturbed[0, 2] = 1000.0  # blow up the masked-out slice
        out_perturbed = pooling(embeddings_perturbed, mask)

    assert torch.allclose(out_original, out_perturbed, atol=1e-6)


def test_attention_pooling_all_valid_weights_sum_to_one() -> None:
    torch.manual_seed(0)
    pooling = AttentionPooling(in_dim=4)
    pooling.eval()

    embeddings = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)

    scores = pooling.score(embeddings).squeeze(-1)
    weights = torch.softmax(scores, dim=1)
    assert torch.allclose(weights.sum(dim=1), torch.ones(2), atol=1e-6)


def test_classifier_forward_shape_varying_batch_and_length() -> None:
    model = HemorrhageSequenceClassifier(num_classes=5, pretrained=False)
    model.eval()

    slices = torch.randn(2, 4, 3, 64, 64)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])

    with torch.no_grad():
        logits = model(slices, mask)

    assert logits.shape == (2, 5)


def test_classifier_skips_encoder_compute_on_padded_slices() -> None:
    """Padded positions must never reach the CNN encoder — memory/compute must
    scale with the number of real slices, not batch_size * max_len."""
    model = HemorrhageSequenceClassifier(num_classes=5, pretrained=False)
    model.eval()

    seen_batch_sizes: list[int] = []
    original_forward = model.encoder.forward

    def spy_forward(x: torch.Tensor) -> torch.Tensor:
        seen_batch_sizes.append(x.shape[0])
        return original_forward(x)

    model.encoder.forward = spy_forward  # type: ignore[method-assign]

    slices = torch.randn(2, 4, 3, 64, 64)
    mask = torch.tensor([[True, True, True, False], [True, False, False, False]])

    with torch.no_grad():
        model(slices, mask)

    assert seen_batch_sizes == [int(mask.sum())]  # 4 real slices, not 8


def test_classifier_forward_single_slice_sequence() -> None:
    """Edge case: T=1, matching real single-slice sequences in the dataset."""
    model = HemorrhageSequenceClassifier(num_classes=5, pretrained=False)
    model.eval()

    slices = torch.randn(1, 1, 3, 64, 64)
    mask = torch.tensor([[True]])

    with torch.no_grad():
        logits = model(slices, mask)

    assert logits.shape == (1, 5)
    assert torch.isfinite(logits).all()
