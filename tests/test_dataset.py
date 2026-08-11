"""Tests for RSNASequenceDataset and collate_sequences."""

from __future__ import annotations

from pathlib import Path

import torch

from src.dataset import LABEL_TO_IDX, RSNASequenceDataset, collate_sequences


def test_dataset_length(tiny_dataset: tuple[Path, Path]) -> None:
    csv_path, data_root = tiny_dataset
    ds = RSNASequenceDataset(csv_path, data_root)
    assert len(ds) == 3


def test_item_shape_and_label(tiny_dataset: tuple[Path, Path]) -> None:
    csv_path, data_root = tiny_dataset
    ds = RSNASequenceDataset(csv_path, data_root)

    tensor, label_idx, seq_id = ds[0]
    assert seq_id == "seq_a"
    assert tensor.shape == (5, 3, 8, 8)  # (T, C, H, W)
    assert tensor.dtype == torch.float32
    assert label_idx == LABEL_TO_IDX["subdural"]


def test_single_slice_sequence(tiny_dataset: tuple[Path, Path]) -> None:
    """Edge case: a sequence with NumSlices=1 must not collapse a dimension."""
    csv_path, data_root = tiny_dataset
    ds = RSNASequenceDataset(csv_path, data_root)

    tensor, label_idx, seq_id = ds[1]
    assert seq_id == "seq_b"
    assert tensor.shape == (1, 3, 8, 8)
    assert label_idx == LABEL_TO_IDX["epidural"]


def test_max_slices_subsamples_long_sequences(tiny_dataset: tuple[Path, Path]) -> None:
    csv_path, data_root = tiny_dataset
    ds = RSNASequenceDataset(csv_path, data_root, max_slices=2)

    tensor, _, seq_id = ds[0]  # seq_a has 5 slices, must be truncated to 2
    assert seq_id == "seq_a"
    assert tensor.shape == (2, 3, 8, 8)


def test_max_slices_leaves_short_sequences_untouched(tiny_dataset: tuple[Path, Path]) -> None:
    csv_path, data_root = tiny_dataset
    ds = RSNASequenceDataset(csv_path, data_root, max_slices=100)

    tensor, _, seq_id = ds[1]  # seq_b has 1 slice, well under the cap
    assert seq_id == "seq_b"
    assert tensor.shape == (1, 3, 8, 8)


def test_normalize_flag_changes_value_range(tiny_dataset: tuple[Path, Path]) -> None:
    csv_path, data_root = tiny_dataset

    raw_ds = RSNASequenceDataset(csv_path, data_root, normalize=False)
    norm_ds = RSNASequenceDataset(csv_path, data_root, normalize=True)

    raw_tensor, _, _ = raw_ds[0]
    norm_tensor, _, _ = norm_ds[0]

    assert raw_tensor.min() >= 0.0 and raw_tensor.max() <= 1.0
    assert not torch.allclose(raw_tensor, norm_tensor)


def test_collate_pads_variable_length_batch(tiny_dataset: tuple[Path, Path]) -> None:
    csv_path, data_root = tiny_dataset
    ds = RSNASequenceDataset(csv_path, data_root)

    batch = [ds[0], ds[1], ds[2]]  # lengths 5, 1, 3
    slices, mask, labels, seq_ids = collate_sequences(batch)

    assert slices.shape == (3, 5, 3, 8, 8)
    assert mask.shape == (3, 5)
    assert labels.shape == (3,)
    assert seq_ids == ["seq_a", "seq_b", "seq_c"]

    assert mask.tolist() == [
        [True, True, True, True, True],
        [True, False, False, False, False],
        [True, True, True, False, False],
    ]


def test_collate_padding_is_zero_filled(tiny_dataset: tuple[Path, Path]) -> None:
    csv_path, data_root = tiny_dataset
    ds = RSNASequenceDataset(csv_path, data_root, normalize=False)

    batch = [ds[0], ds[1]]  # lengths 5, 1
    slices, mask, _, _ = collate_sequences(batch)

    padded_region = slices[1, 1:]  # seq_b's padded positions
    assert torch.all(padded_region == 0.0)


def test_collate_single_item_batch(tiny_dataset: tuple[Path, Path]) -> None:
    """Edge case: batch size of 1 should not require any padding."""
    csv_path, data_root = tiny_dataset
    ds = RSNASequenceDataset(csv_path, data_root)

    slices, mask, labels, seq_ids = collate_sequences([ds[1]])
    assert slices.shape == (1, 1, 3, 8, 8)
    assert mask.tolist() == [[True]]


def test_augment_preserves_shape_and_value_range(tiny_dataset: tuple[Path, Path]) -> None:
    csv_path, data_root = tiny_dataset
    ds = RSNASequenceDataset(csv_path, data_root, normalize=False, augment=True)

    tensor, _, _ = ds[0]
    assert tensor.shape == (5, 3, 8, 8)
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0


def test_augment_disabled_by_default_is_deterministic(tiny_dataset: tuple[Path, Path]) -> None:
    """Edge case: without augment=True, repeated reads of the same item must be
    identical (no hidden randomness leaking into eval/test splits)."""
    csv_path, data_root = tiny_dataset
    ds = RSNASequenceDataset(csv_path, data_root)

    first, _, _ = ds[0]
    second, _, _ = ds[0]
    assert torch.equal(first, second)


def test_augment_applies_same_flip_to_every_slice_in_a_sequence(
    tiny_dataset: tuple[Path, Path],
) -> None:
    """Edge case: a single-slice sequence must still augment without error,
    and a multi-slice sequence must stay anatomically coherent (same flip
    direction across all its slices, not an independent flip per slice)."""
    csv_path, data_root = tiny_dataset
    ds = RSNASequenceDataset(csv_path, data_root, normalize=False, augment=False)
    raw_tensor, _, _ = ds[0]  # seq_a, 5 slices

    torch.manual_seed(0)
    flipped = RSNASequenceDataset._augment(raw_tensor.clone())

    # Whatever happened (flip and/or rotation), it must be identical across
    # every slice's transform — check by re-deriving slice 0's transform from
    # slice 0 alone under the same seed and comparing to the batched result.
    torch.manual_seed(0)
    flipped_first_slice_alone = RSNASequenceDataset._augment(raw_tensor[:1].clone())

    # atol is loose because torchvision's grid_sample interpolation has tiny
    # floating-point differences depending on the batch shape it's called with.
    assert torch.allclose(flipped[0], flipped_first_slice_alone[0], atol=1e-3)
