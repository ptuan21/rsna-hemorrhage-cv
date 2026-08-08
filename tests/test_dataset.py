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
