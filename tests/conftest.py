"""Shared fixtures: builds a tiny synthetic RSNA-format dataset for fast, offline tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.dataset import LABELS

# (SequenceID, Label, NumSlices)
_ROWS: list[tuple[str, str, int]] = [
    ("seq_a", "subdural", 5),
    ("seq_b", "epidural", 1),  # edge case: single-slice sequence
    ("seq_c", "intraparenchymal", 3),
]


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Writes a handful of small .npz sequences + a matching CSV under tmp_path.

    Returns (csv_path, data_root).
    """
    split_dir = tmp_path / "train"
    split_dir.mkdir()

    records = []
    for seq_id, label, num_slices in _ROWS:
        rng = np.random.default_rng(hash(seq_id) % (2**32))
        sequence = rng.integers(0, 256, size=(num_slices, 8, 8, 3), dtype=np.uint8)
        npz_path = split_dir / f"{seq_id}.npz"
        np.savez_compressed(npz_path, sequence=sequence)
        records.append(
            {
                "SequenceID": seq_id,
                "NPYPath": f"train/{seq_id}.npz",
                "Label": label,
                "NumSlices": num_slices,
                "PatientID": f"patient_{seq_id}",
                "StudyInstanceUID": f"study_{seq_id}",
            }
        )

    assert all(r["Label"] in LABELS for r in records)

    csv_path = tmp_path / "train.csv"
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path, tmp_path
