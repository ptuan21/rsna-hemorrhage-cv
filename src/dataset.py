"""Dataset loading for the RSNA multi-window hemorrhage sequence dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

LABELS: tuple[str, ...] = (
    "epidural",
    "subdural",
    "subarachnoid",
    "intraventricular",
    "intraparenchymal",
)
LABEL_TO_IDX: dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class RSNASequenceDataset(Dataset):
    """Loads variable-length CT slice sequences stored as per-sequence .npz files.

    Each item is one hemorrhage-subtype sequence: a stack of RGB CT slices
    (R=Brain, G=Subdural, B=Bone window) with shape (NumSlices, 384, 384, 3).
    """

    def __init__(
        self,
        csv_path: Union[str, Path],
        data_root: Union[str, Path],
        max_slices: Optional[int] = None,
        normalize: bool = True,
        augment: bool = False,
    ) -> None:
        self.data_root = Path(data_root)
        self.df = pd.read_csv(csv_path)
        self.max_slices = max_slices
        self.normalize = normalize
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self.df.iloc[index]
        npz_path = self.data_root / row["NPYPath"]
        with np.load(npz_path) as data:
            sequence = data["sequence"]  # (T, 384, 384, 3) uint8

        sequence = self._subsample(sequence)

        tensor = torch.from_numpy(sequence.copy()).float() / 255.0  # (T, H, W, C)
        tensor = tensor.permute(0, 3, 1, 2).contiguous()  # (T, C, H, W)
        if self.augment:
            tensor = self._augment(tensor)
        if self.normalize:
            tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD

        label_idx = LABEL_TO_IDX[row["Label"]]
        return tensor, label_idx, str(row["SequenceID"])

    def _subsample(self, sequence: np.ndarray) -> np.ndarray:
        num_slices = sequence.shape[0]
        if self.max_slices is None or num_slices <= self.max_slices:
            return sequence
        indices = np.linspace(0, num_slices - 1, self.max_slices).round().astype(int)
        return sequence[indices]

    @staticmethod
    def _augment(tensor: torch.Tensor) -> torch.Tensor:
        """Light, sequence-consistent augmentation on a raw [0, 1] slice stack.

        The same flip/rotation is applied to every slice in the sequence (they
        come from one CT case and must stay anatomically coherent), while
        brightness/contrast jitter approximates scanner/windowing variation.
        """
        if torch.rand(()).item() < 0.5:
            tensor = tensor.flip(-1)

        angle = float(torch.empty(()).uniform_(-10.0, 10.0))
        tensor = TF.rotate(tensor, angle, fill=0.0)

        brightness = float(torch.empty(()).uniform_(0.9, 1.1))
        contrast = float(torch.empty(()).uniform_(0.9, 1.1))
        mean = tensor.mean()
        tensor = (tensor - mean) * contrast + mean * brightness
        return tensor.clamp(0.0, 1.0)


def collate_sequences(
    batch: Sequence[tuple[torch.Tensor, int, str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
    """Zero-pads a batch of variable-length slice sequences to the batch max length.

    Returns:
        slices: (B, T_max, C, H, W) padded slice tensor.
        mask: (B, T_max) bool tensor, True for real slices, False for padding.
        labels: (B,) long tensor of label indices.
        sequence_ids: list of SequenceID strings, one per item.
    """
    tensors, labels, sequence_ids = zip(*batch)
    lengths = [t.shape[0] for t in tensors]
    max_len = max(lengths)
    _, c, h, w = tensors[0].shape

    padded = torch.zeros(len(tensors), max_len, c, h, w, dtype=tensors[0].dtype)
    mask = torch.zeros(len(tensors), max_len, dtype=torch.bool)
    for i, (t, length) in enumerate(zip(tensors, lengths)):
        padded[i, :length] = t
        mask[i, :length] = True

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    return padded, mask, labels_tensor, list(sequence_ids)
