"""Evaluation entrypoint: runs a trained checkpoint on a dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import LABELS, RSNASequenceDataset, collate_sequences
from src.model import HemorrhageSequenceClassifier
from src.utils import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--data-root", type=Path, default=Path("rsna_data"))
    parser.add_argument(
        "--split", type=str, default="test", choices=["train", "validation", "test"]
    )
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-slices", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    dataset = RSNASequenceDataset(
        args.data_root / f"{args.split}.csv", args.data_root, max_slices=args.max_slices
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sequences,
    )

    model = HemorrhageSequenceClassifier(num_classes=len(LABELS), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []
    all_ids: list[str] = []

    for slices, mask, labels, sequence_ids in tqdm(loader):
        slices, mask = slices.to(device), mask.to(device)
        logits = model(slices, mask)
        preds = logits.argmax(dim=1).cpu().tolist()

        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
        all_ids.extend(sequence_ids)

    print(classification_report(all_labels, all_preds, target_names=LABELS, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(all_labels, all_preds))

    if args.output_csv is not None:
        pd.DataFrame(
            {
                "SequenceID": all_ids,
                "true_label": [LABELS[i] for i in all_labels],
                "pred_label": [LABELS[i] for i in all_preds],
            }
        ).to_csv(args.output_csv, index=False)
        print(f"Predictions written to {args.output_csv}")


if __name__ == "__main__":
    main()
