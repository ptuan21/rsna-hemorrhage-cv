"""Training entrypoint for the RSNA hemorrhage sequence classifier."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.dataset import LABEL_TO_IDX, LABELS, RSNASequenceDataset, collate_sequences
from src.losses import FocalLoss
from src.model import HemorrhageSequenceClassifier
from src.utils import (
    build_sample_weights,
    compute_class_weights,
    compute_metrics,
    get_device,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the RSNA hemorrhage sequence classifier."
    )
    parser.add_argument("--data-root", type=Path, default=Path("rsna_data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-slices", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Stop early after this many epochs with no val macro-F1 improvement.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def run_epoch(
    model: HemorrhageSequenceClassifier,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip_norm: Optional[float] = None,
) -> tuple[float, dict[str, float]]:
    """Runs one pass over `loader`; trains if `optimizer` is given, else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.set_grad_enabled(is_train):
        for slices, mask, labels, _ in tqdm(loader, leave=False):
            slices, mask, labels = slices.to(device), mask.to(device), labels.to(device)

            logits = model(slices, mask)
            loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            all_preds.extend(logits.argmax(dim=1).detach().cpu().tolist())
            all_labels.extend(labels.detach().cpu().tolist())

    avg_loss = total_loss / len(loader.dataset)
    metrics = compute_metrics(all_labels, all_preds, num_classes=len(LABELS))
    return avg_loss, metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    print(f"Using device: {device}")

    train_ds = RSNASequenceDataset(
        args.data_root / "train.csv", args.data_root, max_slices=args.max_slices, augment=True
    )
    val_ds = RSNASequenceDataset(
        args.data_root / "validation.csv", args.data_root, max_slices=args.max_slices
    )

    train_labels = train_ds.df["Label"].map(LABEL_TO_IDX).tolist()
    class_weights = compute_class_weights(train_labels, num_classes=len(LABELS))
    sample_weights = build_sample_weights(train_labels, class_weights)
    sampler = WeightedRandomSampler(
        sample_weights, num_samples=len(train_labels), replacement=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_sequences,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_sequences,
    )

    model = HemorrhageSequenceClassifier(
        num_classes=len(LABELS), pretrained=not args.no_pretrained
    ).to(device)

    criterion = FocalLoss(gamma=args.focal_gamma, alpha=class_weights).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_macro_f1 = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer, args.grad_clip_norm
        )
        val_loss, val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None)
        scheduler.step()

        print(
            f"epoch {epoch:03d} | lr={scheduler.get_last_lr()[0]:.2e} | "
            f"train_loss={train_loss:.4f} acc={train_metrics['accuracy']:.4f} "
            f"macro_f1={train_metrics['macro_f1']:.4f} | "
            f"val_loss={val_loss:.4f} acc={val_metrics['accuracy']:.4f} "
            f"macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            epochs_without_improvement = 0
            torch.save(model.state_dict(), args.checkpoint_dir / "best_model.pt")
            print(f"  -> saved new best checkpoint (macro_f1={best_macro_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(
                    f"  -> stopping early: no val macro_f1 improvement in "
                    f"{args.patience} epochs"
                )
                break


if __name__ == "__main__":
    main()
