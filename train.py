"""
Training script for the DeepfakeCNNLSTM model.

Usage:
    python train.py --data-root ./data --epochs 15 --batch-size 4

Populate ./data/real/*.mp4 and ./data/fake/*.mp4 first -- use
prepare_dataset.py to sort one or more labeled datasets (e.g. multiple
Kaggle DFDC downloads, FaceForensics++, Celeb-DF, or your own labeled
clips) into those folders. Every dataset you add to data/real and
data/fake gets trained on together automatically -- more and more varied
real-world examples is what actually moves accuracy closer to the ceiling,
more than any single hyperparameter tweak.

On completion, the best checkpoint (by validation F1 score) is written to
saved_models/deepfake_cnn_lstm.pt -- the Flask app automatically detects
and loads it the next time it starts, replacing the heuristic baseline
and pretrained-classifier tiers.
"""

import os
import argparse
import json
import random
from datetime import datetime, timezone

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from dataset import DeepfakeVideoDataset
from models.cnn_lstm import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Train the Deepfake CNN-LSTM detector")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-frames", type=int, default=20)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="./saved_models/deepfake_cnn_lstm.pt")
    p.add_argument("--pretrained-cnn", action="store_true",
                    help="Warm-start the CNN backbone from ImageNet weights (needs internet).")
    p.add_argument("--resume", type=str, default=None,
                    help="Path to a checkpoint to resume training from (continue improving an "
                         "existing model as you add more datasets, instead of starting over).")
    p.add_argument("--early-stop-patience", type=int, default=5,
                    help="Stop training if validation F1 doesn't improve for this many epochs "
                         "in a row (0 disables early stopping).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def stratified_split(dataset: DeepfakeVideoDataset, val_fraction: float, seed: int):
    """
    Splits indices into train/val while preserving each class's real:fake
    ratio in BOTH splits -- plain random_split can accidentally starve the
    validation set of one class entirely on small/imbalanced datasets,
    which makes the reported val accuracy meaningless.
    """
    rng = random.Random(seed)
    real_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 0]
    fake_indices = [i for i, (_, label) in enumerate(dataset.samples) if label == 1]
    rng.shuffle(real_indices)
    rng.shuffle(fake_indices)

    def split_one_class(indices):
        n_val = max(1, int(len(indices) * val_fraction)) if indices else 0
        return indices[n_val:], indices[:n_val]

    real_train, real_val = split_one_class(real_indices)
    fake_train, fake_val = split_one_class(fake_indices)

    train_indices = real_train + fake_train
    val_indices = real_val + fake_val
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def compute_class_weight(dataset: DeepfakeVideoDataset, indices: list) -> float:
    """
    Returns pos_weight for BCEWithLogitsLoss: how much to up-weight the
    FAKE class relative to REAL. Datasets like DFDC are often heavily
    imbalanced (many more fakes than reals, since each real video
    typically spawns several fake variants) -- without this, the model
    can cheat by mostly predicting the majority class and still look
    accurate while being nearly useless.
    """
    labels = [dataset.samples[i][1] for i in indices]
    n_real = sum(1 for l in labels if l == 0)
    n_fake = sum(1 for l in labels if l == 1)
    if n_fake == 0:
        return 1.0
    return n_real / n_fake


def compute_metrics(all_labels, all_preds):
    """Accuracy, precision, recall, F1, and a confusion matrix -- matches
    the evaluation metrics named in the project report."""
    tp = sum(1 for l, p in zip(all_labels, all_preds) if l == 1 and p == 1)
    tn = sum(1 for l, p in zip(all_labels, all_preds) if l == 0 and p == 0)
    fp = sum(1 for l, p in zip(all_labels, all_preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(all_labels, all_preds) if l == 1 and p == 0)

    total = max(len(all_labels), 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total = 0.0, 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for clips, labels in loader:
            clips, labels = clips.to(device), labels.to(device)
            logits = model(clips).squeeze(-1)
            loss = criterion(logits, labels)
            total_loss += loss.item() * clips.size(0)
            total += labels.size(0)

            preds = (torch.sigmoid(logits) >= 0.5).float()
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    metrics = compute_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics


def main():
    args = parse_args()
    device = torch.device(args.device)
    print(f"[train.py] Using device: {device}")

    # Two dataset instances over the same file list: augmentation is only
    # applied to the samples that end up in the training split, validation
    # stays on clean (un-augmented) data for an honest metrics readout.
    full_dataset = DeepfakeVideoDataset(args.data_root, n_frames=args.n_frames, augment=False)
    n_real = sum(1 for _, label in full_dataset.samples if label == 0)
    n_fake = sum(1 for _, label in full_dataset.samples if label == 1)
    print(f"[train.py] Dataset: {len(full_dataset)} videos total ({n_real} real, {n_fake} fake)")

    train_indices, val_indices = stratified_split(full_dataset, args.val_split, args.seed)
    print(f"[train.py] Stratified split: {len(train_indices)} train, {len(val_indices)} val "
          f"(class ratio preserved in both)")

    train_dataset = DeepfakeVideoDataset(args.data_root, n_frames=args.n_frames, augment=True)
    train_set = Subset(train_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model_config = {
        "embedding_dim": 256, "lstm_hidden_dim": 256,
        "lstm_layers": 1, "bidirectional": True, "dropout": 0.4,
    }
    model = build_model({**model_config, "pretrained_cnn": args.pretrained_cnn}).to(device)

    start_epoch = 1
    best_val_f1 = 0.0
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_val_f1 = checkpoint.get("val_f1", checkpoint.get("val_acc", 0.0))
        print(f"[train.py] Resumed from {args.resume} (previous best val_f1={best_val_f1:.4f})")

    # Class-weighted loss to counter real:fake imbalance (see compute_class_weight docstring)
    pos_weight_value = compute_class_weight(full_dataset, train_indices)
    print(f"[train.py] Class imbalance correction: pos_weight={pos_weight_value:.3f} "
          f"(applied to the FAKE class in the loss function)")
    pos_weight = torch.tensor([pos_weight_value], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    history = []
    epochs_without_improvement = 0

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        running_loss, total = 0.0, 0
        train_labels, train_preds = [], []

        for clips, labels in train_loader:
            clips, labels = clips.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(clips).squeeze(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * clips.size(0)
            total += labels.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_labels.extend(labels.cpu().tolist())
            train_preds.extend(preds.cpu().tolist())

        train_metrics = compute_metrics(train_labels, train_preds)
        train_metrics["loss"] = running_loss / max(total, 1)

        val_metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_metrics["f1"])

        print(f"Epoch {epoch:02d}/{start_epoch + args.epochs - 1} | "
              f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
              f"val_precision={val_metrics['precision']:.4f} val_recall={val_metrics['recall']:.4f} "
              f"val_f1={val_metrics['f1']:.4f}")

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"], "train_acc": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"], "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"], "val_confusion_matrix": val_metrics["confusion_matrix"],
        })

        if val_metrics["f1"] >= best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_acc": val_metrics["accuracy"],
                "val_f1": val_metrics["f1"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "epoch": epoch,
                "config": model_config,
            }, args.out)
            print(f"  -> New best model saved to {args.out} (val_f1={best_val_f1:.4f})")
        else:
            epochs_without_improvement += 1
            if args.early_stop_patience and epochs_without_improvement >= args.early_stop_patience:
                print(f"\n[train.py] No improvement in val_f1 for {args.early_stop_patience} epochs. "
                      f"Stopping early at epoch {epoch}.")
                break

    metrics_path = os.path.join(os.path.dirname(args.out), "training_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "best_val_acc": history[-1]["val_acc"] if history else None,
            "best_val_f1": best_val_f1,
            "dataset_size": {"real": n_real, "fake": n_fake},
            "history": history,
        }, f, indent=2)

    print(f"\nTraining complete. Best validation F1: {best_val_f1:.4f}")
    print(f"Checkpoint: {args.out}")
    print(f"Metrics log: {metrics_path}")


if __name__ == "__main__":
    main()
