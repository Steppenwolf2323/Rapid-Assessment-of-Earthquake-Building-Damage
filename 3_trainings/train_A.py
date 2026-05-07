"""
train.py — Model A: Pure End-to-End ML

Two-stage fine-tuning on QQB:

  Stage 1 — Warm-up (10 epochs)
      Backbone frozen. Only the classification head is trained.
      The head learns to read ResNet-50's ImageNet features before
      any gradients touch the pretrained backbone weights.
      LR: 1e-3

  Stage 2 — Full fine-tune (20 epochs)
      Backbone unfrozen. Backbone and head trained together with
      differential learning rates: backbone gets 10× smaller LR
      than the head, to preserve ImageNet knowledge while gently
      adapting to satellite imagery.
      LR: backbone 1e-5  |  head 1e-4

Class imbalance (23:1) is handled by WeightedRandomSampler,
which rebalances mini-batch sampling without touching the images.

Primary metric: F1 on the damaged class.
With 23:1 imbalance, accuracy alone is meaningless — a model that
predicts "intact" for everything gets 95.6% accuracy.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score
)

import config as cfg
from dataset import QQBDataset
from model import ModelA


# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(
    labels: list,
    preds: list,
    probs: list,
    loss: float,
) -> dict:
    """
    All metrics computed on the damaged class (positive class = 1).
    With 23:1 imbalance, F1 and Recall are the metrics to watch.

    - Recall (sensitivity): of all damaged buildings, how many did we catch?
      Missing a damaged building is costly → high recall is critical.
    - Precision: of all buildings we flagged as damaged, how many really are?
    - F1: harmonic mean of precision and recall.
    - AUC-ROC: overall discriminative ability regardless of threshold.
    """
    # Guard against degenerate predictions (all one class)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")

    return {
        "loss":      round(loss, 6),
        "f1":        round(f1_score(labels, preds,        zero_division=0), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall":    round(recall_score(labels, preds,    zero_division=0), 4),
        "auc":       round(auc, 4),
    }


# ── Training loop for one epoch ───────────────────────────────────────────────
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.train()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)          # [B]
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        with torch.no_grad():
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= cfg.THRESHOLD).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.long().cpu().numpy().tolist())

    return compute_metrics(
        all_labels, all_preds, all_probs,
        total_loss / len(loader),
    )


# ── Validation loop ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images).squeeze(1)
        loss   = criterion(logits, labels)

        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= cfg.THRESHOLD).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.long().cpu().numpy().tolist())

    return compute_metrics(
        all_labels, all_preds, all_probs,
        total_loss / len(loader),
    )


# ── Stage runner ──────────────────────────────────────────────────────────────
def run_stage(
    stage_name:    str,
    model:         nn.Module,
    train_loader:  DataLoader,
    val_loader:    DataLoader,
    optimizer:     torch.optim.Optimizer,
    scheduler:     torch.optim.lr_scheduler._LRScheduler,
    criterion:     nn.Module,
    device:        torch.device,
    n_epochs:      int,
    output_dir:    Path,
) -> list[dict]:
    """
    Runs one training stage, saves the best model (by val F1),
    and returns the per-epoch history.
    """
    best_f1    = 0.0
    best_path  = output_dir / f"best_{stage_name}.pt"
    history    = []

    print(f"\n{'─'*60}")
    print(f"  {stage_name.upper()}")
    print(f"{'─'*60}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:>2}/{n_epochs} ({elapsed:.0f}s) | "
            f"Train — loss: {train_m['loss']:.4f}  f1: {train_m['f1']:.4f} | "
            f"Val — loss: {val_m['loss']:.4f}  f1: {val_m['f1']:.4f}  "
            f"recall: {val_m['recall']:.4f}  auc: {val_m['auc']:.4f}"
            + (" ← best" if val_m["f1"] > best_f1 else "")
        )

        history.append({
            "epoch": epoch,
            "train": train_m,
            "val":   val_m,
        })

        # Checkpoint best model (primary metric: val F1)
        if val_m["f1"] > best_f1:
            best_f1 = val_m["f1"]
            torch.save(model.state_dict(), best_path)

    print(f"\n  Best val F1 in {stage_name}: {best_f1:.4f}")
    print(f"  Model saved → {best_path}")
    return history


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = cfg.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Augmentation config dict (passed to dataset) ─────────────────────────
    augment_cfg = {
        "AUGMENT_HFLIP":      cfg.AUGMENT_HFLIP,
        "AUGMENT_VFLIP":      cfg.AUGMENT_VFLIP,
        "AUGMENT_ROTATION":   cfg.AUGMENT_ROTATION,
        "AUGMENT_BRIGHTNESS": cfg.AUGMENT_BRIGHTNESS,
        "AUGMENT_CONTRAST":   cfg.AUGMENT_CONTRAST,
        "AUGMENT_SATURATION": cfg.AUGMENT_SATURATION,
    }

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = QQBDataset(cfg.DATA_DIR / "train", split="train", augment_cfg=augment_cfg)
    val_dataset   = QQBDataset(cfg.DATA_DIR / "val",   split="val")

    # ── WeightedRandomSampler (class imbalance) ───────────────────────────────
    # Rebalances sampling so each mini-batch sees a proportional mix.
    # This is applied to the training set only — val is evaluated as-is.
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.BATCH_SIZE,
        sampler     = sampler,           # WeightedRandomSampler replaces shuffle=True
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = (device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = cfg.BATCH_SIZE,
        shuffle     = False,
        num_workers = cfg.NUM_WORKERS,
        pin_memory  = (device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ModelA(fc_size=cfg.FC_SIZE, dropout=cfg.DROPOUT).to(device)
    params = model.count_parameters()
    print(f"\nModel A parameters:")
    print(f"  Backbone : {params['backbone']:,}")
    print(f"  Head     : {params['head']:,}")
    print(f"  Total    : {params['total']:,}")

    # Loss — BCEWithLogitsLoss folds sigmoid in numerically (more stable than
    # applying sigmoid inside the model and then BCELoss separately).
    criterion = nn.BCEWithLogitsLoss()

    history = {}

    # ─────────────────────────────────────────────────────────────────────────
    #  STAGE 1 — Warm-up: frozen backbone
    # ─────────────────────────────────────────────────────────────────────────
    model.freeze_backbone()
    print(f"\nStage 1 trainable params: {model.count_parameters()['trainable']:,}")

    optimizer_s1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = cfg.STAGE1_LR,
        weight_decay = cfg.STAGE1_WEIGHT_DECAY,
    )
    scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s1, T_max=cfg.STAGE1_EPOCHS
    )

    history["stage1"] = run_stage(
        stage_name   = "stage1",
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        optimizer    = optimizer_s1,
        scheduler    = scheduler_s1,
        criterion    = criterion,
        device       = device,
        n_epochs     = cfg.STAGE1_EPOCHS,
        output_dir   = output_dir,
    )

    # Load the best stage 1 weights before entering stage 2
    model.load_state_dict(torch.load(output_dir / "best_stage1.pt"))
    print("\nRestored best Stage 1 weights before Stage 2.")

    # ─────────────────────────────────────────────────────────────────────────
    #  STAGE 2 — Full fine-tune: differential LR
    # ─────────────────────────────────────────────────────────────────────────
    model.unfreeze_backbone()
    print(f"Stage 2 trainable params: {model.count_parameters()['trainable']:,}")

    optimizer_s2 = torch.optim.AdamW(
        model.get_parameter_groups(
            lr_backbone = cfg.STAGE2_LR_BACKBONE,
            lr_head     = cfg.STAGE2_LR_HEAD,
        ),
        weight_decay = cfg.STAGE2_WEIGHT_DECAY,
    )
    scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s2, T_max=cfg.STAGE2_EPOCHS
    )

    history["stage2"] = run_stage(
        stage_name   = "stage2",
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        optimizer    = optimizer_s2,
        scheduler    = scheduler_s2,
        criterion    = criterion,
        device       = device,
        n_epochs     = cfg.STAGE2_EPOCHS,
        output_dir   = output_dir,
    )

    # ── Save training history ─────────────────────────────────────────────────
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved → {history_path}")
    print(f"Final best model       → {output_dir}/best_stage2.pt")


if __name__ == "__main__":
    main()
