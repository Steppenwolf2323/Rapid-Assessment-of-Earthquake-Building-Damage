"""
train_C.py — Model C: Physics-Guided Dual-Branch Shadow Comparison

Single training stage, 30 epochs, everything from scratch.
No two-stage strategy needed: no pretrained weights to protect.

The DataLoader returns three items per batch:
    observed:  [B, 1, 224, 224] — observed shadow mask
    expected:  [B, 1, 224, 224] — expected shadow coherence map
    labels:    [B]               — 0=intact, 1=damaged

Both tensors are passed separately to the model's two branches.
"""

import sys
import json
import random
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

import config_C as cfg
from dataset_C import QQBDatasetC
from model_C import ModelC


# ── Reproducibility ──────────────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(labels, preds, probs, loss) -> dict:
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


# ── Training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, criterion, device) -> dict:
    model.train()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for observed, expected, labels in loader:
        observed = observed.to(device)
        expected = expected.to(device)
        labels   = labels.to(device)

        optimizer.zero_grad()
        logits = model(observed, expected).squeeze(1)
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
        all_labels, all_preds, all_probs, total_loss / len(loader)
    )


# ── Validation loop ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for observed, expected, labels in loader:
        observed = observed.to(device)
        expected = expected.to(device)
        labels   = labels.to(device)

        logits = model(observed, expected).squeeze(1)
        loss   = criterion(logits, labels)
        total_loss += loss.item()

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= cfg.THRESHOLD).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.long().cpu().numpy().tolist())

    return compute_metrics(
        all_labels, all_preds, all_probs, total_loss / len(loader)
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = cfg.OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Augmentation config ───────────────────────────────────────────────────
    augment_cfg = {
        "AUGMENT_HFLIP":      cfg.AUGMENT_HFLIP,
        "AUGMENT_VFLIP":      cfg.AUGMENT_VFLIP,
        "AUGMENT_ROTATION":   cfg.AUGMENT_ROTATION,
        "AUGMENT_BRIGHTNESS": cfg.AUGMENT_BRIGHTNESS,
        "AUGMENT_CONTRAST":   cfg.AUGMENT_CONTRAST,
        "AUGMENT_SATURATION": cfg.AUGMENT_SATURATION,
    }

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_dataset = QQBDatasetC(
        csv_path          = cfg.TRAIN_CSV,
        split             = "train",
        image_size        = cfg.IMAGE_SIZE,
        augment_cfg       = augment_cfg,
        default_azimuth   = cfg.SUN_AZIMUTH_DEG,
        default_elevation = cfg.SUN_ELEVATION_DEG,
        coherence_kernel  = cfg.COHERENCE_KERNEL_SIZE,
        coherence_sigma   = cfg.COHERENCE_SMOOTH_SIGMA,
        clahe_clip_limit  = cfg.CLAHE_CLIP_LIMIT,
        clahe_tile_grid   = cfg.CLAHE_TILE_GRID,
        min_component_px  = cfg.SHADOW_MIN_COMPONENT_PX,
    )
    val_dataset = QQBDatasetC(
        csv_path          = cfg.VAL_CSV,
        split             = "val",
        image_size        = cfg.IMAGE_SIZE,
        default_azimuth   = cfg.SUN_AZIMUTH_DEG,
        default_elevation = cfg.SUN_ELEVATION_DEG,
        coherence_kernel  = cfg.COHERENCE_KERNEL_SIZE,
        coherence_sigma   = cfg.COHERENCE_SMOOTH_SIGMA,
        clahe_clip_limit  = cfg.CLAHE_CLIP_LIMIT,
        clahe_tile_grid   = cfg.CLAHE_TILE_GRID,
        min_component_px  = cfg.SHADOW_MIN_COMPONENT_PX,
    )

    # ── WeightedRandomSampler ─────────────────────────────────────────────────
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights     = sample_weights,
        num_samples = len(sample_weights),
        replacement = True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size  = cfg.BATCH_SIZE,
        sampler     = sampler,
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
    model  = ModelC(
        feature_dim = cfg.CNN_FEATURE_DIM,
        dropout     = 0.5,
    ).to(device)

    params = model.count_parameters()
    print(f"\nModel C parameters:")
    print(f"  Branch observed : {params['branch_observed']:,}")
    print(f"  Branch expected : {params['branch_expected']:,}")
    print(f"  Head            : {params['head']:,}")
    print(f"  Total           : {params['total']:,}")

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.LR,
        weight_decay = cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_f1   = 0.0
    best_path = output_dir / "best_model.pt"
    history   = []

    print(f"\n{'─'*60}")
    print(f"  TRAINING — {cfg.EPOCHS} epochs")
    print(f"{'─'*60}")

    for epoch in range(1, cfg.EPOCHS + 1):
        t0 = time.time()

        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0
        is_best = val_m["f1"] > best_f1

        print(
            f"  Epoch {epoch:>2}/{cfg.EPOCHS} ({elapsed:.0f}s) | "
            f"Train — loss: {train_m['loss']:.4f}  f1: {train_m['f1']:.4f} | "
            f"Val — loss: {val_m['loss']:.4f}  f1: {val_m['f1']:.4f}  "
            f"recall: {val_m['recall']:.4f}  auc: {val_m['auc']:.4f}"
            + (" ← best" if is_best else "")
        )

        history.append({"epoch": epoch, "train": train_m, "val": val_m})

        if is_best:
            best_f1 = val_m["f1"]
            torch.save(model.state_dict(), best_path)

    print(f"\n  Best val F1: {best_f1:.4f}")
    print(f"  Model saved → {best_path}")

    # ── Save history ──────────────────────────────────────────────────────────
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history → {history_path}")


if __name__ == "__main__":
    main()
