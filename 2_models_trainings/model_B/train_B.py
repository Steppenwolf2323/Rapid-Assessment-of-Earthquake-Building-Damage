"""
train_B.py — Model B: ResNet-50 + Shadow Mask (4-channel input)

Identical training strategy to Model A:
    Stage 1 — Warm-up (backbone frozen, 10 epochs, LR 1e-3)
    Stage 2 — Full fine-tune (differential LR, 20 epochs)
              backbone LR 1e-5  |  head LR 1e-4

Class imbalance handled by WeightedRandomSampler.
Primary metric: F1 on the damaged class.
GPU: NUM_WORKERS = 0 (required on Windows).
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

import config_B as cfg
from dataset_B import QQBDatasetB
from model_B import ModelB


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

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images).squeeze(1)
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

    return compute_metrics(all_labels, all_preds, all_probs, total_loss / len(loader))


# ── Validation loop ───────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
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

    return compute_metrics(all_labels, all_preds, all_probs, total_loss / len(loader))


# ── Stage runner ──────────────────────────────────────────────────────────────
def run_stage(stage_name, model, train_loader, val_loader,
              optimizer, scheduler, criterion, device, n_epochs, output_dir) -> list:

    best_f1   = 0.0
    best_path = output_dir / f"best_{stage_name}.pt"
    history   = []

    print(f"\n{'─'*60}")
    print(f"  {stage_name.upper()}")
    print(f"{'─'*60}")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_m = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_m   = evaluate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - t0
        is_best = val_m["f1"] > best_f1

        print(
            f"  Epoch {epoch:>2}/{n_epochs} ({elapsed:.0f}s) | "
            f"Train — loss: {train_m['loss']:.4f}  f1: {train_m['f1']:.4f} | "
            f"Val — loss: {val_m['loss']:.4f}  f1: {val_m['f1']:.4f}  "
            f"recall: {val_m['recall']:.4f}  auc: {val_m['auc']:.4f}"
            + (" ← best" if is_best else "")
        )

        history.append({"epoch": epoch, "train": train_m, "val": val_m})

        if is_best:
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
    train_dataset = QQBDatasetB(
        csv_path         = cfg.TRAIN_CSV,
        split            = "train",
        image_size       = cfg.IMAGE_SIZE,
        augment_cfg      = augment_cfg,
        clahe_clip_limit = cfg.CLAHE_CLIP_LIMIT,
        clahe_tile_grid  = cfg.CLAHE_TILE_GRID,
        min_component_px = cfg.SHADOW_MIN_COMPONENT_PX,
    )
    val_dataset = QQBDatasetB(
        csv_path         = cfg.VAL_CSV,
        split            = "val",
        image_size       = cfg.IMAGE_SIZE,
        clahe_clip_limit = cfg.CLAHE_CLIP_LIMIT,
        clahe_tile_grid  = cfg.CLAHE_TILE_GRID,
        min_component_px = cfg.SHADOW_MIN_COMPONENT_PX,
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
        batch_size = cfg.BATCH_SIZE,
        sampler    = sampler,
        num_workers = cfg.NUM_WORKERS,   # 0 on Windows
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
    model  = ModelB(fc_size=cfg.FC_SIZE, dropout=cfg.DROPOUT).to(device)
    params = model.count_parameters()
    print(f"\nModel B parameters:")
    print(f"  Backbone : {params['backbone']:,}")
    print(f"  Head     : {params['head']:,}")
    print(f"  Total    : {params['total']:,}")

    criterion = nn.BCEWithLogitsLoss()
    history   = {}

    # ── Stage 1: frozen backbone ──────────────────────────────────────────────
    model.freeze_backbone()
    print(f"\nStage 1 trainable params: {model.count_parameters()['trainable']:,}")

    optimizer_s1 = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.STAGE1_LR, weight_decay=cfg.STAGE1_WEIGHT_DECAY,
    )
    scheduler_s1 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s1, T_max=cfg.STAGE1_EPOCHS
    )

    history["stage1"] = run_stage(
        "stage1", model, train_loader, val_loader,
        optimizer_s1, scheduler_s1, criterion, device,
        cfg.STAGE1_EPOCHS, output_dir,
    )

    # Load best stage 1 weights before stage 2
    model.load_state_dict(torch.load(output_dir / "best_stage1.pt"))
    print("\nRestored best Stage 1 weights before Stage 2.")

    # ── Stage 2: full fine-tune ───────────────────────────────────────────────
    model.unfreeze_backbone()
    print(f"Stage 2 trainable params: {model.count_parameters()['trainable']:,}")

    optimizer_s2 = torch.optim.AdamW(
        model.get_parameter_groups(cfg.STAGE2_LR_BACKBONE, cfg.STAGE2_LR_HEAD),
        weight_decay=cfg.STAGE2_WEIGHT_DECAY,
    )
    scheduler_s2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_s2, T_max=cfg.STAGE2_EPOCHS
    )

    history["stage2"] = run_stage(
        "stage2", model, train_loader, val_loader,
        optimizer_s2, scheduler_s2, criterion, device,
        cfg.STAGE2_EPOCHS, output_dir,
    )

    # ── Save history ──────────────────────────────────────────────────────────
    history_path = output_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history → {history_path}")
    print(f"Final best model → {output_dir / 'best_stage2.pt'}")


if __name__ == "__main__":
    main()
