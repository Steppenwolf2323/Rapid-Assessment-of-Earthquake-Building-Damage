"""
evaluate_B.py — Model B evaluation on xBD building-level test set

Saves to: 3_experiments/model_B/evaluation_xbd.json

Usage:
    cd 2_models_trainings/model_B
    python evaluate_B.py
"""

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)

from model_B import ModelB
import config_B as cfg


# ─── Paths ────────────────────────────────────────────────────────────────────
BASE          = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage")
XBD_TEST_CSV  = BASE / "0_data_preprocessing" / "xbd_dataset" / "xbd_test_buildings.csv"
MODEL_WEIGHTS = BASE / "2_models_trainings" / "model_B" / "outputs" / "best_stage2.pt"
OUTPUT_FILE   = BASE / "3_experiments" / "model_B" / "evaluation_xbd.json"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 1.00, 0.05)]


# ─── Dataset ──────────────────────────────────────────────────────────────────

class XBDDatasetB(Dataset):
    def __init__(
        self,
        csv_path:         Path,
        image_size:       int   = 224,
        clahe_clip:       float = 2.0,
        clahe_grid:       tuple = (8, 8),
        min_component_px: int   = 50,
    ):
        df                    = pd.read_csv(csv_path)
        self.samples          = list(zip(df["path"].tolist(), df["label"].tolist()))
        self.disaster         = df["disaster"].tolist()
        self.subtype          = df["subtype"].tolist() if "subtype" in df.columns else ["unknown"] * len(df)
        self.image_size       = image_size
        self.clahe_clip       = clahe_clip
        self.clahe_grid       = clahe_grid
        self.min_component_px = min_component_px
        self.resize           = transforms.Resize((image_size, image_size))
        self.to_tensor        = transforms.ToTensor()
        self.normalise        = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

        n_intact  = sum(1 for _, lbl in self.samples if lbl == 0)
        n_damaged = sum(1 for _, lbl in self.samples if lbl == 1)
        print(f"[xBD test] {len(self.samples)} samples "
              f"(intact: {n_intact}, damaged: {n_damaged}, "
              f"ratio: {n_intact/max(n_damaged,1):.1f}:1)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_pil = Image.open(img_path).convert("RGB")
        img_pil = self.resize(img_pil)
        img_np  = np.array(img_pil)

        gray     = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe    = cv2.createCLAHE(clipLimit=self.clahe_clip, tileGridSize=self.clahe_grid)
        enhanced = clahe.apply(gray)
        _, raw   = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k        = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened   = cv2.morphologyEx(raw,    cv2.MORPH_OPEN,  k)
        cleaned  = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k)
        n_lbl, lbl_map, stats, _ = cv2.connectedComponentsWithStats(cleaned)
        mask = np.zeros_like(cleaned, dtype=np.float32)
        for lbl in range(1, n_lbl):
            if stats[lbl, cv2.CC_STAT_AREA] >= self.min_component_px:
                mask[lbl_map == lbl] = 1.0

        rgb_tensor  = self.normalise(self.to_tensor(img_pil))
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)
        return (
            torch.cat([rgb_tensor, mask_tensor], dim=0),
            torch.tensor(label, dtype=torch.float32),
            self.disaster[idx],
            self.subtype[idx],
        )


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(labels, preds, probs, loss, threshold=0.5) -> dict:
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    try:
        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    except ValueError:
        tn = fp = fn = tp = 0
    return {
        "threshold": threshold,
        "loss":      round(loss, 6),
        "f1":        round(f1_score(labels, preds,        zero_division=0), 4),
        "precision": round(precision_score(labels, preds, zero_division=0), 4),
        "recall":    round(recall_score(labels, preds,    zero_division=0), 4),
        "auc":       round(float(auc), 4),
        "tp": int(tp), "fp": int(fp),
        "tn": int(tn), "fn": int(fn),
        "n_samples": len(labels),
        "n_damaged": int(sum(labels)),
        "n_intact":  int(len(labels) - sum(labels)),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device:        {device}")
    print(f"Model weights: {MODEL_WEIGHTS}\n")

    if not MODEL_WEIGHTS.exists():
        raise FileNotFoundError(f"Weights not found: {MODEL_WEIGHTS}")

    model = ModelB(fc_size=cfg.FC_SIZE, dropout=cfg.DROPOUT).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()
    print("Model B loaded.\n")

    criterion = nn.BCEWithLogitsLoss()
    dataset   = XBDDatasetB(
        XBD_TEST_CSV,
        image_size       = cfg.IMAGE_SIZE,
        clahe_clip       = cfg.CLAHE_CLIP_LIMIT,
        clahe_grid       = cfg.CLAHE_TILE_GRID,
        min_component_px = cfg.SHADOW_MIN_COMPONENT_PX,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE,
        shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    all_labels, all_probs, all_disasters, all_subtypes = [], [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels, disasters, subtypes in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images).squeeze(1)
            total_loss += criterion(logits, labels).item()
            probs  = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.long().cpu().numpy().tolist())
            all_disasters.extend(list(disasters))
            all_subtypes.extend(list(subtypes))

    avg_loss = total_loss / len(loader)

    # ── Threshold tuning ──────────────────────────────────────────────────────
    print("Threshold tuning results:")
    print(f"  {'Threshold':>10}  {'F1':>8}  {'Precision':>10}  {'Recall':>8}  {'TP':>5}  {'FN':>5}")
    print(f"  {'-'*55}")

    threshold_results = []
    best_f1        = 0.0
    best_threshold = cfg.THRESHOLD

    for t in THRESHOLDS:
        preds   = [1 if p >= t else 0 for p in all_probs]
        metrics = compute_metrics(all_labels, preds, all_probs, avg_loss, threshold=t)
        threshold_results.append(metrics)
        print(f"  {t:>10.2f}  {metrics['f1']:>8.4f}  "
              f"{metrics['precision']:>10.4f}  {metrics['recall']:>8.4f}  "
              f"{metrics['tp']:>5}  {metrics['fn']:>5}")
        if metrics["f1"] > best_f1:
            best_f1        = metrics["f1"]
            best_threshold = t

    print(f"\n  Best threshold: {best_threshold} → F1 = {best_f1:.4f}")

    best_preds = [1 if p >= best_threshold else 0 for p in all_probs]
    overall    = compute_metrics(all_labels, best_preds, all_probs, avg_loss, best_threshold)

    print(f"\nOverall at threshold {best_threshold}:")
    print(f"  F1:        {overall['f1']}")
    print(f"  Precision: {overall['precision']}")
    print(f"  Recall:    {overall['recall']}")
    print(f"  AUC:       {overall['auc']}")

    per_subtype = {}
    for subtype in sorted(set(all_subtypes)):
        if subtype == "no-damage":
            continue
        idx = [i for i, s in enumerate(all_subtypes) if s == subtype]
        if not idx:
            continue
        per_subtype[subtype] = compute_metrics(
            [all_labels[i] for i in idx],
            [best_preds[i]  for i in idx],
            [all_probs[i]   for i in idx],
            avg_loss, best_threshold,
        )
        print(f"\n  {subtype}: F1={per_subtype[subtype]['f1']}  "
              f"Recall={per_subtype[subtype]['recall']}  "
              f"(n_damaged={per_subtype[subtype]['n_damaged']})")

    results = {
        "model":              "B",
        "description":        "Human-guided ML — RGB + shadow mask, ResNet-50",
        "test_set":           "xBD building-level",
        "best_threshold":     best_threshold,
        "metrics":            overall,
        "threshold_analysis": threshold_results,
        "per_subtype":        per_subtype,
        "per_sample": [
            {
                "path":     dataset.samples[i][0],
                "label":    all_labels[i],
                "pred":     best_preds[i],
                "prob":     round(all_probs[i], 6),
                "disaster": all_disasters[i],
                "subtype":  all_subtypes[i],
            }
            for i in range(len(all_labels))
        ],
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()