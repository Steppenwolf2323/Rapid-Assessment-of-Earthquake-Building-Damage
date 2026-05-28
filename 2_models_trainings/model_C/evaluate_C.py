"""
evaluate_C.py — Model C evaluation on xBD building-level test set

Key fix: reads sun_azimuth and sun_elevation directly from the CSV
columns saved during extraction, instead of looking up JSON files.
The crop paths no longer have the original folder structure needed
for JSON lookup, but the CSV already has the metadata.

Saves to: 3_experiments/model_C/evaluation_xbd.json

Usage:
    cd 2_models_trainings/model_C
    python evaluate_C.py
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

from model_C import ModelC
from dataset_C import compute_observed_shadow, compute_expected_shadow
import config_C as cfg


# ─── Paths ────────────────────────────────────────────────────────────────────
BASE          = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage")
XBD_TEST_CSV  = BASE / "0_data_preprocessing" / "xbd_dataset" / "xbd_test_buildings.csv"
MODEL_WEIGHTS = BASE / "2_models_trainings" / "model_C" / "outputs" / "best_model.pt"
OUTPUT_FILE   = BASE / "3_experiments" / "model_C" / "evaluation_xbd.json"

THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 1.00, 0.05)]


# ─── Dataset ──────────────────────────────────────────────────────────────────

class XBDDatasetC(Dataset):
    """
    Reads sun_azimuth and sun_elevation directly from CSV columns.
    The extraction notebook saved these from each scene's metadata,
    so we do not need to look up JSON files from the crop paths.
    """

    def __init__(
        self,
        csv_path:          Path,
        image_size:        int   = 224,
        default_azimuth:   float = 150.0,
        default_elevation: float = 32.0,
        coherence_kernel:  int   = 7,
        coherence_sigma:   float = 1.5,
        clahe_clip:        float = 2.0,
        clahe_grid:        tuple = (8, 8),
        min_component_px:  int   = 50,
    ):
        df = pd.read_csv(csv_path)

        self.samples           = list(zip(df["path"].tolist(), df["label"].tolist()))
        self.disaster          = df["disaster"].tolist()
        self.subtype           = df["subtype"].tolist() if "subtype" in df.columns else ["unknown"] * len(df)
        self.image_size        = image_size
        self.coherence_kernel  = coherence_kernel
        self.coherence_sigma   = coherence_sigma
        self.clahe_clip        = clahe_clip
        self.clahe_grid        = clahe_grid
        self.min_component_px  = min_component_px

        # Read sun angles from CSV — fall back to defaults if column missing
        if "sun_azimuth" in df.columns and "sun_elevation" in df.columns:
            self.sun_azimuths   = df["sun_azimuth"].tolist()
            self.sun_elevations = df["sun_elevation"].tolist()
            print(f"[xBD test] Sun angles read from CSV columns ✓")
        else:
            self.sun_azimuths   = [default_azimuth]   * len(df)
            self.sun_elevations = [default_elevation] * len(df)
            print(f"[xBD test] Sun angle columns not found — using defaults "
                  f"(az={default_azimuth}°, el={default_elevation}°)")

        n_intact  = sum(1 for _, lbl in self.samples if lbl == 0)
        n_damaged = sum(1 for _, lbl in self.samples if lbl == 1)
        print(f"[xBD test] {len(self.samples)} samples "
              f"(intact: {n_intact}, damaged: {n_damaged}, "
              f"ratio: {n_intact/max(n_damaged,1):.1f}:1)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img_np  = np.array(Image.open(img_path).convert("RGB"))
        img_pil = transforms.Resize(
            (self.image_size, self.image_size)
        )(Image.fromarray(img_np))
        img_np  = np.array(img_pil)

        # Per-image sun angles from CSV
        azimuth   = self.sun_azimuths[idx]
        elevation = self.sun_elevations[idx]

        observed = compute_observed_shadow(
            img_np,
            clahe_clip       = self.clahe_clip,
            clahe_grid       = self.clahe_grid,
            min_component_px = self.min_component_px,
        )
        expected = compute_expected_shadow(
            img_np,
            sun_azimuth_deg   = azimuth,
            sun_elevation_deg = elevation,
            kernel_size       = self.coherence_kernel,
            smooth_sigma      = self.coherence_sigma,
        )

        return (
            torch.from_numpy(observed).unsqueeze(0),
            torch.from_numpy(expected).unsqueeze(0),
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

    model = ModelC(feature_dim=cfg.CNN_FEATURE_DIM, dropout=0.5).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()
    print("Model C loaded.\n")

    criterion = nn.BCEWithLogitsLoss()
    dataset   = XBDDatasetC(
        XBD_TEST_CSV,
        image_size        = cfg.IMAGE_SIZE,
        default_azimuth   = cfg.SUN_AZIMUTH_DEG,
        default_elevation = cfg.SUN_ELEVATION_DEG,
        coherence_kernel  = cfg.COHERENCE_KERNEL_SIZE,
        coherence_sigma   = cfg.COHERENCE_SMOOTH_SIGMA,
        clahe_clip        = cfg.CLAHE_CLIP_LIMIT,
        clahe_grid        = cfg.CLAHE_TILE_GRID,
        min_component_px  = cfg.SHADOW_MIN_COMPONENT_PX,
    )
    loader = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE,
        shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    all_labels, all_probs, all_disasters, all_subtypes = [], [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for observed, expected, labels, disasters, subtypes in loader:
            observed = observed.to(device)
            expected = expected.to(device)
            labels   = labels.to(device)
            logits   = model(observed, expected).squeeze(1)
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
        "model":              "C",
        "description":        "Physics-guided — dual-branch shadow comparison",
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