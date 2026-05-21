"""
evaluate_C.py — Model C evaluation on xBD test set

Place this file in:
    2_models_trainings/model_C/

Saves results to:
    3_experiments/model_C/evaluation_xbd.json

Uses per-image sun angles from xBD JSON metadata automatically.

Usage:
    cd 2_models_trainings/model_C
    python evaluate_C.py
"""

import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

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
from dataset_C import get_sun_angles, compute_observed_shadow, compute_expected_shadow
import config_C as cfg


# ─── Paths ────────────────────────────────────────────────────────────────────
BASE          = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage")
XBD_TEST_CSV  = BASE / "0_data_preprocessing" / "xbd_dataset" / "xbd_test.csv"
MODEL_WEIGHTS = BASE / "2_models_trainings" / "model_C" / "outputs" / "best_model.pt"
OUTPUT_FILE   = BASE / "3_experiments" / "model_C" / "evaluation_xbd.json"


# ─── Dataset ──────────────────────────────────────────────────────────────────

class XBDDatasetC(Dataset):
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
        df                     = pd.read_csv(csv_path)
        self.samples           = list(zip(df["path"].tolist(), df["label"].tolist()))
        self.disaster          = df["disaster"].tolist()
        self.image_size        = image_size
        self.default_azimuth   = default_azimuth
        self.default_elevation = default_elevation
        self.coherence_kernel  = coherence_kernel
        self.coherence_sigma   = coherence_sigma
        self.clahe_clip        = clahe_clip
        self.clahe_grid        = clahe_grid
        self.min_component_px  = min_component_px

        n_intact  = sum(1 for _, lbl in self.samples if lbl == 0)
        n_damaged = sum(1 for _, lbl in self.samples if lbl == 1)
        n_meta    = sum(
            1 for path, _ in self.samples
            if (Path(path).parent.parent / "labels" /
                Path(path).with_suffix(".json").name).exists()
        )
        print(f"[xBD test] {len(self.samples)} samples "
              f"(intact: {n_intact}, damaged: {n_damaged})")
        print(f"[xBD test] {n_meta}/{len(self.samples)} images "
              f"with JSON sun angle metadata")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        img_np  = np.array(Image.open(img_path).convert("RGB"))
        img_pil = transforms.Resize(
            (self.image_size, self.image_size)
        )(Image.fromarray(img_np))
        img_np  = np.array(img_pil)

        azimuth, elevation = get_sun_angles(
            img_path, self.default_azimuth, self.default_elevation
        )

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
        )


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(labels, preds, probs, loss) -> dict:
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
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

    all_labels, all_preds, all_probs, all_disasters = [], [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for observed, expected, labels, disasters in loader:
            observed = observed.to(device)
            expected = expected.to(device)
            labels   = labels.to(device)
            logits   = model(observed, expected).squeeze(1)
            total_loss += criterion(logits, labels).item()
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs >= cfg.THRESHOLD).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.long().cpu().numpy().tolist())
            all_disasters.extend(list(disasters))

    avg_loss = total_loss / len(loader)
    overall  = compute_metrics(all_labels, all_preds, all_probs, avg_loss)

    print("Overall results:")
    print(f"  F1:        {overall['f1']}")
    print(f"  Precision: {overall['precision']}")
    print(f"  Recall:    {overall['recall']}")
    print(f"  AUC:       {overall['auc']}")
    print(f"  Loss:      {overall['loss']}")

    per_disaster = {}
    for disaster in sorted(set(all_disasters)):
        idx = [i for i, d in enumerate(all_disasters) if d == disaster]
        per_disaster[disaster] = compute_metrics(
            [all_labels[i] for i in idx],
            [all_preds[i]  for i in idx],
            [all_probs[i]  for i in idx],
            avg_loss,
        )
        print(f"\n  {disaster}:")
        print(f"    F1: {per_disaster[disaster]['f1']}  "
              f"Recall: {per_disaster[disaster]['recall']}  "
              f"AUC: {per_disaster[disaster]['auc']}  "
              f"(n={per_disaster[disaster]['n_samples']})")

    results = {
        "model":        "C",
        "description":  "Physics-guided — dual-branch shadow comparison",
        "test_set":     "xBD",
        "metrics":      overall,
        "per_disaster": per_disaster,
        "per_sample": [
            {
                "path":     dataset.samples[i][0],
                "label":    all_labels[i],
                "pred":     all_preds[i],
                "prob":     round(all_probs[i], 6),
                "disaster": all_disasters[i],
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
