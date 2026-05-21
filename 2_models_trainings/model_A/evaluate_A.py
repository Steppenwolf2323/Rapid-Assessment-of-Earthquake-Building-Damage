"""
evaluate_A.py — Model A evaluation on xBD test set

Place this file in:
    2_models_trainings/model_A/

Saves results to:
    3_experiments/model_A/evaluation_xbd.json

Usage:
    cd 2_models_trainings/model_A
    python evaluate_A.py
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

from model_A import ModelA
import config_A as cfg


# ─── Paths ────────────────────────────────────────────────────────────────────
BASE          = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage")
XBD_TEST_CSV  = BASE / "0_data_preprocessing" / "xbd_dataset" / "xbd_test.csv"
MODEL_WEIGHTS = BASE / "2_models_trainings" / "model_A" / "outputs" / "best_stage2.pt"
OUTPUT_FILE   = BASE / "3_experiments" / "model_A" / "evaluation_xbd.json"

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─── Dataset ──────────────────────────────────────────────────────────────────

class XBDDatasetA(Dataset):
    def __init__(self, csv_path: Path, image_size: int = 224):
        df            = pd.read_csv(csv_path)
        self.samples  = list(zip(df["path"].tolist(), df["label"].tolist()))
        self.disaster = df["disaster"].tolist()
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ])
        n_intact  = sum(1 for _, lbl in self.samples if lbl == 0)
        n_damaged = sum(1 for _, lbl in self.samples if lbl == 1)
        print(f"[xBD test] {len(self.samples)} samples "
              f"(intact: {n_intact}, damaged: {n_damaged})")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32), self.disaster[idx]


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

    model = ModelA(fc_size=cfg.FC_SIZE, dropout=cfg.DROPOUT).to(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
    model.eval()
    print("Model A loaded.\n")

    criterion = nn.BCEWithLogitsLoss()
    dataset   = XBDDatasetA(XBD_TEST_CSV, image_size=cfg.IMAGE_SIZE)
    loader    = DataLoader(
        dataset, batch_size=cfg.BATCH_SIZE,
        shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    all_labels, all_preds, all_probs, all_disasters = [], [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for images, labels, disasters in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images).squeeze(1)
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
        idx      = [i for i, d in enumerate(all_disasters) if d == disaster]
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
        "model":        "A",
        "description":  "Pure end-to-end ML — RGB only, ResNet-50",
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
