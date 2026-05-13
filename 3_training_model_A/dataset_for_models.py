"""
dataset.py — Model A: Pure End-to-End ML

Loads QQB building patches from a CSV file produced by the data pipeline.

CSV format (two files: train.csv and val.csv):
    path,label
    C:\...\building_001_opt.mat,0
    C:\...\building_002_opt.mat,1

Each .mat file is MATLAB v7.3 (HDF5 format), containing:
    key  : 'x3'
    shape: (3, H, W)   ← channels first, uint8
    We transpose to (H, W, 3) before passing to PIL.

Images are NOT 224×224 on disk (they are ~100×99).
Resize to 224×224 is applied in the transform pipeline.

Labels: 0 = intact, 1 = damaged

Transforms applied:
    Training:   Resize → flip → rotation → color jitter → tensor → normalize
    Val / Test: Resize → tensor → normalize  (no randomness)
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ImageNet statistics — backbone was pretrained with these
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def load_mat_image(path: str) -> np.ndarray:
    npy_path = Path(path).with_suffix(".npy")
    return np.load(npy_path)


class QQBDataset(Dataset):
    """
    Args:
        csv_path:    Path to the CSV file (train.csv or val.csv)
        split:       One of 'train', 'val', 'test'.
                     Controls whether augmentation is applied.
        image_size:  Target size for resizing (default 224).
        augment_cfg: Dict of augmentation parameters from config.py.
                     Only used when split == 'train'.
    """

    def __init__(
        self,
        csv_path:    Path | str,
        split:       str  = "train",
        image_size:  int  = 224,
        augment_cfg: dict = None,
    ):
        self.csv_path   = Path(csv_path)
        self.split      = split
        self.image_size = image_size

        # ── Load CSV ─────────────────────────────────────────────────────────
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        if "path" not in df.columns or "label" not in df.columns:
            raise ValueError(
                f"CSV must have 'path' and 'label' columns. "
                f"Found: {list(df.columns)}"
            )

        self.samples = list(zip(df["path"].tolist(), df["label"].tolist()))

        n_intact  = sum(1 for _, lbl in self.samples if lbl == 0)
        n_damaged = sum(1 for _, lbl in self.samples if lbl == 1)
        print(f"[{split}] Loaded {len(self.samples)} samples "
              f"(intact: {n_intact}, damaged: {n_damaged}, "
              f"ratio: {n_intact / max(n_damaged, 1):.1f}:1)")

        # ── Transforms ───────────────────────────────────────────────────────
        self.transform = self._build_transforms(split, augment_cfg or {})

    def _build_transforms(self, split: str, cfg: dict) -> transforms.Compose:
        """
        Training:  resize → augmentation → tensor → normalize
        Val/Test:  resize → tensor → normalize  (deterministic)

        Resize is always applied because images on disk are ~100×99,
        not 224×224. The model expects 224×224.
        """
        resize = [transforms.Resize((self.image_size, self.image_size))]

        to_tensor_and_norm = [
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]

        if split != "train":
            return transforms.Compose(resize + to_tensor_and_norm)

        augmentations = []

        if cfg.get("AUGMENT_HFLIP", True):
            augmentations.append(transforms.RandomHorizontalFlip())

        if cfg.get("AUGMENT_VFLIP", True):
            augmentations.append(transforms.RandomVerticalFlip())

        rotation = cfg.get("AUGMENT_ROTATION", 90)
        if rotation > 0:
            augmentations.append(transforms.RandomRotation(rotation))

        brightness = cfg.get("AUGMENT_BRIGHTNESS", 0.2)
        contrast   = cfg.get("AUGMENT_CONTRAST",   0.2)
        saturation = cfg.get("AUGMENT_SATURATION", 0.2)
        if any([brightness, contrast, saturation]):
            augmentations.append(transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
            ))

        return transforms.Compose(resize + augmentations + to_tensor_and_norm)

    # ── Dataset interface ────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label = self.samples[idx]

        # Load .mat → numpy (H, W, 3) uint8 → PIL Image
        img_np = load_mat_image(img_path)
        image  = Image.fromarray(img_np, mode="RGB")

        image = self.transform(image)

        # Label as float32 for BCEWithLogitsLoss
        return image, torch.tensor(label, dtype=torch.float32)

    # ── Class imbalance ──────────────────────────────────────────────────────
    def get_sample_weights(self) -> torch.Tensor:
        """
        Per-sample weights for WeightedRandomSampler.
        Damaged images get a ~23× higher weight so the DataLoader
        draws both classes at roughly equal frequency.

        weight(class) = total / (n_classes × n_samples_in_class)
        """
        labels    = [lbl for _, lbl in self.samples]
        n_intact  = labels.count(0)
        n_damaged = labels.count(1)
        n_total   = len(labels)
        n_classes = 2

        w_intact  = n_total / (n_classes * n_intact)
        w_damaged = n_total / (n_classes * n_damaged)

        weights = [
            w_intact if lbl == 0 else w_damaged
            for lbl in labels
        ]
        return torch.tensor(weights, dtype=torch.float32)