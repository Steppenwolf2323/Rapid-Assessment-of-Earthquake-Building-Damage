"""
dataset.py — Model A: Pure End-to-End ML

Loads QQB building patches from disk.

Expected folder structure for each split:
    <split_dir>/
    ├── damaged/   →  label 1
    └── intact/    →  label 0

Transforms applied:
    Training:   Random flip, rotation, color jitter → ToTensor → Normalize
    Val / Test: ToTensor → Normalize  (no augmentation — we want reproducible scores)

Normalization uses ImageNet mean and std, since the backbone was
pretrained on ImageNet. This aligns the input distribution with what
the backbone expects.
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ImageNet statistics — required because ResNet-50 was trained with these
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Supported image extensions
_IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")


class QQBDataset(Dataset):
    """
    Args:
        split_dir:  Path to the split folder (e.g. data/QQB/train)
        split:      One of 'train', 'val', 'test'.
                    Controls whether augmentation is applied.
        augment_cfg: Dict of augmentation parameters (from config.py).
                     Only used when split == 'train'.
    """

    def __init__(self, split_dir: Path | str, split: str = "train", augment_cfg: dict = None):
        self.split_dir = Path(split_dir)
        self.split     = split
        self.samples   = []   # list of (path, label) tuples

        # ── Collect samples ──────────────────────────────────────────────────
        for label, class_name in enumerate(["intact", "damaged"]):
            class_dir = self.split_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Expected class folder not found: {class_dir}\n"
                    f"Make sure your split has 'damaged/' and 'intact/' subfolders."
                )
            for ext in _IMG_EXTENSIONS:
                for img_path in class_dir.glob(f"*{ext}"):
                    self.samples.append((img_path, label))

        if len(self.samples) == 0:
            raise RuntimeError(f"No images found in {self.split_dir}")

        n_intact  = sum(1 for _, lbl in self.samples if lbl == 0)
        n_damaged = sum(1 for _, lbl in self.samples if lbl == 1)
        print(f"[{split}] Loaded {len(self.samples)} images "
              f"(intact: {n_intact}, damaged: {n_damaged}, "
              f"ratio: {n_intact / max(n_damaged, 1):.1f}:1)")

        # ── Transforms ──────────────────────────────────────────────────────
        self.transform = self._build_transforms(split, augment_cfg or {})

    def _build_transforms(self, split: str, cfg: dict) -> transforms.Compose:
        """
        Training:  augmentation → tensor → normalize
        Val/Test:  tensor → normalize  (deterministic)
        """
        to_tensor_and_norm = [
            transforms.ToTensor(),
            transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ]

        if split != "train":
            # No augmentation for val/test — results must be reproducible
            return transforms.Compose(to_tensor_and_norm)

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

        return transforms.Compose(augmentations + to_tensor_and_norm)

    # ── Dataset interface ────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        # Label as float32 for BCEWithLogitsLoss
        return image, torch.tensor(label, dtype=torch.float32)

    # ── Class imbalance ──────────────────────────────────────────────────────
    def get_sample_weights(self) -> torch.Tensor:
        """
        Computes per-sample weights for WeightedRandomSampler.

        With QQB's 23:1 imbalance (intact vs damaged), a standard DataLoader
        would almost never sample a damaged building. WeightedRandomSampler
        rebalances the draw so each mini-batch sees a proportional mix of
        both classes, without duplicating images on disk.

        Weight formula:
            weight(class) = total_samples / (n_classes × n_samples_in_class)
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
