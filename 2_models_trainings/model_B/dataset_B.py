"""
dataset_B.py — Model B: ResNet-50 + Shadow Mask

Extends the Model A dataset with an on-the-fly shadow mask pipeline.
The mask is derived entirely from the RGB image — no extra files needed —
so it works on any optical dataset including xBD at test time.

Shadow mask pipeline (Sanskriti-inspired):
    1. Convert RGB → grayscale
    2. Apply CLAHE to enhance local contrast
    3. Apply Otsu thresholding → raw binary mask
    4. Morphological opening (remove thin noise)
    5. Morphological closing (fill small holes)
    6. Remove connected components smaller than SHADOW_MIN_COMPONENT_PX
    7. Normalise mask to [0, 1]

Output tensor: [4, 224, 224]
    Channels 0-2: RGB normalised with ImageNet mean/std
    Channel 3:    Shadow mask normalised to [0, 1]

Augmentation is applied to the PIL image BEFORE mask generation,
so the mask always corresponds to the (already augmented) RGB image.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


# ── Shadow mask generation ────────────────────────────────────────────────────

def generate_shadow_mask(
    img_np:            np.ndarray,
    clahe_clip_limit:  float = 2.0,
    clahe_tile_grid:   tuple = (8, 8),
    min_component_px:  int   = 50,
) -> np.ndarray:
    """
    Generates a binary shadow mask from an RGB image (uint8, H×W×3).

    Steps:
        1. Convert to grayscale
        2. CLAHE — enhance local contrast to make shadows more distinct
        3. Otsu threshold — auto-select threshold to separate shadow/non-shadow
        4. Morphological opening — removes thin noise bridges
        5. Morphological closing — fills small holes inside shadow regions
        6. Remove small components — eliminates noise blobs below threshold
        7. Normalise to [0.0, 1.0]

    Returns:
        mask: np.ndarray of shape (H, W), float32, values in [0, 1]
    """
    # Step 1: Grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # Step 2: CLAHE
    clahe = cv2.createCLAHE(
        clipLimit   = clahe_clip_limit,
        tileGridSize = clahe_tile_grid,
    )
    enhanced = clahe.apply(gray)

    # Step 3: Otsu thresholding — shadows are dark, so we invert
    # THRESH_BINARY_INV marks dark pixels (shadows) as white (255)
    _, raw_mask = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    # Step 4: Morphological opening (erosion then dilation)
    # Removes thin noise bridges and isolated noise pixels
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened  = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    # Step 5: Morphological closing (dilation then erosion)
    # Fills small holes inside genuine shadow regions
    cleaned = cv2.morphologyEx(opened,   cv2.MORPH_CLOSE, kernel, iterations=1)

    # Step 6: Remove connected components smaller than min_component_px
    # This eliminates dark noise blobs that survived morphological cleaning
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned, connectivity=8
    )
    filtered = np.zeros_like(cleaned)
    for label_idx in range(1, num_labels):   # skip background (label 0)
        area = stats[label_idx, cv2.CC_STAT_AREA]
        if area >= min_component_px:
            filtered[labels == label_idx] = 255

    # Step 7: Normalise to [0, 1]
    mask = filtered.astype(np.float32) / 255.0
    return mask


# ── Dataset ───────────────────────────────────────────────────────────────────

class QQBDatasetB(Dataset):
    """
    Args:
        csv_path:          Path to train.csv or val.csv
        split:             'train', 'val', or 'test'
        image_size:        Resize target (default 224)
        augment_cfg:       Dict of augmentation flags from config_B.py
        clahe_clip_limit:  CLAHE clip limit for shadow mask generation
        clahe_tile_grid:   CLAHE tile grid size
        min_component_px:  Min connected component size to keep in mask
    """

    def __init__(
        self,
        csv_path:         Path | str,
        split:            str   = "train",
        image_size:       int   = 224,
        augment_cfg:      dict  = None,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid:  tuple = (8, 8),
        min_component_px: int   = 50,
    ):
        self.split            = split
        self.image_size       = image_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid  = clahe_tile_grid
        self.min_component_px = min_component_px

        # ── Load CSV ─────────────────────────────────────────────────────────
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "path" not in df.columns or "label" not in df.columns:
            raise ValueError(f"CSV must have 'path' and 'label' columns.")

        self.samples = list(zip(df["path"].tolist(), df["label"].tolist()))

        n_intact  = sum(1 for _, lbl in self.samples if lbl == 0)
        n_damaged = sum(1 for _, lbl in self.samples if lbl == 1)
        print(f"[{split}] Loaded {len(self.samples)} samples "
              f"(intact: {n_intact}, damaged: {n_damaged}, "
              f"ratio: {n_intact / max(n_damaged, 1):.1f}:1)")

        # ── Spatial augmentation transforms (applied to PIL before masking) ──
        self.spatial_aug = self._build_spatial_aug(split, augment_cfg or {})

        # ── Colour augmentation (applied to PIL after spatial aug) ───────────
        self.colour_aug = self._build_colour_aug(split, augment_cfg or {})

    def _build_spatial_aug(self, split: str, cfg: dict):
        """
        Spatial augmentations (flip, rotation) applied BEFORE mask generation.
        The mask is computed AFTER these, so it always matches the image geometry.
        """
        if split != "train":
            return None

        ops = [transforms.Resize((self.image_size, self.image_size))]

        if cfg.get("AUGMENT_HFLIP", True):
            ops.append(transforms.RandomHorizontalFlip())
        if cfg.get("AUGMENT_VFLIP", True):
            ops.append(transforms.RandomVerticalFlip())

        rotation = cfg.get("AUGMENT_ROTATION", 90)
        if rotation > 0:
            ops.append(transforms.RandomRotation(rotation))

        return transforms.Compose(ops)

    def _build_colour_aug(self, split: str, cfg: dict):
        """
        Colour augmentation (brightness, contrast, saturation) applied to
        the RGB image only — not to the mask.
        """
        if split != "train":
            return None

        brightness = cfg.get("AUGMENT_BRIGHTNESS", 0.2)
        contrast   = cfg.get("AUGMENT_CONTRAST",   0.2)
        saturation = cfg.get("AUGMENT_SATURATION", 0.2)

        if any([brightness, contrast, saturation]):
            return transforms.ColorJitter(
                brightness=brightness,
                contrast=contrast,
                saturation=saturation,
            )
        return None

    def _to_tensor_normalised(self, img_pil: Image.Image) -> torch.Tensor:
        """Converts PIL RGB image to normalised tensor [3, H, W]."""
        t = transforms.ToTensor()(img_pil)
        t = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)(t)
        return t

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label = self.samples[idx]

        # ── Load image ───────────────────────────────────────────────────────
        npy_path = Path(img_path).with_suffix(".npy")
        img_np   = np.load(npy_path)                          # (H, W, 3) uint8
        img_pil  = Image.fromarray(img_np, mode="RGB")

        # ── Resize (always, both splits) ─────────────────────────────────────
        img_pil = transforms.Resize((self.image_size, self.image_size))(img_pil)

        # ── Spatial augmentation (training only) ─────────────────────────────
        if self.spatial_aug is not None:
            img_pil = self.spatial_aug(img_pil)

        # ── Generate shadow mask from the (possibly augmented) RGB image ─────
        # Must happen AFTER spatial augmentation so mask geometry matches image
        img_np_aug = np.array(img_pil)                        # (H, W, 3) uint8
        mask = generate_shadow_mask(
            img_np_aug,
            clahe_clip_limit = self.clahe_clip_limit,
            clahe_tile_grid  = self.clahe_tile_grid,
            min_component_px = self.min_component_px,
        )                                                     # (H, W) float32

        # ── Colour augmentation on RGB only (training only) ──────────────────
        if self.colour_aug is not None:
            img_pil = self.colour_aug(img_pil)

        # ── Normalise RGB → tensor [3, H, W] ─────────────────────────────────
        rgb_tensor  = self._to_tensor_normalised(img_pil)     # [3, H, W]

        # ── Shadow mask → tensor [1, H, W] ───────────────────────────────────
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)     # [1, H, W]

        # ── Concatenate → [4, H, W] ───────────────────────────────────────────
        image_4ch = torch.cat([rgb_tensor, mask_tensor], dim=0)  # [4, H, W]

        return image_4ch, torch.tensor(label, dtype=torch.float32)

    # ── Class imbalance weights ───────────────────────────────────────────────
    def get_sample_weights(self) -> torch.Tensor:
        labels    = [lbl for _, lbl in self.samples]
        n_intact  = labels.count(0)
        n_damaged = labels.count(1)
        n_total   = len(labels)
        w_intact  = n_total / (2 * n_intact)
        w_damaged = n_total / (2 * n_damaged)
        weights   = [w_intact if lbl == 0 else w_damaged for lbl in labels]
        return torch.tensor(weights, dtype=torch.float32)
