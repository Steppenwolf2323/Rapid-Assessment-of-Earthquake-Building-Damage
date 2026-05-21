"""
dataset_C.py — Model C: Physics-Guided Dual-Branch Shadow Comparison

Produces two single-channel maps per image:
    observed_mask:  binary shadow mask via CLAHE + Otsu (same as Model B)
    expected_map:   directional coherence map from sun geometry

Sun angle strategy (adaptive):
    If a JSON metadata file exists alongside the image (xBD format):
        → reads sun_azimuth and sun_elevation per image from metadata
        → each image uses its own precise acquisition sun angles
    If no JSON exists (QQB format):
        → falls back to manually configured values in config_C.py
        → all images use the same fixed sun angles

JSON path convention (xBD):
    image: .../train/images/mexico-earthquake_00000055_post_disaster.png
    json:  .../train/labels/mexico-earthquake_00000055_post_disaster.json

No RGB is returned. The model reasons purely about shadows.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ── Sun angle helpers ─────────────────────────────────────────────────────────

def get_sun_angles(
    image_path:        str,
    default_azimuth:   float,
    default_elevation: float,
) -> tuple[float, float]:
    """
    Returns (sun_azimuth, sun_elevation) for a given image.

    If a JSON metadata file exists in the sibling 'labels/' folder
    (xBD convention), reads the precise acquisition sun angles from it.
    Otherwise falls back to the provided default values (QQB convention).

    Args:
        image_path:        Full path to the image file
        default_azimuth:   Fallback sun azimuth (degrees from North)
        default_elevation: Fallback sun elevation (degrees above horizon)

    Returns:
        (azimuth, elevation) in degrees
    """
    p         = Path(image_path)
    json_path = p.parent.parent / "labels" / p.with_suffix(".json").name

    if json_path.exists():
        with open(json_path) as f:
            meta = json.load(f).get("metadata", {})
        azimuth   = meta.get("sun_azimuth",   default_azimuth)
        elevation = meta.get("sun_elevation",  default_elevation)
        return float(azimuth), float(elevation)

    return default_azimuth, default_elevation


# ── Shadow map generation ─────────────────────────────────────────────────────

def compute_observed_shadow(
    img_np:           np.ndarray,
    clahe_clip:       float = 2.0,
    clahe_grid:       tuple = (8, 8),
    min_component_px: int   = 50,
) -> np.ndarray:
    """
    Binary shadow mask from RGB image. Identical to Model B pipeline.
    Returns float32 (H, W) in [0, 1].
    """
    gray     = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)
    enhanced = clahe.apply(gray)

    _, raw = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )

    k       = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened  = cv2.morphologyEx(raw,    cv2.MORPH_OPEN,  k)
    cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k)

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    mask = np.zeros_like(cleaned, dtype=np.float32)
    for lbl in range(1, n_labels):
        if stats[lbl, cv2.CC_STAT_AREA] >= min_component_px:
            mask[labels == lbl] = 1.0

    return mask


def compute_expected_shadow(
    img_np:            np.ndarray,
    sun_azimuth_deg:   float = 150.0,
    sun_elevation_deg: float = 32.0,
    kernel_size:       int   = 7,
    smooth_sigma:      float = 1.5,
) -> np.ndarray:
    """
    Directional coherence map from sun geometry.

    An intact building has a clean vertical edge casting a shadow
    in the predicted direction → high coherence response.
    Collapsed rubble has no consistent vertical edge → low response.

    Returns float32 (H, W) in [0, 1].
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Shadows fall opposite to sun direction
    shadow_azimuth = (sun_azimuth_deg + 180.0) % 360.0
    shadow_rad     = np.deg2rad(shadow_azimuth)
    shadow_dx      = np.sin(shadow_rad)
    shadow_dy      = np.cos(shadow_rad)

    ksize  = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)

    directional = np.abs(grad_x * shadow_dx + grad_y * shadow_dy)
    magnitude   = np.sqrt(grad_x**2 + grad_y**2) + 1e-8
    coherence   = directional / magnitude

    if smooth_sigma > 0:
        coherence = cv2.GaussianBlur(
            coherence, (0, 0),
            sigmaX=smooth_sigma, sigmaY=smooth_sigma,
        )

    cmin, cmax = coherence.min(), coherence.max()
    if cmax - cmin > 1e-6:
        coherence = (coherence - cmin) / (cmax - cmin)
    else:
        coherence = np.zeros_like(coherence)

    return coherence.astype(np.float32)


# ── Dataset ───────────────────────────────────────────────────────────────────

class QQBDatasetC(Dataset):
    """
    Dual-branch shadow dataset.

    Returns per sample:
        observed:  [1, H, W] float32  — observed shadow mask
        expected:  [1, H, W] float32  — expected shadow coherence map
        label:     float32 scalar     — 0=intact, 1=damaged

    Sun angles are resolved per-image:
        xBD images  → read from JSON metadata (precise, per-acquisition)
        QQB images  → use default values from config (fixed approximation)
    """

    def __init__(
        self,
        csv_path:           Path | str,
        split:              str   = "train",
        image_size:         int   = 224,
        augment_cfg:        dict  = None,
        default_azimuth:    float = 150.0,
        default_elevation:  float = 32.0,
        coherence_kernel:   int   = 7,
        coherence_sigma:    float = 1.5,
        clahe_clip_limit:   float = 2.0,
        clahe_tile_grid:    tuple = (8, 8),
        min_component_px:   int   = 50,
    ):
        self.split             = split
        self.image_size        = image_size
        self.default_azimuth   = default_azimuth
        self.default_elevation = default_elevation
        self.coherence_kernel  = coherence_kernel
        self.coherence_sigma   = coherence_sigma
        self.clahe_clip_limit  = clahe_clip_limit
        self.clahe_tile_grid   = clahe_tile_grid
        self.min_component_px  = min_component_px

        # ── Load CSV ─────────────────────────────────────────────────────────
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if "path" not in df.columns or "label" not in df.columns:
            raise ValueError("CSV must have 'path' and 'label' columns.")

        self.samples = list(zip(df["path"].tolist(), df["label"].tolist()))

        n_intact  = sum(1 for _, lbl in self.samples if lbl == 0)
        n_damaged = sum(1 for _, lbl in self.samples if lbl == 1)

        # Count how many images have JSON metadata available
        n_with_meta = sum(
            1 for path, _ in self.samples
            if (Path(path).parent.parent / "labels" /
                Path(path).with_suffix(".json").name).exists()
        )

        print(f"[{split}] Loaded {len(self.samples)} samples "
              f"(intact: {n_intact}, damaged: {n_damaged}, "
              f"ratio: {n_intact / max(n_damaged, 1):.1f}:1)")
        print(f"[{split}] Sun angles: {n_with_meta} images with JSON metadata, "
              f"{len(self.samples) - n_with_meta} using config defaults "
              f"(az={default_azimuth}°, el={default_elevation}°)")

        # ── Spatial augmentation ─────────────────────────────────────────────
        self.spatial_aug = self._build_spatial_aug(split, augment_cfg or {})

    def _build_spatial_aug(self, split, cfg):
        if split != "train":
            return None
        ops = [transforms.Resize((self.image_size, self.image_size))]
        if cfg.get("AUGMENT_HFLIP", True):
            ops.append(transforms.RandomHorizontalFlip())
        if cfg.get("AUGMENT_VFLIP", True):
            ops.append(transforms.RandomVerticalFlip())
        rot = cfg.get("AUGMENT_ROTATION", 90)
        if rot > 0:
            ops.append(transforms.RandomRotation(rot))
        return transforms.Compose(ops)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        # ── Load image ───────────────────────────────────────────────────────
        # QQB: .npy files · xBD: .png files
        p = Path(img_path)
        if p.suffix == ".png":
            img_np = np.array(Image.open(p).convert("RGB"))
        else:
            npy_path = p.with_suffix(".npy")
            img_np   = np.load(npy_path)

        img_pil = Image.fromarray(img_np, mode="RGB")

        # ── Resize ───────────────────────────────────────────────────────────
        img_pil = transforms.Resize((self.image_size, self.image_size))(img_pil)

        # ── Spatial augmentation ─────────────────────────────────────────────
        if self.spatial_aug is not None:
            img_pil = self.spatial_aug(img_pil)

        img_np_aug = np.array(img_pil)

        # ── Resolve sun angles for this specific image ────────────────────────
        azimuth, elevation = get_sun_angles(
            img_path,
            self.default_azimuth,
            self.default_elevation,
        )

        # ── Branch 1: observed shadow mask ────────────────────────────────────
        observed = compute_observed_shadow(
            img_np_aug,
            clahe_clip       = self.clahe_clip_limit,
            clahe_grid       = self.clahe_tile_grid,
            min_component_px = self.min_component_px,
        )

        # ── Branch 2: expected shadow map ─────────────────────────────────────
        expected = compute_expected_shadow(
            img_np_aug,
            sun_azimuth_deg   = azimuth,
            sun_elevation_deg = elevation,
            kernel_size       = self.coherence_kernel,
            smooth_sigma      = self.coherence_sigma,
        )

        # ── Tensors [1, H, W] ─────────────────────────────────────────────────
        observed_t = torch.from_numpy(observed).unsqueeze(0)
        expected_t = torch.from_numpy(expected).unsqueeze(0)

        return observed_t, expected_t, torch.tensor(label, dtype=torch.float32)

    # ── Class imbalance ───────────────────────────────────────────────────────
    def get_sample_weights(self) -> torch.Tensor:
        labels    = [lbl for _, lbl in self.samples]
        n_intact  = labels.count(0)
        n_damaged = labels.count(1)
        n_total   = len(labels)
        w_intact  = n_total / (2 * n_intact)
        w_damaged = n_total / (2 * n_damaged)
        weights   = [w_intact if lbl == 0 else w_damaged for lbl in labels]
        return torch.tensor(weights, dtype=torch.float32)
