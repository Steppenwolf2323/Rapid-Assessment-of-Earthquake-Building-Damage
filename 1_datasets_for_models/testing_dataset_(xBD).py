import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Callable, Optional


# ImageNet stats are the same as QQBDataset so both datasets are compatible
# with the same pretrained backbone
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_default_transform(
    image_size: tuple[int, int],
    normalize: bool,
    augment: bool = False,
) -> transforms.Compose:
    """
    Build the default torchvision transform pipeline.

    Identical signature to the QQB version so both datasets can share
    the same training loop without changes.

    Args:
        image_size:  Target (height, width). xBD source images are 1024×1024;
                     they will be downsampled to match the backbone input size.
        normalize:   Apply ImageNet mean/std normalization.
        augment:     Add training-time augmentations (flip, color jitter).
                     Pass True for the training CSV, False for validation/test.
    """
    ops = []

    if augment:
        ops += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]

    ops += [
        transforms.Resize(image_size),
        transforms.ToTensor(),          # uint8 HWC  →  float32 CHW  in [0, 1]
    ]

    if normalize:
        ops.append(transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return transforms.Compose(ops)


class XBDDataset(Dataset):
    """
    PyTorch Dataset for the xBD earthquake subset.

    Reads a CSV produced by xbd_preprocessing.ipynb with columns:
        - path     : full path to the post-disaster PNG image
        - label    : 0 = intact, 1 = damaged  (binary, mapped from 4-class xBD)
        - source   : 'tier3' or 'train'  (informational, not used in training)
        - disaster : disaster event name  (informational, useful for per-event eval)

    Optionally enriches each sample with image-level metadata extracted from
    the companion GeoJSON label file (sun_azimuth, sun_elevation, etc.).
    This mirrors the metadata hook in QQBDataset and is used by Model C.

    Designed to be a drop-in companion to QQBDataset:
        Model A  →  use as-is (pure data-driven)
        Model B  →  pass a custom `transform` that includes RGB masking
        Model C  →  pass `load_metadata=True` to get sun geometry; or pass a
                    custom `pre_transform` for physics-guided shadow operations
    """

    def __init__(
        self,
        csv_file: str,
        image_size: tuple[int, int] = (224, 224),
        normalize: bool = True,
        augment: bool = False,
        load_metadata: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            csv_file:       Path to xbd_train.csv or xbd_val.csv.
            image_size:     Target spatial resolution for the CNN input.
                            xBD images are 1024×1024 and will be resized.
            normalize:      Apply ImageNet normalization.
            augment:        Enable training-time augmentations.
                            Pass True for training CSV, False for validation.
            load_metadata:  If True, parse the companion GeoJSON for each image
                            and return sun_azimuth, sun_elevation, off_nadir_angle,
                            capture_date, and sensor as a metadata dict alongside
                            the (image, label) pair. Required for Model C.
            transform:      Fully custom torchvision transform that REPLACES the
                            default pipeline. Use for Model B masking.
            pre_transform:  Optional callable applied to the raw PIL Image BEFORE
                            the main transform. Useful for physics-based operations
                            in Model C (e.g., shadow masking).
        """
        self.df            = pd.read_csv(csv_file)
        self.image_size    = image_size
        self.pre_transform = pre_transform
        self.load_metadata = load_metadata

        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain 'path' and 'label' columns.")

        self.transform = transform or build_default_transform(
            image_size=image_size,
            normalize=normalize,
            augment=augment,
        )

    # Dataset protocol

    def __len__(self) -> int:
        return len(self.df)

    # Metadata fields available in xBD GeoJSON files.
    # Mirrors METADATA_COLS in QQBDataset so Model C code is reusable.
    METADATA_COLS = [
        "sun_azimuth",      # degrees clockwise from north
        "sun_elevation",    # degrees above horizon
        "off_nadir_angle",  # satellite off-nadir angle in degrees
        "capture_date",     # UTC timestamp string
        "sensor",           # satellite sensor name (e.g. WORLDVIEW02)
        "gsd",              # ground sampling distance in metres
    ]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict]:
        row       = self.df.iloc[idx]
        file_path = str(row["path"])
        label     = int(row["label"])

        # 1. Load PNG as PIL Image (already HWC uint8 RGB)
        image_pil = self._load_png_image(file_path)

        # 2. Optional physics / masking pre-processing (Model B / C )
        if self.pre_transform is not None:
            image_pil = self.pre_transform(image_pil)

        # 3. Main transform → Tensor (CHW, float32, optionally normalized)
        image_tensor: Tensor = self.transform(image_pil)
        label_tensor: Tensor = torch.tensor(label, dtype=torch.long)

        # 4. Metadata (Model C) — loaded from GeoJSON only when requested.
        if self.load_metadata:
            metadata = self._load_geojson_metadata(file_path)
            if metadata:
                return image_tensor, label_tensor, metadata

        return image_tensor, label_tensor

    # Internal helpers

    def _load_png_image(self, file_path: str) -> Image.Image:
        """
        Load a post-disaster PNG image from xBD.

        xBD images are stored as standard RGB PNGs (1024×1024, uint8).
        Unlike QQB (HDF5 with CHW layout), no axis transposition is needed.

        Returns a PIL Image in RGB mode, ready for torchvision transforms.
        """
        img = Image.open(file_path)

        # Ensure RGB
        if img.mode != "RGB":
            img = img.convert("RGB")

        return img

    def _load_geojson_metadata(self, image_path: str) -> dict:
        """
        Parse the companion post-disaster GeoJSON for image-level metadata.

        The GeoJSON lives in a sibling `labels/` directory and has the same
        stem as the image but with a .json extension.

        xBD metadata fields (stored under data['metadata']):
            sun_azimuth, sun_elevation, off_nadir_angle,
            capture_date, sensor, gsd, ...

        Returns an empty dict if the JSON is missing or unreadable,
        which keeps __getitem__ falling back to the plain (image, label) pair.
        """
        # Derive the label path from the image path:
        
        label_path = image_path.replace(
            os.sep + "images" + os.sep,
            os.sep + "labels" + os.sep,
        ).replace(".png", ".json")

        if not os.path.exists(label_path):
            return {}

        try:
            with open(label_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

        raw_meta = data.get("metadata", {})

        metadata = {}
        for col in self.METADATA_COLS:
            val = raw_meta.get(col)
            if val is not None:
                metadata[col] = val

        return metadata

    
