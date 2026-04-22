import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from typing import Callable, Optional


# ImageNet stats used by most pretrained CNNs (ResNet, EfficientNet, ViT, ...)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_default_transform(
    image_size: tuple[int, int],
    normalize: bool,
    augment: bool = False,
) -> transforms.Compose:
    """
    The function build a default torch transformer.

    Args:
        image_size:  Target (height, width)
        normalize:   Apply ImageNet mean/std normalization                     
        augment:     Add basic training-time augmentations (flip, color jitter).
                     Set to True for the training split, False for validation/test.
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


class QQBDataset(Dataset):
    """
    PyTorch Dataset for the QuickQuakeBuildings (QQB) optical subset.

    Reads a CSV with columns:
        - path  : full path to the HDF5 (.mat) file
        - label : 0 = intact, 1 = damaged

    Designed to be reusable across all three thesis models:
        Model A  →  use as-is (pure data-driven)
        Model B  →  pass a custom `transform` that includes RGB masking
        Model C  →  pass a custom `transform` or `pre_transform` for
                    physics-guided shadow manipulation
    """

    def __init__(
        self,
        csv_file: str,
        image_key: str = "x3",
        image_size: tuple[int, int] = (224, 224),
        normalize: bool = True,
        augment: bool = False,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            csv_file:       Path to the CSV file (produced by qqb_preprocessing.ipynb).
            image_key:      Key inside the HDF5 file that holds the RGB array.
            image_size:     Target spatial resolution for the CNN input.
            normalize:      Apply ImageNet normalization. Set False only if your
                            backbone was trained on non-ImageNet data.
            augment:        Enable training-time augmentations (flip, color jitter).
                            Pass True for the training CSV, False for validation.
            transform:      Fully custom torchvision transform that REPLACES the
                            default pipeline. Use this for Model B masking or any
                            other manual preprocessing.
            pre_transform:  Optional callable applied to the raw PIL Image BEFORE
                            the main transform. Useful for physics-based operations
                            in Model C (e.g., shadow masking, histogram adjustments).
        """
        self.df = pd.read_csv(csv_file)
        self.image_key = image_key
        self.image_size = image_size
        self.pre_transform = pre_transform

        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain 'path' and 'label' columns.")

        # If a fully custom transform is supplied, use it directly.
        # Otherwise build the default pipeline from the flags.
        self.transform = transform or build_default_transform(
            image_size=image_size,
            normalize=normalize,
            augment=augment,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    # Columns that are treated as metadata when present in the CSV.
    # Extend this list as new metadata fields become available.
    METADATA_COLS = [
        "sun_azimuth",      # degrees clockwise from north
        "sun_elevation",    # degrees above horizon
        "acquisition_time", # UTC timestamp string
        "satellite_angle",  # off-nadir angle in degrees
    ]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, dict]:
        row = self.df.iloc[idx]
        file_path: str = row["path"]
        label: int = int(row["label"])

        # 1. Load raw numpy array (always uint8 HWC)
        image_np = self._load_h5_image(file_path)

        # 2. Convert to PIL so torchvision transforms can work on it
        image_pil = Image.fromarray(image_np)

        # 3. Optional physics / masking pre-processing (Model B / C hook)
        if self.pre_transform is not None:
            image_pil = self.pre_transform(image_pil)

        # 4. Main transform → Tensor (CHW, float32, optionally normalized)
        image_tensor: Tensor = self.transform(image_pil)

        label_tensor: Tensor = torch.tensor(label, dtype=torch.long)

        # 5. Metadata (Model C) — returned only when at least one column is present.
        #    When real metadata is unavailable the CSV simply won't have these
        #    columns, so the dataset silently falls back to the (image, label) pair.
        metadata = self._extract_metadata(row)
        if metadata:
            return image_tensor, label_tensor, metadata

        return image_tensor, label_tensor

    def _extract_metadata(self, row: pd.Series) -> dict:
        """
        Extract any available metadata fields from the current CSV row.

        Returns an empty dict if none of the expected columns are present,
        which keeps __getitem__ returning a plain (image, label) tuple and
        makes Models A and B completely unaffected.

        The dict values are left as Python scalars (float / str) so they
        can be collated by a custom DataLoader collate_fn if needed.
        """
        metadata = {}
        for col in self.METADATA_COLS:
            if col in self.df.columns:
                val = row[col]
                # NaN means the field exists in the CSV but wasn't filled yet
                if pd.notna(val):
                    metadata[col] = val
        return metadata

    @property
    def has_metadata(self) -> bool:
        """True if the CSV contains at least one recognised metadata column."""
        return any(col in self.df.columns for col in self.METADATA_COLS)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_h5_image(self, file_path: str) -> np.ndarray:
        """
        Load the RGB image from a QQB HDF5 (.mat) file.

        The QQB files store images in CHW order (3, H, W) with dtype uint8.
        This method always returns a uint8 array in HWC order (H, W, 3),
        ready to be wrapped in a PIL Image.

        Any float/scaling normalization is intentionally left to the
        torchvision transform pipeline (ToTensor handles [0,255]→[0,1]).
        """
        with h5py.File(file_path, "r") as f:
            if self.image_key not in f:
                raise KeyError(
                    f"Key '{self.image_key}' not found in {file_path}. "
                    f"Available keys: {list(f.keys())}"
                )
            image = f[self.image_key][:]          # load into memory

        image = np.asarray(image)

        # --- Axis ordering ---
        if image.ndim == 3:
            if image.shape[0] == 3 and image.shape[-1] != 3:
                # CHW  →  HWC
                image = np.transpose(image, (1, 2, 0))
            elif image.shape[-1] == 3:
                pass                              # already HWC
            else:
                raise ValueError(
                    f"Cannot interpret 3-D shape {image.shape} as RGB in {file_path}"
                )
        elif image.ndim == 2:
            # Grayscale → replicate to 3 channels
            image = np.stack([image] * 3, axis=-1)
        else:
            raise ValueError(
                f"Unexpected array ndim={image.ndim}, shape={image.shape} in {file_path}"
            )

        # --- dtype: always return uint8 ---
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                # Float image in [0, 1] → scale to [0, 255]
                image = (image * 255).clip(0, 255).astype(np.uint8)
            else:
                # Already in a larger integer range → cast directly
                image = image.astype(np.uint8)

        return image                              # HWC, uint8



