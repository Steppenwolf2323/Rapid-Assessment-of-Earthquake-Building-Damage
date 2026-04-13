import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class QQBDataset(Dataset):
    """
    This class create a PYTorch dataset for the trainings of the different models.
    It feeds on the dataset previously adapted from the orignal QQB dataset and transformed 
    in a CSV file with the following columns:

    - path: full path to the .mat file
    - labels: 0 for intact, 1 for damaged

    """

    def __init__(self, csv_file: str, image_key: str = "x3",image_size: tuple[int, int] = (224, 224),normalization: bool = True, transform=None) -> None:
        """
        Args:
            csv_file: Path to the CSV file containing paths and labels.
            image_key: Key inside the .mat.
            image_size: Target image size (height, width).
            normalization: Normalized size images.
            transform: Optional custom transform. If None, a default transform is used.
        """
        self.df = pd.read_csv(csv_file)
        self.image_key = image_key
        self.image_size = image_size
        self.normalization = normalization

        if "path" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("CSV must contain at least 'path' and 'label' columns.")

        # Default transform: resize + tensor conversion
        self.transform = transform or transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.df)

    def _load_h5_image(self, file_path: str) -> np.ndarray:
        """
        Load the RGB image from a QQB .mat file.

        Returns:
            image as a NumPy array in HWC format.
        """
        with h5py.File(file_path, "r") as f:
            if self.image_key not in f:
                raise KeyError(
                    f"Key '{self.image_key}' not found in file: {file_path}. "
                    f"Available keys: {list(f.keys())}"
                )

            image = f[self.image_key][:]

        # We want final format HWC.
        image = np.array(image)

        if image.ndim == 3:
            
            if image.shape[-1] == 3:
                pass
            elif image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            else:
                raise ValueError(f"Unexpected 3D image shape {image.shape} in file: {file_path}")
        elif image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        else:
            raise ValueError(
                f"Unexpected image ndim={image.ndim} with shape {image.shape} in file: {file_path}")

        # Ensure numeric type is compatible with PIL / transforms
        if image.dtype != np.uint8:
            # If values are already in [0,1], convert safely to uint8 for PIL
            if self.normalization and image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        return image

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        row = self.df.iloc[idx]
        file_path = row["path"]
        label = int(row["label"])

        image = self._load_h5_image(file_path)

        image = Image.fromarray(image)

        image = self.transform(image)

        label_tensor = torch.tensor(label, dtype=torch.long)

        return image, label_tensor