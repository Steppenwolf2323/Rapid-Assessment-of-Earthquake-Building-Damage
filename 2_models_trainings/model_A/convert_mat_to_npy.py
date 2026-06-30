"""
convert_mat_to_npy.py — One-time preprocessing utility

Converts all .mat (MATLAB v7.3 / HDF5) image files referenced in the
QQB CSV files into .npy (NumPy) files saved alongside the originals.

Why:
    Loading .mat files via h5py during training is slow and starves
    the GPU. NumPy .npy files load ~10x faster, keeping the GPU fed.

Run this ONCE before training. It does not delete or modify the
original .mat files. If a .npy file already exists it is skipped.

Usage:
    python convert_mat_to_npy.py

Output:
    For each file like:  building_001_opt.mat
    Creates alongside:   building_001_opt.npy   shape=(H, W, 3) uint8
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path

CSV_FILES = [
    r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\0_data_preprocessing\qqb_dataset\qqb_train.csv",
    r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\0_data_preprocessing\qqb_dataset\qqb_val.csv",
]


def convert_file(mat_path: Path) -> bool:
    """
    Converts a single .mat file to .npy.
    Returns True if converted, False if skipped (already exists).
    """
    npy_path = mat_path.with_suffix(".npy")

    if npy_path.exists():
        return False  # Already converted, skip

    with h5py.File(mat_path, 'r') as f:
        img = f['x3'][:]                        # (3, H, W) uint8
    img = np.transpose(img, (1, 2, 0))          # → (H, W, 3) uint8
    np.save(npy_path, img)
    return True


def main():
    all_paths = []

    for csv_path in CSV_FILES:
        df = pd.read_csv(csv_path)
        all_paths.extend(df["path"].tolist())

    all_paths = list(set(all_paths))  # Remove duplicates
    total     = len(all_paths)

    print(f"Found {total} unique .mat files across all CSVs.")
    print("Starting conversion...\n")

    converted = 0
    skipped   = 0
    errors    = 0

    for i, path in enumerate(all_paths, 1):
        mat_path = Path(path)

        if not mat_path.exists():
            print(f"  [ERROR] File not found: {mat_path.name}")
            errors += 1
            continue

        try:
            was_converted = convert_file(mat_path)
            if was_converted:
                converted += 1
            else:
                skipped += 1

            if i % 100 == 0 or i == total:
                print(f"  Progress: {i}/{total} "
                      f"(converted: {converted}, skipped: {skipped}, errors: {errors})")

        except Exception as e:
            print(f"  [ERROR] {mat_path.name}: {e}")
            errors += 1

    print(f"\nDone.")
    print(f"  Converted : {converted}")
    print(f"  Skipped   : {skipped}  (already existed)")
    print(f"  Errors    : {errors}")


if __name__ == "__main__":
    main()