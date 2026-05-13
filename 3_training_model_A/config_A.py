"""
config_A.py — Model A: Pure End-to-End ML
All hyperparameters in one place. Edit here, nothing else needs to change.
"""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
# Point these to the two CSV files produced by your data pipeline.
TRAIN_CSV = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\0_data_preprocessing\qqb_dataset\qqb_train.csv")
VAL_CSV   = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\0_data_preprocessing\qqb_dataset\qqb_val.csv")

OUTPUT_DIR = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\3_trainings\outputs\model_A")

# ─── Image ───────────────────────────────────────────────────────────────────
# Images on disk are ~100×99 (3, H, W) stored in .mat files.
# We resize to 224×224 in the DataLoader to match ResNet-50's expected input.
IMAGE_SIZE   = 224
NUM_CHANNELS = 3      # RGB only — no SAR, no mask

# ─── Model ───────────────────────────────────────────────────────────────────
FC_SIZE = 512         # Hidden size of the FC layer in the classification head
DROPOUT = 0.5         # Dropout probability

# ─── Stage 1 — Warm-up (backbone frozen) ─────────────────────────────────────
STAGE1_EPOCHS       = 10
STAGE1_LR           = 1e-3
STAGE1_WEIGHT_DECAY = 1e-4

# ─── Stage 2 — Full fine-tune (differential LR) ──────────────────────────────
STAGE2_EPOCHS       = 20
STAGE2_LR_BACKBONE  = 1e-5   # Small: backbone already good, just nudge it
STAGE2_LR_HEAD      = 1e-4   # Larger: head still has more to learn
STAGE2_WEIGHT_DECAY = 1e-4

# ─── DataLoader ──────────────────────────────────────────────────────────────
BATCH_SIZE  = 32
NUM_WORKERS = 0

# ─── Augmentation (training only) ────────────────────────────────────────────
AUGMENT_HFLIP      = True
AUGMENT_VFLIP      = True
AUGMENT_ROTATION   = 90     # Max rotation degrees
AUGMENT_BRIGHTNESS = 0.2
AUGMENT_CONTRAST   = 0.2
AUGMENT_SATURATION = 0.2

# ─── Inference ───────────────────────────────────────────────────────────────
THRESHOLD = 0.5       # Sigmoid threshold: ≥ 0.5 → damaged, < 0.5 → intact

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42