"""
config_B.py — Model B: ResNet-50 + Shadow Mask (4-channel input)
All hyperparameters in one place. Edit here, nothing else needs to change.
"""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
TRAIN_CSV = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\0_data_preprocessing\qqb_dataset\qqb_train.csv")
VAL_CSV   = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\0_data_preprocessing\qqb_dataset\qqb_val.csv")

OUTPUT_DIR = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\2_models_trainings\model_B")

# ─── Image ───────────────────────────────────────────────────────────────────
IMAGE_SIZE   = 224
NUM_CHANNELS = 4      # RGB (3) + shadow mask (1)

# ─── Shadow mask generation ───────────────────────────────────────────────────
# CLAHE clip limit: controls how aggressively local contrast is enhanced.
# Higher = more contrast enhancement, but also more noise amplification.
# Sanskriti used 2.0 — a safe, well-tested default.
CLAHE_CLIP_LIMIT = 2.0

# CLAHE tile grid size: the image is divided into this many tiles for
# local histogram equalisation. (8, 8) is standard for 224×224 images.
CLAHE_TILE_GRID  = (8, 8)

# Minimum connected component size (pixels) to keep after morphological
# cleaning. Blobs smaller than this are treated as noise and removed.
# Sanskriti used 100px on 224×224 images. We use 50px because our
# building patches are small individual crops — real shadow regions
# from a single building may be relatively small.
# Increase if the mask is too noisy; decrease if real shadows are removed.
SHADOW_MIN_COMPONENT_PX = 50

# ─── Model ───────────────────────────────────────────────────────────────────
FC_SIZE = 512
DROPOUT = 0.5

# ─── Stage 1 — Warm-up (backbone frozen) ─────────────────────────────────────
STAGE1_EPOCHS       = 10
STAGE1_LR           = 1e-3
STAGE1_WEIGHT_DECAY = 1e-4

# ─── Stage 2 — Full fine-tune (differential LR) ──────────────────────────────
STAGE2_EPOCHS       = 20
STAGE2_LR_BACKBONE  = 1e-5
STAGE2_LR_HEAD      = 1e-4
STAGE2_WEIGHT_DECAY = 1e-4

# ─── DataLoader ──────────────────────────────────────────────────────────────
BATCH_SIZE  = 32
NUM_WORKERS = 0       # Must be 0 on Windows — h5py + multiprocessing conflict

# ─── Augmentation (training only) ────────────────────────────────────────────
# Applied to the RGB image BEFORE shadow mask generation, so the mask
# always corresponds to the augmented image.
AUGMENT_HFLIP      = True
AUGMENT_VFLIP      = True
AUGMENT_ROTATION   = 90
AUGMENT_BRIGHTNESS = 0.2
AUGMENT_CONTRAST   = 0.2
AUGMENT_SATURATION = 0.2

# ─── Inference ───────────────────────────────────────────────────────────────
THRESHOLD = 0.5

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
