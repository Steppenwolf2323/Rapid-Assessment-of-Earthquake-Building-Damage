"""
config_C.py — Model C: Physics-Guided Dual-Branch Shadow Comparison

Two branches, no RGB, no pretrained weights:
    Branch 1: observed shadow mask  (CLAHE + Otsu, same pipeline as Model B)
    Branch 2: expected shadow map   (sun geometry → directional coherence)

Both branches processed by identical small CNNs trained from scratch.
Features concatenated → classification head → binary output.

No two-stage fine-tuning: no pretrained weights to protect.
Single stage, 30 epochs, everything learns from scratch.
"""

from pathlib import Path

TRAIN_CSV = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\0_data_preprocessing\qqb_dataset\qqb_train.csv")
VAL_CSV   = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\0_data_preprocessing\qqb_dataset\qqb_val.csv")

OUTPUT_DIR = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\2_trainings\outputs\model_C")

 
IMAGE_SIZE = 224

# The QQB dataset uses a single Maxar acquisition over Turkey in Feb 2023.
# Sun azimuth ~150° (south-southeast), elevation ~32°.
# Shadows fall in the opposite direction (~330°, north-northwest).
# These values are fixed and used at both train and test time.
# At test time on xBD (different earthquakes, different locations),
# the same fixed values are used — making this a truly deployable model.
SUN_AZIMUTH_DEG   = 150.0   # degrees from North, clockwise
SUN_ELEVATION_DEG = 32.0    # degrees above horizon

# Branch 1: Observed shadow (CLAHE + Otsu)
CLAHE_CLIP_LIMIT        = 2.0
CLAHE_TILE_GRID         = (8, 8)
SHADOW_MIN_COMPONENT_PX = 50

# Branch 2: Expected shadow (directional coherence) ───────────────────────
COHERENCE_KERNEL_SIZE  = 7     # gradient filter kernel (must be odd)
COHERENCE_SMOOTH_SIGMA = 1.5   # gaussian smoothing sigma

# Small CNN architecture 
CNN_FEATURE_DIM = 256

EPOCHS       = 30
LR           = 1e-3
WEIGHT_DECAY = 1e-4

BATCH_SIZE  = 32
NUM_WORKERS = 0       

AUGMENT_HFLIP      = True
AUGMENT_VFLIP      = True
AUGMENT_ROTATION   = 90
AUGMENT_BRIGHTNESS = 0.2
AUGMENT_CONTRAST   = 0.2
AUGMENT_SATURATION = 0.2

THRESHOLD = 0.5

# Reproducibility
SEED = 42
