"""
config_B.py — Model B: ResNet-50 + Shadow Mask (4-channel input)
All hyperparameters in one place. Edit here, nothing else needs to change.
"""

from pathlib import Path

TRAIN_CSV = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\0_data_preprocessing\qqb_dataset\qqb_train.csv")
VAL_CSV   = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\0_data_preprocessing\qqb_dataset\qqb_val.csv")

OUTPUT_DIR = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\2_models_trainings\model_B")

IMAGE_SIZE   = 224
NUM_CHANNELS = 4      # RGB (3) + shadow mask (1)


CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)
SHADOW_MIN_COMPONENT_PX = 50

FC_SIZE = 512
DROPOUT = 0.5

STAGE1_EPOCHS       = 10
STAGE1_LR           = 1e-3
STAGE1_WEIGHT_DECAY = 1e-4

STAGE2_EPOCHS       = 20
STAGE2_LR_BACKBONE  = 1e-5
STAGE2_LR_HEAD      = 1e-4
STAGE2_WEIGHT_DECAY = 1e-4

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
