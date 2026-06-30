"""
config_A.py — Model A: Pure End-to-End ML
All hyperparameters in one place. Edit here, nothing else needs to change.
"""

from pathlib import Path


TRAIN_CSV = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\0_data_preprocessing\qqb_dataset\qqb_train.csv")
VAL_CSV   = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\0_data_preprocessing\qqb_dataset\qqb_val.csv")

OUTPUT_DIR = Path(r"C:\Users\zanca\OneDrive\Desktop\Vrij Unversiteit\extra_year\Thesis\Rapid Assessment of Earthquake Building Damage\3_trainings\outputs\model_A")


IMAGE_SIZE   = 224
NUM_CHANNELS = 3      

FC_SIZE = 512         # Hidden size of the FC layer in the classification head
DROPOUT = 0.5         # Dropout probability

# Stage 1
STAGE1_EPOCHS       = 10
STAGE1_LR           = 1e-3
STAGE1_WEIGHT_DECAY = 1e-4

# Stage 2 
STAGE2_EPOCHS       = 20
STAGE2_LR_BACKBONE  = 1e-5   
STAGE2_LR_HEAD      = 1e-4   
STAGE2_WEIGHT_DECAY = 1e-4

# DataLoader 
BATCH_SIZE  = 32
NUM_WORKERS = 0

# Augmentation (training only) 
AUGMENT_HFLIP      = True
AUGMENT_VFLIP      = True
AUGMENT_ROTATION   = 90     
AUGMENT_BRIGHTNESS = 0.2
AUGMENT_CONTRAST   = 0.2
AUGMENT_SATURATION = 0.2

# Inference
THRESHOLD = 0.5       # Sigmoid threshold: ≥ 0.5 → damaged, < 0.5 → intact

# Reproducibility 
SEED = 42