"""
config.py — Model A: Pure End-to-End ML
All hyperparameters in one place. Edit here, nothing else needs to change.
"""

from pathlib import Path

# ─── Paths ───────────────────────────────────────────────────────────────────
# Adjust DATA_DIR to wherever your prepared QQB splits live.
# Expected structure:
#   DATA_DIR/
#   ├── train/
#   │   ├── damaged/
#   │   └── intact/
#   └── val/
#       ├── damaged/
#       └── intact/

DATA_DIR   = Path("data/QQB")
OUTPUT_DIR = Path("outputs/model_A")

# ─── Image ───────────────────────────────────────────────────────────────────
IMAGE_SIZE = 224        # Already prepared at this resolution
NUM_CHANNELS = 3        # RGB only — no SAR, no mask

# ─── Model ───────────────────────────────────────────────────────────────────
FC_SIZE  = 512          # Hidden size of the FC layer in the head
DROPOUT  = 0.5          # Dropout probability

# ─── Stage 1 — Warm-up (backbone frozen) ─────────────────────────────────────
STAGE1_EPOCHS       = 10
STAGE1_LR           = 1e-3
STAGE1_WEIGHT_DECAY = 1e-4

# ─── Stage 2 — Full fine-tune (differential LR) ──────────────────────────────
STAGE2_EPOCHS       = 20
STAGE2_LR_BACKBONE  = 1e-5   # Small: backbone is already good, just nudge it
STAGE2_LR_HEAD      = 1e-4   # Larger: head still has more to learn
STAGE2_WEIGHT_DECAY = 1e-4

# ─── DataLoader ──────────────────────────────────────────────────────────────
BATCH_SIZE  = 32
NUM_WORKERS = 4

# ─── Augmentation (training only) ────────────────────────────────────────────
AUGMENT_HFLIP       = True
AUGMENT_VFLIP       = True
AUGMENT_ROTATION    = 90          # Max rotation degrees
AUGMENT_BRIGHTNESS  = 0.2
AUGMENT_CONTRAST    = 0.2
AUGMENT_SATURATION  = 0.2

# ─── Inference ───────────────────────────────────────────────────────────────
THRESHOLD = 0.5         # Sigmoid threshold for damaged (1) vs intact (0)

# ─── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42
