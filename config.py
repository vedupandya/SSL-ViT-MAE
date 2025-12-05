from pathlib import Path
import os

NETID = os.getenv("USER")
SCRATCH = Path(f"/scratch/{NETID}")
LOG_DIR = SCRATCH / "mae_logs"
CKPT_DIR = SCRATCH / "mae_checkpoints/75"
TRAIN_DIR = SCRATCH / "dataset" / "train"
EVAL_DIR = Path("./eval_data")

# --- Global Training Parameters ---
IMG_SIZE = 96
BATCH_SIZE = 1024
NUM_WORKERS = 8
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"
LOGGING = True
EPOCHS = 200
LP_BATCH = 256
LP_EPOCHS = 10

# --- Model Configurations (Architecture and Hyperparameters) ---

# Original Small Model (~21.5M params)
CONFIG_SMALL = {
    'PATCH_SIZE': 16,
    'ENC_DIM': 384,
    'ENC_DEPTH': 12,
    'ENC_HEADS': 6,
    'DEC_DIM': 192,
    'DEC_DEPTH': 4,
    'DEC_HEADS': 6,
    'MASK_RATIO': 0.75,
    'LR': 1.5e-4,          # Original LR for this model
    'WEIGHT_DECAY': 0.05,
}

# New ViT-Base-like Model (~85.3M params)
CONFIG_BASE = {
    'PATCH_SIZE': 8,       # Changed for better performance at 96x96
    'ENC_DIM': 768,        # Increased dimension (ViT-Base size)
    'ENC_DEPTH': 12,
    'ENC_HEADS': 12,       # Increased heads
    'DEC_DIM': 384,        # Increased decoder dimension
    'DEC_DEPTH': 8,        # Increased decoder depth
    'DEC_HEADS': 12,
    'MASK_RATIO': 0.75,
    'LR': 1.5e-4,          # Scale LR for 1024 batch size
    'WEIGHT_DECAY': 0.05,
}

# Default model size to use if not specified
DEFAULT_MODEL_SIZE = 'base'

# Mapping for easy access
MODEL_CONFIGS = {
    'small': CONFIG_SMALL,
    'base': CONFIG_BASE,
}

# --- Directory Setup ---
SCRATCH.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)