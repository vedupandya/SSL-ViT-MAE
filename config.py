from pathlib import Path
import os

NETID = os.getenv("USER")
SCRATCH = Path(f"/scratch/{NETID}")
LOG_DIR = SCRATCH / "mae_logs"
CKPT_DIR = SCRATCH / "mae_checkpoints/75"
TRAIN_DIR = SCRATCH / "dataset" / "train"
EVAL_DIR = Path("./eval_data")

IMG_SIZE = 96
PATCH_SIZE = 16
BATCH_SIZE = 1024
NUM_WORKERS = 8
DEVICE = "cuda" if __import__('torch').cuda.is_available() else "cpu"
LOGGING = True

ENC_DIM = 384
ENC_DEPTH = 12
ENC_HEADS = 6
DEC_DIM = 192
DEC_DEPTH = 4
DEC_HEADS = 6
MASK_RATIO = 0.75
LR = 1.5e-4
WEIGHT_DECAY = 0.05
EPOCHS = 200

LP_BATCH = 256
LP_EPOCHS = 15

SCRATCH.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
