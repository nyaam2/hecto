# config.py
import random
from pathlib import Path

import numpy as np
import torch

SEED = 42

MODEL_ID = "prithivMLmods/Deep-Fake-Detector-v2-Model"
TEST_DIR = Path("./test_data")

# Submission
OUTPUT_DIR = Path("./output")
OUT_CSV = OUTPUT_DIR / "baseline_submission.csv"
SAMPLE_SUBMISSION = Path("./sample_submission.csv")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif"}
VIDEO_EXTS = {".mp4", ".mov"}

TARGET_SIZE = (224, 224)
NUM_FRAMES = 10  # 비디오 샘플링 프레임 수

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
