import os
import random
import numpy as np
import torch
import yaml

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 재현성 강화(속도 약간 손해)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
