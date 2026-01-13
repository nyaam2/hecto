import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ImageCSVDataset(Dataset):
    """
    train.csv: path,label (label: 0/1)
    test.csv : path[,video_id,frame_id]
    """
    def __init__(self, csv_path: str, img_root: str, img_size: int, mean, std, is_train: bool, has_label: bool):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.has_label = has_label

        if is_train:
            self.tf = T.Compose([
                T.Resize((img_size, img_size)),
                # 딥페이크에서 과한 증강은 독이 될 수 있어 “기본+압축/리사이즈”는 train.py에서 더 붙이는 방식 추천
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])
        else:
            self.tf = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ])

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, p: str) -> str:
        if os.path.isabs(p) or self.img_root == "":
            return p
        return os.path.join(self.img_root, p)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self._resolve_path(row["path"])
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)

        meta = {"path": row["path"]}
        if "video_id" in row:
            meta["video_id"] = row["video_id"]
        if "frame_id" in row:
            meta["frame_id"] = row["frame_id"]

        if self.has_label:
            y = int(row["label"])
            return x, y, meta
        else:
            return x, meta
