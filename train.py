import os
import math
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from src.models import build_hf_swinv2


from src.utils import load_yaml, seed_everything, ensure_dir, get_device
from src.dataset import ImageCSVDataset
from src.models import SwinV2Binary, save_model_pt

def build_loaders(cfg: dict):
    data_cfg = cfg["data"]
    train_csv = data_cfg["train_csv"]
    val_csv = data_cfg.get("val_csv", "")

    df = pd.read_csv(train_csv)

    if val_csv and os.path.exists(val_csv):
        df_train = pd.read_csv(train_csv)
        df_val = pd.read_csv(val_csv)
        train_tmp = "__train_tmp.csv"
        val_tmp = "__val_tmp.csv"
        df_train.to_csv(train_tmp, index=False)
        df_val.to_csv(val_tmp, index=False)
        train_csv_use, val_csv_use = train_tmp, val_tmp
    else:
        # 자동 split
        tr, va = train_test_split(df, test_size=data_cfg["val_ratio"], random_state=cfg["seed"], stratify=df["label"])
        train_csv_use, val_csv_use = "__train_tmp.csv", "__val_tmp.csv"
        tr.to_csv(train_csv_use, index=False)
        va.to_csv(val_csv_use, index=False)

    train_ds = ImageCSVDataset(
        csv_path=train_csv_use,
        img_root=data_cfg["img_root"],
        img_size=data_cfg["img_size"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        is_train=True,
        has_label=True
    )
    val_ds = ImageCSVDataset(
        csv_path=val_csv_use,
        img_root=data_cfg["img_root"],
        img_size=data_cfg["img_size"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        is_train=False,
        has_label=True
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=True
    )
    return train_loader, val_loader

def make_optimizer(cfg, model):
    tr = cfg["train"]
    # 단순 AdamW (대회 baseline로 충분)
    return torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])

def make_scheduler(cfg, optimizer, total_steps: int):
    warmup_steps = int(total_steps * cfg["train"]["warmup_ratio"])

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine decay
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum / max(1, total), correct / max(1, total)

def main():
    cfg = load_yaml("config/config.yaml")
    seed_everything(cfg["seed"])
    processor, model = build_hf_swinv2(cfg["model"]["hf_id"])

    device = get_device()

    train_loader, val_loader = build_loaders(cfg)

    mcfg = cfg["model"]
    model = SwinV2Binary(
        timm_name=mcfg["timm_name"],
        num_classes=mcfg["num_classes"],
        drop_rate=mcfg.get("drop_rate", 0.0),
        pretrained=True
    ).to(device)

    optimizer = make_optimizer(cfg, model)
    total_steps = cfg["train"]["epochs"] * len(train_loader)
    scheduler = make_scheduler(cfg, optimizer, total_steps)

    scaler = GradScaler(enabled=cfg["train"]["amp"])
    # label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])

    best_val_acc = -1.0
    ensure_dir("model")

    global_step = 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg["train"]["amp"]):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            if cfg["train"]["grad_clip"] and cfg["train"]["grad_clip"] > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            global_step += 1

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"[epoch {epoch+1}] val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model_pt(model, "model/model.pt")
            print("saved -> model/model.pt")

    print("done. best_val_acc =", best_val_acc)

if __name__ == "__main__":
    main()


