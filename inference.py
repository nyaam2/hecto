import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax

from src.utils import load_yaml, seed_everything, get_device
from src.dataset import ImageCSVDataset
from src.models import SwinV2Binary, load_model_pt

@torch.no_grad()
def aggregate_video(df_pred: pd.DataFrame, mode: str, topk_ratio: float) -> pd.DataFrame:
    # df_pred: columns = [video_id, prob_fake]
    if "video_id" not in df_pred.columns:
        return df_pred

    out_rows = []
    for vid, g in df_pred.groupby("video_id"):
        probs = g["prob_fake"].to_numpy()

        if mode == "mean":
            score = float(probs.mean())
        elif mode == "median":
            score = float(np.median(probs))
        elif mode == "topk_mean":
            k = max(1, int(len(probs) * topk_ratio))
            score = float(np.sort(probs)[-k:].mean())
        else:
            raise ValueError(f"Unknown aggregation: {mode}")

        out_rows.append({"video_id": vid, "prob_fake": score})

    return pd.DataFrame(out_rows)

def main():
    cfg = load_yaml("config/config.yaml")
    seed_everything(cfg["seed"])
    device = get_device()

    data_cfg = cfg["data"]
    inf_cfg = cfg["inference"]
    mcfg = cfg["model"]

    test_ds = ImageCSVDataset(
        csv_path=data_cfg["test_csv"],
        img_root=data_cfg["img_root"],
        img_size=data_cfg["img_size"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
        is_train=False,
        has_label=False
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=inf_cfg["batch_size"],
        shuffle=False,
        num_workers=inf_cfg["num_workers"],
        pin_memory=True
    )

    model = SwinV2Binary(
        timm_name=mcfg["timm_name"],
        num_classes=mcfg["num_classes"],
        drop_rate=mcfg.get("drop_rate", 0.0),
        pretrained=False  # 오프라인: 가중치는 model.pt로 로드
    ).to(device)
    load_model_pt(model, "model/model.pt", map_location=device)
    model.eval()

    rows = []
    for x, meta in test_loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = softmax(logits, dim=1)[:, 1].detach().cpu().numpy()  # fake 확률

        # meta는 batch별 list가 아니라 collate된 dict 형태가 아니므로, DataLoader 기본 collate 기준으로 처리
        # 여기서는 meta가 list of dict로 들어온다고 가정하고 처리
        # (torch 기본 collate는 dict를 dict of lists로 바꿈)
        if isinstance(meta, dict):
            batch_paths = meta.get("path", [])
            batch_vids = meta.get("video_id", None)
        else:
            batch_paths = [m["path"] for m in meta]
            batch_vids = [m.get("video_id") for m in meta]

        for i, p in enumerate(probs):
            r = {"path": batch_paths[i], "prob_fake": float(p)}
            if batch_vids is not None:
                r["video_id"] = batch_vids[i]
            rows.append(r)

    df_pred = pd.DataFrame(rows)

    # 영상 단위 집계(허용된 후처리)
    if "video_id" in df_pred.columns:
        df_out = aggregate_video(df_pred, inf_cfg["aggregation"], inf_cfg["topk_ratio"])
        # 제출 형식이 video_id/label이면 여길 맞춰야 함
        # 예: label 컬럼이 필요하면 df_out["label"]=df_out["prob_fake"] 같은 식으로
        df_out.to_csv(inf_cfg["output_csv"], index=False, encoding="utf-8")
    else:
        df_pred.to_csv(inf_cfg["output_csv"], index=False, encoding="utf-8")

    print("saved:", inf_cfg["output_csv"])

if __name__ == "__main__":
    main()
