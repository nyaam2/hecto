import torch
import torch.nn as nn
import timm

class SwinV2Binary(nn.Module):
    def __init__(self, timm_name: str, num_classes: int = 2, drop_rate: float = 0.0, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,           # feature extractor로 사용
            drop_rate=drop_rate
        )
        feat_dim = self.backbone.num_features
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)      # [B, C]
        logits = self.head(feat)     # [B, 2]
        return logits

def save_model_pt(model: nn.Module, path: str) -> None:
    state = model.state_dict()
    torch.save(state, path)

def load_model_pt(model: nn.Module, path: str, map_location="cpu") -> None:
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state, strict=True)

from transformers import AutoImageProcessor, AutoModelForImageClassification

def build_hf_swinv2(model_id: str, num_labels: int = 2):
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )
    model.config.id2label = {0: "real", 1: "fake"}
    model.config.label2id = {"real": 0, "fake": 1}
    return processor, model

