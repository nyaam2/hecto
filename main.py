import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification

model_id = "microsoft/swinv2-base-patch4-window8-256"

processor = AutoImageProcessor.from_pretrained(model_id)
model = AutoModelForImageClassification.from_pretrained(
    model_id,
    num_labels=2,
    ignore_mismatched_sizes=True,  # 1000클래스 head -> 2클래스 head로 교체
)

model.config.id2label = {0:"real", 1:"fake"}
model.config.label2id = {"real":0, "fake":1}