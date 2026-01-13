# Deepfake Detection (Single SwinV2)

## Train
python train.py

## Inference
python inference.py

## Outputs
- model/model.pt : final single-model weight
- submission.csv : inference output

## Notes
- No TTA used.
- Per-frame inference + allowed post-processing aggregation (mean/median/topk_mean).
