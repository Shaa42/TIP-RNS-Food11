# TIP-RNS Food-11

Food-11 image classification project comparing a CNN trained from scratch, a ResNet-18 baseline, and ImageNet-pretrained ShuffleNetV2 models. The final lightweight model uses knowledge distillation from ShuffleNetV2 x1.0 to ShuffleNetV2 x0.5.

## Setup

```bash
uv sync
```

Place the dataset in `food-11/`, with one subdirectory per class under `food-11/train/` and the unlabeled images under `food-11/test/`. Dataset files, checkpoints, and local `runs_*` directories are ignored by Git.

## Optional resizing

```bash
uv run python scripts/resize.py --size 224 224 --output-dir food-11-224x224 --input-dir food-11
```

The transfer-learning scripts can also resize images online, so a separate resized copy is not required.

## Training

### ShuffleNetV2 baseline

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python transfer/train_transfer.py \
  --data-dir food-11/train \
  --test-dir food-11/test \
  --arch shufflenet_v2_x1_0 \
  --epochs 15 \
  --class-weight none \
  --augmentation current \
  --out-dir runs_shufflenet_v2_x1_0
```

Use `--arch shufflenet_v2_x0_5` for the lightweight baseline. Training uses online spatial and colour augmentation, ImageNet normalization, and flip-2 validation (original image plus horizontal mirror, with averaged probabilities). The checkpoint is selected by validation macro-F1.

### Knowledge distillation

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python transfer/train_knowledge_distillation.py \
  --data-dir food-11/train \
  --test-dir food-11/test \
  --teacher-checkpoint runs_shufflenet_v2_x1_0/best_model.pt \
  --student-arch shufflenet_v2_x0_5 \
  --epochs 15 \
  --alpha 0.5 \
  --temperature 3 \
  --label-smoothing 0.1 \
  --out-dir runs_shufflenet_v2_x0_5_distill
```

The frozen x1.0 teacher supplies soft targets while the x0.5 student also learns from the ground-truth labels.

### Generate predictions from a checkpoint

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python transfer/generate_predictions.py \
  --checkpoint runs_shufflenet_v2_x0_5_distill/epoch_15.pt \
  --test-dir food-11/test \
  --output predictions.json
```

## Main results

| Model | Parameters | Test F1 | Accuracy | Frugality |
|---|---:|---:|---:|---:|
| ShuffleNetV2 x1.0, flip-2 | 1.265 M | **0.9376** | 0.9461 | 0.7701 |
| ShuffleNetV2 x0.5, flip-2 | 0.353 M | 0.8936 | 0.9057 | 0.8153 |
| ShuffleNetV2 x0.5, distilled, epoch 15 | 0.353 M | 0.8980 | 0.9154 | **0.8197** |

The x1.0 model gives the best classification F1, while the distilled x0.5 model gives the best challenge frugality score. Detailed experiment comparisons are recorded in `results/shufflenet/experiments.csv`; compact logs, confusion matrices, summaries, and the final submission JSON are stored under `results/shufflenet/`.

## Script guide

- `transfer/train_transfer.py`: baseline ShuffleNet transfer learning and the augmentation/class-weight experiments.
- `transfer/train_knowledge_distillation.py`: frozen-teacher knowledge distillation.
- `transfer/generate_predictions.py`: flip-2 inference and JSON generation from a saved checkpoint.
- `transfer/evaluate_multiscale_tta.py`: deterministic multi-scale TTA experiment.
- `transfer/experiments/average_checkpoints.py`: auxiliary checkpoint-averaging experiment; it did not improve the final score.
