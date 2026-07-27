# E1 — Image Classification: MobileNetV2 vs ResNet18 (PKLot)

Comparison of two image classification models (ImageNet transfer learning)
for parking-lot occupancy detection, on the **PKLot** dataset. This folder
covers experiment E1 (binary empty/occupied classification), alongside
experiment E2 (YOLO object detection, see the rest of this repository).

## 1. Dataset origin and preparation

The source dataset ([PKLot via Kaggle/Roboflow](https://public.roboflow.ai/object-detection/pklot))
provides **full-scene images** (640×640) with **COCO** annotations — one
bounding box per parking space, category `space-empty` or
`space-occupied`. It is not already cropped per space.

```
raw_kaggle_download/archive/
├── train/            scene images + _annotations.coco.json
├── valid/
└── test/
```

`prepare_dataset.py` crops each annotated bounding box into an individual
image and builds the `ImageFolder` structure expected by the rest of the
pipeline:

```
dataset/
├── train/{empty,occupied}/*.jpg
├── val/{empty,occupied}/*.jpg
└── test/{empty,occupied}/*.jpg
```

The full dataset amounts to ~712k crops (far too many for CPU training in
reasonable time); a balanced subsample is used instead (4000/1000/1000
images per class for train/val/test).

> ⚠️ **Known methodological limitation (see section 6 below):** the
> train/valid/test split provided by the Roboflow export assigns
> individual frames at random, without grouping by capture day — which
> causes session-level data leakage. Use `build_grouped_split.py` for a
> leakage-free split (`dataset_grouped/`).

## 2. Installation

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## 3. Full workflow

```bash
# Dataset preparation (once, from the extracted Kaggle zip)
python prepare_dataset.py --source raw_kaggle_download/archive \
    --per-class-train 4000 --per-class-val 1000 --per-class-test 1000

# Training (saves the best checkpoint + history)
python train.py --model mobilenet_v2
python train.py --model resnet18

# Evaluation on the test set (accuracy, precision, recall, F1, confusion
# matrix, error examples, per-weather metrics if detectable)
python evaluate.py --model mobilenet_v2
python evaluate.py --model resnet18

# Final comparison (table + inference time)
python compare.py
```

## 4. Project files

- `config.py` — all hyperparameters (never hardcoded elsewhere)
- `utils.py` — dataloaders + model construction
- `prepare_dataset.py` — crops parking spaces from COCO annotations
- `train.py --model {mobilenet_v2|resnet18}` — training with early stopping
- `evaluate.py --model {mobilenet_v2|resnet18}` — test metrics + errors + weather
- `compare.py` — final comparison table (requires train+evaluate for both models)
- `build_grouped_split.py` — rebuilds a leakage-free split, grouped by capture date
- `leakage_crosseval.py --model {...}` — evaluates an existing checkpoint on the leakage-free split
- `leakage_diagnostic_train.py --model {...}` — trains + evaluates fully on the leakage-free split

Generated files (not version-controlled, see `.gitignore`):
- `checkpoints/{model}_best.pt` — weights of the best model
- `dataset/`, `dataset_grouped/`, `raw_kaggle_download/` — data (regenerate locally)

Version-controlled in `results/*.json`: training histories, evaluation
metrics, and data-leakage diagnostic results.

## 5. Results (original split)

| Model | Accuracy | Precision | Recall | F1 | Params | Train (min) | Epochs | Infer. (ms/img, CPU) |
|---|---|---|---|---|---|---|---|---|
| MobileNetV2 | 97.35% | 95.49% | 99.40% | 97.40% | 2,226,434 | 184.4 | 16 | 19.67 |
| ResNet18 | 96.90% | 94.58% | 99.50% | 96.98% | 11,177,538 | 100.0 | 8 | 23.89 |

⚠️ **These figures are inflated by data leakage — see section 6.**

## 6. Data leakage investigation

The original Roboflow split assigns individual frames (captured every ~5
minutes) at random to train/valid/test, without grouping by day. Two
images taken a few minutes apart on the same day share the same
background and, for the most part, the same parked cars — a model can
memorize day-specific cues seen in training and "recognize" the same day
at test time.

**Observed extent**: 96/96 dates in the original test split also appear in
train (99/99 for valid); 100% of the test/val crops used come from a
source image whose date is also present in train.

**Fix**: `build_grouped_split.py` pools the images from the 3 Roboflow
splits and redistributes them with
`sklearn.model_selection.GroupShuffleSplit`, using the capture date as the
group (no date shared between the new splits, verified by assertion).

**Results on the leakage-free split** (`dataset_grouped/`, 1500/300/300
images per class) — three scenarios per model: ① original (contaminated),
② original checkpoint re-evaluated without retraining, ③ new model
trained and evaluated entirely on the leakage-free split:

| Model | ① Original | ② Cross-eval | ③ End-to-end clean |
|---|---|---|---|
| MobileNetV2 | 97.35% | 91.33% | 91.00% |
| ResNet18 | 96.90% | 91.00% | 90.50% |

② and ③ converge for both models (gap < 1 point), confirming that leakage
inflated accuracy by **~6 points** for both architectures. **~91% is the
reference accuracy to report**, not ~97%. The relative ranking (MobileNetV2
slightly ahead, with 5× fewer parameters) is unchanged on the clean split.

Full details (precision/recall/F1, confusion matrices) in
`results/leakage_crosseval_*.json` and `results/leakage_diagnostic_*.json`.

## 7. Code conventions

- All hyperparameters go through `config.py`, never hardcoded in the scripts
- Class 0 = "empty", class 1 = "occupied" (ImageFolder's alphabetical order) —
  don't change the order in `config.CLASS_NAMES` without adapting
  `evaluate.py` (the FP/FN computation assumes empty=0, occupied=1)
- Light data augmentation during training (horizontal flip + color jitter,
  see `utils.get_transforms`)
