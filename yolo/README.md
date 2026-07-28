# E2 — Object Detection: YOLOv8n (PKLot)

Real-time object detection using YOLOv8n for parking-lot occupancy detection on the **PKLot** dataset. This folder covers experiment E2 (parking space detection via bounding boxes), alongside experiment E1 (binary image classification, see the `classification/` folder).

## 1. Dataset origin and preparation

The source dataset ([PKLot via Kaggle/Roboflow](https://public.roboflow.ai/object-detection/pklot))
provides **full-scene images** (640×640) with **COCO** annotations — one bounding box per parking space, category `space-empty` or `space-occupied`. The dataset itself is **not included in this repository** (too large, see `.gitignore`) — it must be downloaded once, locally, before running anything below.

### Getting the data

1. Download the zip from Kaggle: **[PKLot Dataset](https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset/data)**
   (requires a free Kaggle account; use the page's "Download" button).
2. Extract and convert the COCO annotations to YOLO format (txt label files), organizing into:

   ```
   yolo_dataset/
   ├── train/
   │   ├── images/     (all training images)
   │   └── labels/     (corresponding .txt label files)
   ├── valid/
   │   ├── images/
   │   └── labels/
   └── test/
       ├── images/
       └── labels/
   ```

   (Tools like Roboflow's export or the `ultralytics` library can automate this conversion.)

3. Continue with the workflow below.

## 2. Installation

```bash
python -m venv venv
venv/Scripts/activate          # Windows
source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt
```

## 3. Full workflow

```bash
# Compute baseline performance (always predicting 'occupied')
python baseline.py

# Training (15 epochs, with early stopping patience=10)
python train_yolo_mac.py

# Evaluation on the test set (mAP@0.5, mAP@0.5:0.95, precision, recall)
python evaluate_yolo_mac.py
```

## 4. Project files

- `baseline.py` — computes the baseline metric (always predicting 'occupied')
- `train_yolo_mac.py` — YOLOv8n training with automatic device selection (MPS for Mac, CPU fallback)
  - 15 epochs, batch size 8, image size 640×640
  - Early stopping with patience=10
  - Saves checkpoints to `runs/detect/yolov8n_parking/`
- `evaluate_yolo_mac.py` — evaluates the best checkpoint on the test set
  - Computes mAP@0.5, mAP@0.5:0.95, precision, recall
  - Compares against baseline
  - Saves results to `results/yolo_eval.json`
- `configs/dataset.yaml` — YOLO dataset configuration (paths, class count, class names)
- `yolov8n.pt` — YOLOv8 Nano pretrained weights (auto-downloaded by `ultralytics` on first run)

Generated files (not version-controlled, see `.gitignore`):
- `runs/detect/yolov8n_parking/` — training checkpoints, logs, and validation results
- `results/yolo_eval.json` — final evaluation metrics
- `yolo_dataset/` — data (regenerate locally from Kaggle download)

## 5. Code conventions

- **Device handling**: automatically selects MPS (Apple Silicon) if available, otherwise falls back to CPU
- **Hyperparameters**: all training parameters are hardcoded in `train_yolo_mac.py` (epochs, batch size, image size, patience)
- **Results format**: evaluation metrics are saved as JSON in `results/` for reproducibility and downstream analysis
- **Classes**: single class (index 0) representing a parking space. Category ID 2 in the original COCO data maps to occupied.
