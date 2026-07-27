"""
Central configuration for the MobileNetV2 vs ResNet18 comparison
on the parking-lot occupancy classification task (PKLot).
"""

import torch

# --- Dataset paths ---
DATA_DIR = "dataset"          # expects dataset/train, dataset/val, dataset/test
TRAIN_DIR = f"{DATA_DIR}/train"
VAL_DIR = f"{DATA_DIR}/val"
TEST_DIR = f"{DATA_DIR}/test"

RESULTS_DIR = "results"
CHECKPOINTS_DIR = "checkpoints"

# --- Classes ---
CLASS_NAMES = ["empty", "occupied"]  # must match ImageFolder's alphabetical order
NUM_CLASSES = 2

# --- Training hyperparameters (aligned with the project plan) ---
BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 1e-3
OPTIMIZER = "AdamW"
EARLY_STOPPING_PATIENCE = 5   # epochs without improvement before stopping
IMAGE_SIZE = 224              # standard size for pretrained MobileNetV2 / ResNet18

# --- Misc ---
SEED = 42
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Models to compare ---
MODELS_TO_TRAIN = ["mobilenet_v2", "resnet18"]
