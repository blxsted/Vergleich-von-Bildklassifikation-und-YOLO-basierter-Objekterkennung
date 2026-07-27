"""
Data-leakage diagnostic: trains a model (mobilenet_v2 or resnet18) on
dataset_grouped/ (leakage-free split, grouped by capture date - see
build_grouped_split.py) and evaluates it on the corresponding test set.

Same hyperparameters as the official train.py (AdamW lr=0.001, batch=16,
up to config.EPOCHS with early stopping config.EARLY_STOPPING_PATIENCE),
for a direct comparison with the results obtained on the original
(contaminated) split documented in results/{model}_eval.json.

Usage:
    python leakage_diagnostic_train.py --model mobilenet_v2
    python leakage_diagnostic_train.py --model resnet18
"""

import argparse
import json
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import config
from utils import get_transforms, build_model, set_seed, count_parameters

ORIGINAL_LEAKY_ACCURACY = {
    "mobilenet_v2": 0.9735,
    "resnet18": 0.969,
}


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            if is_train:
                optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def main(model_name: str):
    set_seed()

    train_ds = datasets.ImageFolder("dataset_grouped/train", transform=get_transforms(train=True))
    val_ds = datasets.ImageFolder("dataset_grouped/val", transform=get_transforms(train=False))
    test_ds = datasets.ImageFolder("dataset_grouped/test", transform=get_transforms(train=False))
    assert train_ds.classes == config.CLASS_NAMES == val_ds.classes == test_ds.classes

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    model = build_model(model_name)
    print(f"Trainable parameters: {count_parameters(model):,}", flush=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    best_state = None
    start = time.time()
    epoch = 0

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        print(f"Epoch {epoch:02d}/{config.EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"| val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  -> new best (val_acc={val_acc:.4f})", flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping after {epoch} epochs.", flush=True)
                break

    training_time = time.time() - start
    model.load_state_dict(best_state)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds).tolist()

    result = {
        "note": "Data-leakage diagnostic: training + evaluation on the date-grouped split (leakage-free)",
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "epochs_trained": epoch,
        "training_time_seconds": round(training_time, 1),
        "test_accuracy": round(acc, 4),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_f1": round(f1, 4),
        "confusion_matrix": cm,
        "n_test_samples": len(all_labels),
        "comparison_original_leaky_test_accuracy": ORIGINAL_LEAKY_ACCURACY[model_name],
    }
    out_path = f"{config.RESULTS_DIR}/leakage_diagnostic_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== DIAGNOSTIC RESULT (leakage-free, date-grouped split) ===", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    print(f"\nSaved: {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                         choices=config.MODELS_TO_TRAIN,
                         help="Model to train on the leakage-free split: mobilenet_v2 or resnet18")
    args = parser.parse_args()
    main(args.model)
