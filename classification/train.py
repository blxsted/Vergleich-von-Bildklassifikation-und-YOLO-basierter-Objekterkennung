"""
Trains a model (mobilenet_v2 or resnet18) on the PKLot dataset.

Usage:
    python train.py --model mobilenet_v2
    python train.py --model resnet18
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from torch.optim import AdamW

import config
from utils import set_seed, get_dataloaders, build_model, count_parameters


def run_epoch(model, loader, criterion, optimizer=None):
    """One epoch of training (if optimizer is given) or evaluation."""
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


def train_model(model_name: str):
    set_seed()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.CHECKPOINTS_DIR, exist_ok=True)

    print(f"\n{'='*60}\nTraining {model_name} on {config.DEVICE}\n{'='*60}")

    train_loader, val_loader, _ = get_dataloaders()
    model = build_model(model_name)
    print(f"Trainable parameters: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    checkpoint_path = f"{config.CHECKPOINTS_DIR}/{model_name}_best.pt"
    start_time = time.time()
    progress_path = f"{config.RESULTS_DIR}/{model_name}_progress.json"

    for epoch in range(1, config.EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:02d}/{config.EPOCHS} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> New best model saved ({checkpoint_path})", flush=True)
        else:
            epochs_without_improvement += 1

        # Real-time tracking: overwritten every epoch so progress can be
        # followed while training runs in the background.
        with open(progress_path, "w") as f:
            json.dump({
                "model_name": model_name,
                "epoch": epoch,
                "total_epochs": config.EPOCHS,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_val_acc,
                "epochs_without_improvement": epochs_without_improvement,
                "elapsed_seconds": round(time.time() - start_time, 1),
            }, f, indent=2)

        if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
            print(f"Early stopping after {epoch} epochs "
                  f"(no improvement for {config.EARLY_STOPPING_PATIENCE} epochs).", flush=True)
            break

    training_time = time.time() - start_time

    # Save training history + metadata
    result = {
        "model_name": model_name,
        "best_val_acc": best_val_acc,
        "epochs_trained": len(history["train_loss"]),
        "training_time_seconds": round(training_time, 1),
        "trainable_parameters": count_parameters(model),
        "hyperparameters": {
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "optimizer": config.OPTIMIZER,
            "early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
        },
        "history": history,
    }

    out_path = f"{config.RESULTS_DIR}/{model_name}_train_history.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nTraining finished in {training_time/60:.1f} min. "
          f"Best val_acc: {best_val_acc:.4f}")
    print(f"History saved: {out_path}")

    return checkpoint_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                         choices=config.MODELS_TO_TRAIN,
                         help="Model to train: mobilenet_v2 or resnet18")
    args = parser.parse_args()
    train_model(args.model)
