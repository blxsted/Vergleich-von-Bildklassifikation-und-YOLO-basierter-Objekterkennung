"""
Data-leakage diagnostic: evaluates an already-trained checkpoint from the
original (potentially contaminated) split on the leakage-free test set of
dataset_grouped/ (grouped by capture date - see build_grouped_split.py),
without any retraining.

Used to check whether a model learned on the original split generalizes
just as well to days it never saw, and to compare this figure with the
one obtained by training and evaluating entirely on the leakage-free
split (leakage_diagnostic_train.py), confirming the gap isn't an artifact
of a single run.

Usage:
    python leakage_crosseval.py --model mobilenet_v2
    python leakage_crosseval.py --model resnet18
"""

import argparse
import json

import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import config
from utils import get_transforms, build_model, set_seed


def main(model_name: str):
    set_seed()

    test_ds = datasets.ImageFolder("dataset_grouped/test", transform=get_transforms(train=False))
    assert test_ds.classes == config.CLASS_NAMES

    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    checkpoint_path = f"{config.CHECKPOINTS_DIR}/{model_name}_best.pt"
    model = build_model(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
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

    print(f"n_test = {len(all_labels)}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Confusion matrix: {cm}")

    result = {
        "note": "Checkpoint trained on the original (potentially contaminated) split, "
                "evaluated on the leakage-free test set dataset_grouped/test (dates never seen in train)",
        "model_name": model_name,
        "checkpoint": checkpoint_path,
        "test_accuracy": round(acc, 4),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_f1": round(f1, 4),
        "confusion_matrix": cm,
        "n_test_samples": len(all_labels),
    }
    out_path = f"{config.RESULTS_DIR}/leakage_crosseval_{model_name}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                         choices=config.MODELS_TO_TRAIN,
                         help="Model (existing checkpoint) to evaluate on the leakage-free split")
    args = parser.parse_args()
    main(args.model)
