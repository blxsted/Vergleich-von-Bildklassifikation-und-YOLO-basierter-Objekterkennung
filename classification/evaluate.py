"""
Evaluates a trained model on the test set:
- Accuracy, Precision, Recall, F1
- Confusion matrix
- 10 error examples (false positives / false negatives)
- Metrics per weather condition (if detectable in the file path)

Usage:
    python evaluate.py --model mobilenet_v2
    python evaluate.py --model resnet18
"""

import argparse
import json
import os
import re

import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)

import config
from utils import set_seed, get_dataloaders, build_model

WEATHER_KEYWORDS = ["sunny", "cloudy", "rainy"]


def detect_weather(path: str):
    """Looks for a weather keyword in the file path (PKLot convention)."""
    path_lower = path.lower()
    for kw in WEATHER_KEYWORDS:
        if kw in path_lower:
            return kw
    return "unknown"


def evaluate_model(model_name: str):
    set_seed()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    checkpoint_path = f"{config.CHECKPOINTS_DIR}/{model_name}_best.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run train.py --model {model_name} first"
        )

    print(f"\n{'='*60}\nEvaluating {model_name} on the test set\n{'='*60}")

    _, _, test_loader = get_dataloaders()
    model = build_model(model_name)
    model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
    model.eval()

    all_preds, all_labels, all_paths, all_confidences = [], [], [], []

    # Retrieve file paths in the same order as the DataLoader (shuffle=False)
    sample_paths = [s[0] for s in test_loader.dataset.samples]

    idx = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)

            batch_size = images.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_confidences.extend(confidences.cpu().tolist())
            all_paths.extend(sample_paths[idx: idx + batch_size])
            idx += batch_size

    # --- Overall metrics ---
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")
    cm = confusion_matrix(all_labels, all_preds).tolist()

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"Confusion Matrix: {cm}")

    # --- Error examples (up to 5 FP + 5 FN) ---
    false_positives, false_negatives = [], []
    for path, true_label, pred_label, conf in zip(all_paths, all_labels, all_preds, all_confidences):
        if pred_label != true_label:
            entry = {
                "path": path,
                "true_label": config.CLASS_NAMES[true_label],
                "predicted_label": config.CLASS_NAMES[pred_label],
                "confidence": round(conf, 4),
            }
            # FP: predicted "occupied" while true = "empty" (class 0 = empty, 1 = occupied)
            if true_label == 0 and pred_label == 1 and len(false_positives) < 5:
                false_positives.append(entry)
            elif true_label == 1 and pred_label == 0 and len(false_negatives) < 5:
                false_negatives.append(entry)

    # --- Metrics per weather condition ---
    weather_metrics = {}
    weather_groups = {}
    for path, true_label, pred_label in zip(all_paths, all_labels, all_preds):
        w = detect_weather(path)
        weather_groups.setdefault(w, {"labels": [], "preds": []})
        weather_groups[w]["labels"].append(true_label)
        weather_groups[w]["preds"].append(pred_label)

    for w, data in weather_groups.items():
        if len(set(data["labels"])) < 2:
            # avoid sklearn errors when only one class is present
            w_acc = accuracy_score(data["labels"], data["preds"])
            weather_metrics[w] = {"accuracy": round(w_acc, 4), "n_samples": len(data["labels"])}
        else:
            weather_metrics[w] = {
                "accuracy": round(accuracy_score(data["labels"], data["preds"]), 4),
                "precision": round(precision_score(data["labels"], data["preds"], average="binary"), 4),
                "recall": round(recall_score(data["labels"], data["preds"], average="binary"), 4),
                "f1": round(f1_score(data["labels"], data["preds"], average="binary"), 4),
                "n_samples": len(data["labels"]),
            }

    if list(weather_groups.keys()) == ["unknown"]:
        print("\nNote: no weather condition detected in the file paths. "
              "Per-weather metrics are not available. "
              "Make sure file/folder names contain 'Sunny', 'Cloudy' or 'Rainy' "
              "if you want this breakdown (PKLot convention).")

    # --- Save the full report ---
    report = {
        "model_name": model_name,
        "test_accuracy": round(acc, 4),
        "test_precision": round(precision, 4),
        "test_recall": round(recall, 4),
        "test_f1": round(f1, 4),
        "confusion_matrix": {
            "labels": config.CLASS_NAMES,
            "matrix": cm,  # [[TN, FP], [FN, TP]]
        },
        "error_examples": {
            "false_positives": false_positives,
            "false_negatives": false_negatives,
        },
        "metrics_per_weather": weather_metrics,
        "n_test_samples": len(all_labels),
    }

    out_path = f"{config.RESULTS_DIR}/{model_name}_eval.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nEvaluation report saved: {out_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                         choices=config.MODELS_TO_TRAIN,
                         help="Model to evaluate: mobilenet_v2 or resnet18")
    args = parser.parse_args()
    evaluate_model(args.model)
