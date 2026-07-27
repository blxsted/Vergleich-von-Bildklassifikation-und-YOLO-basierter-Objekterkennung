"""
Compares MobileNetV2 and ResNet18 on all dimensions relevant to E1:
- Accuracy / Precision / Recall / F1 (test set)
- Parameter count
- Average inference time (CPU and/or GPU depending on availability)
- Training time

Requires having run train.py then evaluate.py for both models beforehand.

Usage:
    python compare.py
"""

import json
import time

import torch

import config
from utils import set_seed, get_dataloaders, build_model


def measure_inference_time(model, loader, n_batches: int = 20) -> float:
    """Average inference time per image, in milliseconds."""
    model.eval()
    times = []
    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= n_batches:
                break
            images = images.to(config.DEVICE)
            if config.DEVICE.type == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _ = model(images)
            if config.DEVICE.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            times.append(elapsed / images.size(0))  # time per image
    return sum(times) / len(times) * 1000  # in ms


def load_json(path: str):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def compare_all():
    set_seed()
    _, _, test_loader = get_dataloaders()

    rows = []
    for model_name in config.MODELS_TO_TRAIN:
        train_report = load_json(f"{config.RESULTS_DIR}/{model_name}_train_history.json")
        eval_report = load_json(f"{config.RESULTS_DIR}/{model_name}_eval.json")

        if train_report is None or eval_report is None:
            print(f"Missing results for {model_name}. "
                  f"Run train.py and evaluate.py for this model first.")
            continue

        checkpoint_path = f"{config.CHECKPOINTS_DIR}/{model_name}_best.pt"
        model = build_model(model_name)
        model.load_state_dict(torch.load(checkpoint_path, map_location=config.DEVICE))
        inference_ms = measure_inference_time(model, test_loader)

        rows.append({
            "model": model_name,
            "test_accuracy": eval_report["test_accuracy"],
            "test_precision": eval_report["test_precision"],
            "test_recall": eval_report["test_recall"],
            "test_f1": eval_report["test_f1"],
            "trainable_parameters": train_report["trainable_parameters"],
            "training_time_min": round(train_report["training_time_seconds"] / 60, 1),
            "epochs_trained": train_report["epochs_trained"],
            "inference_time_ms_per_image": round(inference_ms, 3),
        })

    if not rows:
        print("No results available. Train and evaluate at least one model first.")
        return

    # --- Print table ---
    print("\n" + "=" * 100)
    print("COMPARISON MobileNetV2 vs ResNet18")
    print("=" * 100)
    header = f"{'Model':<15}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}{'Params':>14}{'Train(min)':>12}{'Epochs':>8}{'Infer(ms)':>11}"
    print(header)
    print("-" * 100)
    for r in rows:
        print(f"{r['model']:<15}{r['test_accuracy']:>8.4f}{r['test_precision']:>8.4f}"
              f"{r['test_recall']:>8.4f}{r['test_f1']:>8.4f}{r['trainable_parameters']:>14,}"
              f"{r['training_time_min']:>12.1f}{r['epochs_trained']:>8}{r['inference_time_ms_per_image']:>11.3f}")
    print("=" * 100)

    out_path = f"{config.RESULTS_DIR}/comparison_mobilenet_vs_resnet18.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nComparison saved: {out_path}")


if __name__ == "__main__":
    compare_all()
