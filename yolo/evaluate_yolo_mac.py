from ultralytics import YOLO
import torch
import json

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("=" * 60)
print("YOLOv8n EVALUATION")
print("=" * 60)
print(f"Device: {device}\n")

model = YOLO("runs/detect/yolov8n_parking/weights/best.pt")

print("Evaluating on test set...")
results = model.val(
    data='configs/dataset.yaml',
    device=device
)

map50 = results.box.map50
map_val = results.box.map
precision = results.box.mp
recall = results.box.mr

print("\n" + "=" * 60)
print("YOLO RESULTS")
print("=" * 60)
print(f"mAP@0.5:      {map50:.3f}")
print(f"mAP@0.5:0.95: {map_val:.3f}")
print(f"Precision:    {precision:.3f}")
print(f"Recall:       {recall:.3f}")

# Vergleich mit Baseline
baseline = 0.482
print(f"\n" + "=" * 60)
print("COMPARISON WITH BASELINE")
print("=" * 60)
print(f"Baseline:     {baseline:.1%} (always 'occupied')")
print(f"YOLO mAP@0.5: {map50:.1%}")
print(f"Improvement:  +{(map50*100 - baseline*100):.1f}%")

# Speichern
with open("./results/yolo_eval.json", 'w') as f:
    json.dump({
        "map50": float(map50),
        "map": float(map_val),
        "precision": float(precision),
        "recall": float(recall),
        "device": device
    }, f, indent=2)

print("\n✓ Results saved to ./results/yolo_eval.json")
