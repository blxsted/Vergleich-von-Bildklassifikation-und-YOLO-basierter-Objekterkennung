from ultralytics import YOLO
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("=" * 60)
print("YOLOv8n TRAINING - MAC")
print("=" * 60)
print(f"Device: {device}\n")

model = YOLO('yolov8n.pt')

print("Starting training (15 epochs)...")
results = model.train(
    data='configs/dataset.yaml',
    epochs=15,
    imgsz=640,
    batch=8,
    device=device, # if device == "mps" else "cpu",
    patience=10,
    save=True,
    project='runs/detect',
    name='yolov8n_parking',
)

print("\n" + "=" * 60)
print("✓ TRAINING COMPLETED!")
print("=" * 60)
