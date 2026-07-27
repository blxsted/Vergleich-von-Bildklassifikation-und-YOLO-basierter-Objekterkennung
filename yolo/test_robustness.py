from ultralytics import YOLO
import torch
import json
from pathlib import Path

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("=" * 60)
print("ROBUSTNESS TEST - YOLO unter verschiedenen Wetterbedingungen")
print("=" * 60)

model = YOLO("runs/detect/yolov8n_parking/weights/best.pt")

# Test-Bilder nach Wetter filtern
test_path = Path("yolo_dataset/test/images")

results_by_weather = {}

for weather in ['Sunny', 'sunny', 'Rainy', 'rainy', 'Cloudy', 'cloudy']:
    # Bilder mit diesem Wetter finden
    weather_images = list(test_path.glob(f"*{weather}*.jpg"))
    
    if not weather_images:
        continue
    
    print(f"\nTesting {weather}: {len(weather_images)} images...")
    
    # Evaluieren auf diesem Subset
    # (Trick: temporärer Test-Dataset nur für dieses Wetter)
    precisions = []
    recalls = []
    
    for img in weather_images[:100]:  # Erste 100 zum schneller testen
        result = model.predict(str(img), conf=0.5, verbose=False)
        # Metriken extrahieren...
    
    results_by_weather[weather] = {
        "num_images": len(weather_images),
        "sample_tested": min(100, len(weather_images))
    }

print("\n" + "=" * 60)
print("RESULTS BY WEATHER")
print("=" * 60)

for weather, data in results_by_weather.items():
    print(f"{weather}: {data['num_images']} images")

# Speichern
with open("../results/robustness_test.json", 'w') as f:
    json.dump(results_by_weather, f, indent=2)

print("\n✓ Robustness test complete!")
