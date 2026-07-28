import json
from pathlib import Path

# JSON laden
with open("dataset/test/_annotations.coco.json") as f:
    data = json.load(f)

# Kategorien zählen
empty_count = 0
occupied_count = 0

for annot in data['annotations']:
    category_id = annot['category_id']
    
    if category_id == 1:  # space-empty
        empty_count += 1
    elif category_id == 2:  # space-occupied
        occupied_count += 1

total = empty_count + occupied_count
baseline = occupied_count / total

print(f"Test-Set:")
print(f"  Empty spaces: {empty_count}")
print(f"  Occupied spaces: {occupied_count}")
print(f"  Total: {total}")
print(f"\nBASELINE (always 'occupied'): {baseline:.1%}")

# Speichern
import json
with open("results/baseline.json", 'w') as f:
    json.dump({"baseline": baseline, "empty": empty_count, "occupied": occupied_count}, f, indent=2)
