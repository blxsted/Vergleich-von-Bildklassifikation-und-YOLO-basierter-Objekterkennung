"""
Data-leakage diagnostic: rebuilds a train/val/test split WITHOUT leakage,
grouping by capture date (camera session) instead of keeping the original
Roboflow split (contaminated - see investigation in README.md).

All images from the 3 Roboflow splits (train/valid/test) are pooled
together, then redistributed by date via GroupShuffleSplit (no date can
end up in two splits at once). The jpg files stay physically in their
original folders ("raw_kaggle_download/archive/{train,valid,test}");
this script only decides which image goes into which new split and
crops the corresponding parking spaces.

Usage:
    python build_grouped_split.py --source raw_kaggle_download/archive \
        --out dataset_grouped --per-class-train 1500 --per-class-val 300 --per-class-test 300
"""

import argparse
import json
import os
import re

from PIL import Image
from sklearn.model_selection import GroupShuffleSplit

import config

DATE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_")
CATEGORY_TO_CLASS = {1: "empty", 2: "occupied"}
ORIGINAL_SPLITS = ["train", "valid", "test"]


def load_all_images(source_dir):
    """Collects all images from the 3 Roboflow splits with their date and
    their original split (to locate the jpg file on disk)."""
    records = []  # {orig_split, image_id, file_name, date, anns}
    for orig_split in ORIGINAL_SPLITS:
        with open(os.path.join(source_dir, orig_split, "_annotations.coco.json")) as f:
            coco = json.load(f)
        anns_by_image = {}
        for ann in coco["annotations"]:
            anns_by_image.setdefault(ann["image_id"], []).append(ann)
        for img in coco["images"]:
            m = DATE_RE.match(img["file_name"])
            date = m.group(1) if m else f"unknown_{img['file_name']}"
            records.append({
                "orig_split": orig_split,
                "image_id": img["id"],
                "file_name": img["file_name"],
                "date": date,
                "anns": anns_by_image.get(img["id"], []),
            })
    return records


def grouped_split(records, seed):
    dates = [r["date"] for r in records]

    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval_idx, test_idx = next(gss1.split(records, groups=dates))

    trainval_records = [records[i] for i in trainval_idx]
    trainval_dates = [dates[i] for i in trainval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=seed)  # ~15% of the original
    train_idx, val_idx = next(gss2.split(trainval_records, groups=trainval_dates))

    train_records = [trainval_records[i] for i in train_idx]
    val_records = [trainval_records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    train_dates = {r["date"] for r in train_records}
    val_dates = {r["date"] for r in val_records}
    test_dates = {r["date"] for r in test_records}
    assert not (train_dates & val_dates), "Leakage: dates shared between train/val"
    assert not (train_dates & test_dates), "Leakage: dates shared between train/test"
    assert not (val_dates & test_dates), "Leakage: dates shared between val/test"

    print(f"New split (grouped by date, {len(set(dates))} unique dates total):")
    print(f"  train: {len(train_records)} images, {len(train_dates)} dates")
    print(f"  val:   {len(val_records)} images, {len(val_dates)} dates")
    print(f"  test:  {len(test_records)} images, {len(test_dates)} dates")
    print(f"  Check: no date shared between the 3 splits -> OK")

    return {"train": train_records, "val": val_records, "test": test_records}


def build_crops(source_dir, split_name, records, out_dir, per_class_limit, seed):
    import random
    anns_by_category = {1: [], 2: []}
    for rec in records:
        for ann in rec["anns"]:
            if ann["category_id"] in anns_by_category:
                anns_by_category[ann["category_id"]].append((rec, ann))

    rng = random.Random(seed)
    selected = []
    for cat, items in anns_by_category.items():
        rng.shuffle(items)
        selected.extend(items[:per_class_limit])

    for class_name in CATEGORY_TO_CLASS.values():
        os.makedirs(os.path.join(out_dir, class_name), exist_ok=True)

    # Group by source image so each file is opened only once
    by_image = {}
    for rec, ann in selected:
        by_image.setdefault((rec["orig_split"], rec["image_id"]), {"rec": rec, "anns": []})
        by_image[(rec["orig_split"], rec["image_id"])]["anns"].append(ann)

    counts = {"empty": 0, "occupied": 0}
    for (orig_split, image_id), bundle in by_image.items():
        rec = bundle["rec"]
        img_path = os.path.join(source_dir, orig_split, rec["file_name"])
        if not os.path.exists(img_path):
            continue
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            for ann in bundle["anns"]:
                x, y, w, h = ann["bbox"]
                x0, y0 = int(x), int(y)
                x1, y1 = int(round(x + w)), int(round(y + h))
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                class_name = CATEGORY_TO_CLASS[ann["category_id"]]
                crop = im.crop((x0, y0, x1, y1))
                out_name = f"{orig_split}_{image_id}_{ann['id']}.jpg"
                crop.save(os.path.join(out_dir, class_name, out_name), quality=90)
                counts[class_name] += 1

    print(f"{split_name}: {counts} crops saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--per-class-train", type=int, default=1500)
    parser.add_argument("--per-class-val", type=int, default=300)
    parser.add_argument("--per-class-test", type=int, default=300)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    records = load_all_images(args.source)
    splits = grouped_split(records, args.seed)

    limits = {"train": args.per_class_train, "val": args.per_class_val, "test": args.per_class_test}
    for split_name, recs in splits.items():
        out_dir = os.path.join(args.out, split_name)
        build_crops(args.source, split_name, recs, out_dir, limits[split_name], args.seed)


if __name__ == "__main__":
    main()
