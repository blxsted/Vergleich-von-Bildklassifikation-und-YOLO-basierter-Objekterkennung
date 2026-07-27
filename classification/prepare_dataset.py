"""
Crops the full-scene images of the PKLot dataset (Roboflow export, COCO
format) into individual parking-space crops, sorted into
dataset/{train,val,test}/{empty,occupied} as expected by the pipeline.

The Kaggle zip contains full-scene images (640x640) plus one
_annotations.coco.json file per split (train/valid/test) with one
bounding box per parking space ("space-empty" or "space-occupied").
This script crops each annotated space into an individual image.

Usage:
    python prepare_dataset.py --source raw_kaggle_download/archive \
        --per-class-train 4000 --per-class-val 1000 --per-class-test 1000
"""

import argparse
import json
import os
import random

from PIL import Image

import config

CATEGORY_TO_CLASS = {1: "empty", 2: "occupied"}
SPLIT_TO_OUT_DIR = {
    "train": config.TRAIN_DIR,
    "valid": config.VAL_DIR,
    "test": config.TEST_DIR,
}


def process_split(source_dir: str, split_name: str, out_dir: str, per_class_limit: int, seed: int):
    ann_path = os.path.join(source_dir, split_name, "_annotations.coco.json")
    with open(ann_path) as f:
        coco = json.load(f)

    images_by_id = {img["id"]: img for img in coco["images"]}

    anns_by_category = {1: [], 2: []}
    for ann in coco["annotations"]:
        if ann["category_id"] in anns_by_category:
            anns_by_category[ann["category_id"]].append(ann)

    rng = random.Random(seed)
    selected_anns = []
    for cat, anns in anns_by_category.items():
        rng.shuffle(anns)
        selected_anns.extend(anns[:per_class_limit])

    # Group by image so each source file is opened only once
    anns_by_image = {}
    for ann in selected_anns:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    for class_name in CATEGORY_TO_CLASS.values():
        os.makedirs(os.path.join(out_dir, class_name), exist_ok=True)

    counts = {"empty": 0, "occupied": 0}
    for image_id, anns in anns_by_image.items():
        img_info = images_by_id[image_id]
        img_path = os.path.join(source_dir, split_name, img_info["file_name"])
        if not os.path.exists(img_path):
            continue
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            for ann in anns:
                x, y, w, h = ann["bbox"]
                x0, y0 = int(x), int(y)
                x1, y1 = int(round(x + w)), int(round(y + h))
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                class_name = CATEGORY_TO_CLASS[ann["category_id"]]
                crop = im.crop((x0, y0, x1, y1))
                out_name = f"{image_id}_{ann['id']}.jpg"
                crop.save(os.path.join(out_dir, class_name, out_name), quality=90)
                counts[class_name] += 1

    for class_name, n in counts.items():
        print(f"{split_name}/{class_name}: {n} crops saved to {out_dir}/{class_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                         help="Folder containing train/valid/test with _annotations.coco.json")
    parser.add_argument("--per-class-train", type=int, default=4000)
    parser.add_argument("--per-class-val", type=int, default=1000)
    parser.add_argument("--per-class-test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    limits = {
        "train": args.per_class_train,
        "valid": args.per_class_val,
        "test": args.per_class_test,
    }

    for split_name, out_dir in SPLIT_TO_OUT_DIR.items():
        process_split(args.source, split_name, out_dir, limits[split_name], args.seed)


if __name__ == "__main__":
    main()
