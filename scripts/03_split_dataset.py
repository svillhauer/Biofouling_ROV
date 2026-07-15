#!/usr/bin/env python3
"""
Splits labeled images into data/dataset/{images,labels}/{train,val,test} in
YOLO format, for a multi-class biofouling detector (see data.yaml for the
current class list - one class per organism).

Every species is a real trainable class here (unlike the single-species +
suppressed-look-alike project this pipeline was adapted from) - there's no
"hard negative" special-casing, since the goal is to identify species
specifically, not just suppress one look-alike.

Splitting is done PER CLASS, not globally: each class's labeled images are
grouped by GBIF record ID (so near-duplicate photos of the same occurrence
never leak across splits) and independently divided into train/val/test at
the same fractions, then merged. This matters because classes have very
different amounts of labeled data (some inherited from an earlier project,
some brand new) - a global split could easily leave a small class with zero
val/test examples.

Confirmed-background images (empty label file - no organism visible) are
pooled across all source folders and split the same way; they're useful
negative examples for every class alike, not tied to whichever folder they
happened to be downloaded into.

Usage:
    python scripts/03_split_dataset.py
    python scripts/03_split_dataset.py --val-frac 0.15 --test-frac 0.1
"""
import argparse
import random
import shutil
from pathlib import Path
from collections import defaultdict

import yaml

ROOT = Path(__file__).resolve().parent.parent
LABELS_DIR = ROOT / "data" / "labels_yolo_seg"
DATASET_DIR = ROOT / "data" / "dataset"
IMAGES_DIR = ROOT / "data" / "images"
ROV_FRAMES_DIR = ROOT / "data" / "rov_frames"


def source_dirs():
    dirs = [ROV_FRAMES_DIR]
    if IMAGES_DIR.exists():
        dirs.extend(sorted(p for p in IMAGES_DIR.iterdir() if p.is_dir()))
    return dirs


def find_image(stem: str, dirs):
    for d in dirs:
        if not d.exists():
            continue
        for ext in (".jpg", ".jpeg", ".png"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def group_key(stem: str) -> str:
    # gbifID_<index> -> group by gbifID so near-duplicate photos of the same
    # occurrence record stay together in one split.
    return stem.split("_")[0]


def stratified_split(groups: dict, val_frac: float, test_frac: float, seed: int):
    """Splits one class's (or pool's) grouped (img, lbl) pairs into train/val/test."""
    keys = list(groups.keys())
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_val = max(1, int(n * val_frac)) if n > 3 else 0
    n_test = max(1, int(n * test_frac)) if n > 5 else 0
    val_keys = set(keys[:n_val])
    test_keys = set(keys[n_val:n_val + n_test])

    out = {"train": [], "val": [], "test": []}
    for k in keys:
        split = "val" if k in val_keys else "test" if k in test_keys else "train"
        out[split].extend(groups[k])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link", action="store_true",
                     help="Symlink instead of copy (saves disk space)")
    args = ap.parse_args()

    data_yaml = yaml.safe_load((ROOT / "data.yaml").read_text())
    class_names = data_yaml["names"]

    label_files = sorted(LABELS_DIR.glob("*.txt"))
    label_files = [p for p in label_files if p.name != "_progress.txt"]
    if not label_files:
        print(f"No labels found in {LABELS_DIR}. Run the labeling tool first.")
        return

    dirs = source_dirs()
    class_groups = defaultdict(lambda: defaultdict(list))  # class_id(str) -> group_key -> [(img, lbl)]
    bg_groups = defaultdict(list)  # group_key -> [(img, lbl)] for confirmed-background images
    missing = []

    for lp in label_files:
        img = find_image(lp.stem, dirs)
        if img is None:
            missing.append(lp.stem)
            continue
        lines = [l for l in lp.read_text().splitlines() if l.strip()]
        classes = {l.split()[0] for l in lines}
        if not classes:
            bg_groups[group_key(lp.stem)].append((img, lp))
            continue
        # An image's instances are all written to its label file regardless;
        # this only decides which single split-pool the whole image counts
        # toward. In practice every image has one species, but if a file ever
        # mixes classes, group it under the lowest class id rather than
        # risking the same image landing in two different splits.
        primary = min(classes, key=lambda c: int(c))
        if len(classes) > 1:
            print(f"Warning: {lp.name} has multiple classes ({sorted(classes)}) - "
                  f"grouping under class {primary} for split purposes only.")
        class_groups[primary][group_key(lp.stem)].append((img, lp))

    if missing:
        print(f"Warning: {len(missing)} label files have no matching image, skipping them.")

    splits = {"train": [], "val": [], "test": []}

    for class_id in sorted(class_names, key=lambda c: str(c)):
        groups = class_groups.get(str(class_id), {})
        if not groups:
            print(f"class {class_id} ({class_names[class_id]}): 0 labeled images yet")
            continue
        class_splits = stratified_split(groups, args.val_frac, args.test_frac, args.seed)
        for split, pairs in class_splits.items():
            splits[split].extend(pairs)
        n_total = sum(len(v) for v in class_splits.values())
        print(f"class {class_id} ({class_names[class_id]}): {n_total} images -> "
              f"train {len(class_splits['train'])}, val {len(class_splits['val'])}, "
              f"test {len(class_splits['test'])}")

    # Confirmed-background images: pooled across all species, split the same
    # way, useful as negatives for every class.
    if bg_groups:
        bg_splits = stratified_split(bg_groups, args.val_frac, args.test_frac, args.seed + 1)
        for split, pairs in bg_splits.items():
            splits[split].extend(pairs)
        print(f"confirmed-background: {sum(len(v) for v in bg_splits.values())} images -> "
              f"train {len(bg_splits['train'])}, val {len(bg_splits['val'])}, "
              f"test {len(bg_splits['test'])}")

    for split, pairs in splits.items():
        img_out = DATASET_DIR / "images" / split
        lbl_out = DATASET_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img, lbl in pairs:
            dst_img = img_out / img.name
            dst_lbl = lbl_out / lbl.name
            if args.link:
                if dst_img.exists() or dst_img.is_symlink():
                    dst_img.unlink()
                if dst_lbl.exists() or dst_lbl.is_symlink():
                    dst_lbl.unlink()
                dst_img.symlink_to(img.resolve())
                dst_lbl.symlink_to(lbl.resolve())
            else:
                shutil.copy2(img, dst_img)
                shutil.copy2(lbl, dst_lbl)
        class_counts = defaultdict(int)
        n_empty = 0
        for _, lbl in pairs:
            lines = [l for l in lbl.read_text().splitlines() if l.strip()]
            if not lines:
                n_empty += 1
            for line in lines:
                class_counts[line.split()[0]] += 1
        counts_str = ", ".join(f"class {c}: {n}" for c, n in sorted(class_counts.items()))
        print(f"{split}: {len(pairs)} images ({n_empty} confirmed-background) - instances: {counts_str}")

    print(f"Dataset written to {DATASET_DIR}")


if __name__ == "__main__":
    main()
