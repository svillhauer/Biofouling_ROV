#!/usr/bin/env python3
"""
Generates ROV-like degraded copies of training images.

Purpose: the GBIF training photos are sharp, well-lit, in-air/tidepool macro
shots, but the actual ROV footage every class needs to be detected in is
blurry, hazy, color-cast, and much lower-detail. This closes part of that
gap by producing degraded duplicates (downscale/upscale detail loss, haze,
reduced saturation/contrast, color-cast, noise) of the existing training
images, added alongside the originals so the model sees both conditions.

The degrade() parameters below are the ones used to produce the training
data for runs/segment/train-4 (recovered from a stale __pycache__ copy
after the function was later edited and the on-disk _degr images had
already been regenerated with the newer, lighter parameters - if you
re-tune these again, keep a copy of what train-4 actually used).

Confirmed-background (empty-label) images are degraded too, alongside
whichever classes are targeted, at the same rate. This matters: an earlier,
unrelated experiment (see scripts/04_train.py's --quality-degrade, now
disabled by default) applied heavy blur/noise augmentation globally during
training and it taught the model to associate blur itself with "organism
present," because degraded positives vastly outnumbered degraded negatives.
Degrading backgrounds in lockstep here keeps that ratio intact.

Only touches data/dataset/images/train + data/dataset/labels/train (never
val/test) - run this AFTER scripts/03_split_dataset.py. Labels are copied
unchanged since every transform here is purely photometric or a resize
round-trip back to the original dimensions, so polygon coordinates
(normalized 0-1) stay valid.

Usage:
    python scripts/06_degrade_domain_match.py
    python scripts/06_degrade_domain_match.py --class-ids 2 --suffix _degr
"""
import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "data" / "dataset"


def degrade(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]

    # 1. Downscale/upscale round-trip for detail loss, plus an extra
    #    Gaussian blur pass on top.
    scale = rng.uniform(0.08, 0.18)
    small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))),
                        interpolation=cv2.INTER_LINEAR)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    k = int(rng.uniform(9, 17)) | 1
    img = cv2.GaussianBlur(img, (k, k), 0)

    # 2. Reduce saturation & contrast - underwater light is muted/hazy.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rng.uniform(0.15, 0.35)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    mean = img.mean()
    contrast = rng.uniform(0.35, 0.5)
    img = np.clip((img.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)

    # 3. Color cast - blend in a blurred, tinted "haze veil" rather than
    #    flatly tinting the sharp image.
    tint = np.full_like(img, (95, 70, 45), dtype=np.uint8)  # BGR
    veil = cv2.GaussianBlur(img, (0, 0), sigmaX=w * 0.08)
    veil = cv2.addWeighted(veil, 0.4, tint, 0.6, 0)
    alpha = rng.uniform(0.45, 0.65)
    img = cv2.addWeighted(img, 1 - alpha, veil, alpha, 0)

    # 4. Sensor/turbidity noise.
    noise = rng.normal(0, rng.uniform(8, 16), img.shape).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return img


def find_image(stem, img_dir):
    for ext in (".jpg", ".jpeg", ".png"):
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--class-ids", default="0,1,2,3",
                     help="Comma-separated class IDs to degrade (default: all classes)")
    ap.add_argument("--suffix", default="_degr")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-backgrounds", action="store_true",
                     help="Skip degrading confirmed-background images too "
                          "(NOT recommended - see docstring on blur/class imbalance)")
    args = ap.parse_args()
    target_classes = set(args.class_ids.split(","))

    data_yaml = yaml.safe_load((ROOT / "data.yaml").read_text())
    class_names = data_yaml["names"]

    img_dir = DATASET_DIR / "images" / "train"
    lbl_dir = DATASET_DIR / "labels" / "train"
    rov_frames_dir = ROOT / "data" / "rov_frames"
    rov_stems = {p.stem for p in rov_frames_dir.iterdir()} if rov_frames_dir.exists() else set()

    rng = np.random.default_rng(args.seed)
    made = 0
    made_bg = 0
    for lbl_path in sorted(lbl_dir.glob("*.txt")):
        if lbl_path.stem.endswith(args.suffix):
            continue  # don't re-degrade an already-degraded file
        if lbl_path.stem in rov_stems:
            continue  # already real ROV footage - no need to simulate the domain gap

        lines = [l for l in lbl_path.read_text().splitlines() if l.strip()]
        classes = {l.split()[0] for l in lines}
        is_background = not classes
        if not is_background and not (classes & target_classes):
            continue

        img_path = find_image(lbl_path.stem, img_dir)
        if img_path is None:
            continue

        out_stem = f"{lbl_path.stem}{args.suffix}"
        out_img_path = img_dir / f"{out_stem}{img_path.suffix}"
        out_lbl_path = lbl_dir / f"{out_stem}.txt"
        if out_img_path.exists():
            continue  # already generated
        if is_background and args.no_backgrounds:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        degraded = degrade(img, rng)
        cv2.imwrite(str(out_img_path), degraded)
        shutil.copy2(lbl_path, out_lbl_path)
        if is_background:
            made_bg += 1
        else:
            made += 1

    names = ", ".join(class_names[int(c)] for c in sorted(target_classes))
    print(f"Generated {made} degraded copies for class(es): {names}")
    print(f"Generated {made_bg} degraded confirmed-background copies")
    print(f"Written into {img_dir} / {lbl_dir}")


if __name__ == "__main__":
    main()
