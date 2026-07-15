#!/usr/bin/env python3
"""
Runs the trained multi-class biofouling model over a folder of ROV photos
(or a video) and reports, per image, whether biofouling was detected and
which species - plus an overall summary. This is the "is there biofouling on
this wall" tool: presence/absence falls straight out of a multi-class model
(any class firing = biofouling present), you don't need a separate detector
for that question.

Usage:
    python scripts/05_scan_for_biofouling.py --weights runs/segment/train/weights/best.pt --source rov_photos --save
"""
import argparse
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # headless: never try a GUI backend, only ever save to file

import yaml
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
IMG_EXTS = (".jpg", ".jpeg", ".png")


def prepare_source(source_dir: Path, max_dim: int = 2500) -> Path:
    """Downscales any unusually large image into a temp copy before predict(save=True) -
    full-resolution mask plotting scales with pixel count and can OOM the GPU on an
    outlier photo (seen in practice: a 41-megapixel image among otherwise ~2-4MP ones)."""
    if not source_dir.is_dir():
        return source_dir
    imgs = [p for p in source_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
    oversized = [p for p in imgs if _too_big(p, max_dim)]
    if not oversized:
        return source_dir

    tmp_dir = Path(tempfile.mkdtemp(prefix="biofouling_resized_"))
    for p in imgs:
        dst = tmp_dir / p.name
        if p in oversized:
            with Image.open(p) as im:
                im = im.convert("RGB")
                ratio = max_dim / max(im.size)
                im = im.resize((max(1, int(im.width * ratio)), max(1, int(im.height * ratio))))
                im.save(dst)
        else:
            try:
                dst.symlink_to(p.resolve())
            except OSError:
                shutil.copy2(p, dst)
    print(f"Note: {len(oversized)} image(s) over {max_dim}px downscaled into a temp "
          f"copy to avoid an OOM during mask plotting.")
    return tmp_dir


def _too_big(p: Path, max_dim: int) -> bool:
    try:
        with Image.open(p) as im:
            return max(im.size) > max_dim
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="Path to trained best.pt")
    ap.add_argument("--source", required=True, help="Folder of images (or a video/single image)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--save", action="store_true", help="Save annotated images")
    ap.add_argument("--name", default="biofouling_scan")
    args = ap.parse_args()

    data_yaml = yaml.safe_load((ROOT / "data.yaml").read_text())
    class_names = data_yaml["names"]

    source = Path(args.source)
    predict_source = prepare_source(source) if (args.save and source.is_dir()) else source

    model = YOLO(args.weights)
    results = model.predict(
        source=str(predict_source),
        conf=args.conf,
        save=args.save,
        project=str(ROOT / "runs" / "segment"),
        name=args.name,
        stream=True,
        verbose=False,
    )

    n_images = 0
    n_with_fouling = 0
    species_counts = defaultdict(int)   # class name -> images it appeared in
    species_instance_counts = defaultdict(int)  # class name -> total instances

    print(f"\n{'image':40s}  {'fouling?':9s}  species (max confidence)")
    print("-" * 80)
    for r in results:
        n_images += 1
        name = Path(r.path).name
        if r.boxes is None or len(r.boxes) == 0:
            print(f"{name:40s}  {'no':9s}")
            continue

        n_with_fouling += 1
        cls_ids = r.boxes.cls.tolist()
        confs = r.boxes.conf.tolist()
        per_class_best = {}
        for c, conf in zip(cls_ids, confs):
            cname = class_names[int(c)]
            species_instance_counts[cname] += 1
            per_class_best[cname] = max(per_class_best.get(cname, 0.0), conf)
        for cname in per_class_best:
            species_counts[cname] += 1
        summary = ", ".join(f"{c} ({conf:.2f})" for c, conf in sorted(
            per_class_best.items(), key=lambda kv: -kv[1]))
        print(f"{name:40s}  {'YES':9s}  {summary}")

    print("-" * 80)
    print(f"\n{n_images} images scanned, {n_with_fouling} had biofouling detected "
          f"({n_with_fouling / n_images:.1%})" if n_images else "no images found")
    if species_counts:
        print("\nBy species (images containing at least one detection / total instances):")
        for cname in class_names.values():
            if cname in species_counts:
                print(f"  {cname}: {species_counts[cname]} images, "
                      f"{species_instance_counts[cname]} instances")

    if args.save:
        actual_save_dir = getattr(getattr(model, "predictor", None), "save_dir", None)
        print(f"\nAnnotated images saved under {actual_save_dir or (ROOT / 'runs' / 'segment' / args.name)}/")


if __name__ == "__main__":
    main()
