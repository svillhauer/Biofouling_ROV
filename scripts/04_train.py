#!/usr/bin/env python3
"""
Trains a multi-class YOLOv11-seg biofouling detector on data.yaml (one class
per organism - see data.yaml). Sized for an 8GB GPU by default.

Initial training on GBIF images:
    python scripts/04_train.py

Fine-tuning a trained model on newly-added ROV frames (small dataset, low LR
so it adapts to the underwater domain without forgetting what it learned from
the larger GBIF set):
    python scripts/04_train.py --weights runs/segment/train/weights/best.pt \\
        --epochs 50 --lr0 0.001 --name finetune_rov

Image-quality degradation augmentation (--quality-degrade to enable, OFF by
default): tried as a synthetic proxy for the domain gap between clean GBIF
photos and blurry/low-contrast/noisy ROV footage (random blur, reduced
brightness/contrast, sensor noise, JPEG-compression artifacts via
Albumentations). Tested once (see runs/segment/train_quality_degrade) -
detections on real ROV photos went from 1/13 to 10/13, but visual inspection
showed the new "detections" were masking smooth open-water gaps between
rocks, not organism texture: the model appears to have learned to associate
blur/smoothness itself with the target class, rather than gaining real
robustness. Left here disabled by default rather than removed, in case it's
worth revisiting with gentler settings - but real labeled ROV frames are the
more reliable fix once available, not further tuning of this.
"""
import argparse
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def enable_quality_degradation():
    """Monkey-patches ultralytics' Albumentations wrapper to use a stronger,
    underwater-quality-simulating transform list instead of its tame defaults
    (which are mostly p=0.0/0.01 - built for general-purpose light augmentation,
    not for deliberately simulating a harsh domain shift). Scoped to this
    process only - doesn't touch the installed package on disk, so it has no
    effect on other projects sharing this same virtualenv.
    """
    import albumentations as A
    import ultralytics.data.augment as ul_augment

    original_init = ul_augment.Albumentations.__init__

    def patched_init(self, p=1.0, transforms=None):
        if transforms is None:
            transforms = [
                A.OneOf([
                    A.Blur(blur_limit=(3, 9), p=1.0),
                    A.MotionBlur(blur_limit=(3, 9), p=1.0),
                    A.MedianBlur(blur_limit=5, p=1.0),
                ], p=0.35),
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.35, -0.05), contrast_limit=(-0.35, -0.05), p=0.4),
                A.GaussNoise(p=0.3),
                A.ImageCompression(quality_range=(40, 85), p=0.3),
                A.RandomGamma(gamma_limit=(70, 100), p=0.2),
                A.ToGray(p=0.02),
                A.CLAHE(p=0.05),
            ]
        original_init(self, p=p, transforms=transforms)

    ul_augment.Albumentations.__init__ = patched_init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo11s-seg.pt",
                    help="Starting weights: yolo11n/s/m-seg.pt for fresh training, "
                         "or a previous best.pt to fine-tune")
    ap.add_argument("--data", default=str(ROOT / "data.yaml"))
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="train")
    # Bumped color-space augmentation: harbor/ROV footage has a strong
    # blue-green color cast that plain GBIF photos don't, so we train the
    # model to be more color-invariant than the YOLO defaults assume.
    ap.add_argument("--hsv_h", type=float, default=0.02)
    ap.add_argument("--hsv_s", type=float, default=0.8)
    ap.add_argument("--hsv_v", type=float, default=0.5)
    ap.add_argument("--quality-degrade", action="store_true",
                     help="Enable the blur/noise/contrast/compression augmentation "
                          "(OFF by default - see docstring: caused smooth-water false "
                          "positives on real ROV photos in testing, not recommended "
                          "as-is)")
    args = ap.parse_args()

    if args.quality_degrade:
        enable_quality_degradation()

    model = YOLO(args.weights)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        patience=args.patience,
        name=args.name,
        project=str(ROOT / "runs" / "segment"),
        hsv_h=args.hsv_h,
        hsv_s=args.hsv_s,
        hsv_v=args.hsv_v,
        degrees=15,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        mosaic=1.0,
    )


if __name__ == "__main__":
    main()
