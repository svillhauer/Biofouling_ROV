"""Canonical repository paths, so scripts don't depend on their own location."""
import os
from pathlib import Path

# src/biofouling/paths.py -> repo root. Override with BIOFOULING_ROOT when the
# package is installed somewhere other than an editable checkout.
ROOT = Path(os.environ.get("BIOFOULING_ROOT", Path(__file__).resolve().parents[2]))

CONFIG_DIR = ROOT / "configs"
DATA_YAML = CONFIG_DIR / "data.yaml"

DATA_DIR = ROOT / "data"
IMAGES_DIR = DATA_DIR / "images"
ROV_FRAMES_DIR = DATA_DIR / "rov_frames"
LABELS_DIR = DATA_DIR / "labels_yolo_seg"
DATASET_DIR = DATA_DIR / "dataset"

MODELS_DIR = ROOT / "models"
DEFAULT_WEIGHTS = MODELS_DIR / "train-4" / "weights" / "best.pt"

# Scratch output: training runs, scan logs, recordings, plots (gitignored).
RUNS_DIR = ROOT / "runs"
