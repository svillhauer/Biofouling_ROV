#!/usr/bin/env python3
"""
Presentation-ready re-renders of two train-4 (models/train-4) figures:
  - confusion_matrix_normalized.png, whose long class-name ticks get clipped
    by Ultralytics' own layout
  - results.png, whose 12-panel grid is too dense for a slide

Re-derives the confusion matrix by re-running val() on the same weights/split
(Ultralytics doesn't persist the raw matrix to disk), and re-plots only the
headline metrics from results.csv.

Usage:
    python scripts/train/plot_training_summary.py
    python scripts/train/plot_training_summary.py --run-dir runs/segment/train-5
"""
import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from biofouling.paths import DATA_YAML, MODELS_DIR, RUNS_DIR
from biofouling.plotting import BLUE, GRID, INK, MUTED, ORANGE, apply_style
from biofouling.species import display_name

apply_style()


def plot_confusion_matrix(run_dir, out_path):
    from ultralytics import YOLO

    model = YOLO(run_dir / "weights" / "best.pt")
    metrics = model.val(data=str(DATA_YAML), split="val", plots=True, verbose=False,
                         project=str(RUNS_DIR / "segment"), name="_val_scratch", exist_ok=True)
    names = [model.names[i] for i in range(len(model.names))] + ["background"]
    matrix = metrics.confusion_matrix.matrix.astype(float)
    norm = matrix / np.clip(matrix.sum(axis=0, keepdims=True), 1, None)

    labels = [display_name(n, wrap=True) if n != "background" else "Background" for n in names]

    fig, ax = plt.subplots(figsize=(9, 8), dpi=200)
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = norm[i, j]
            if v == 0:
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else INK, fontsize=11)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("True", fontsize=11)
    ax.set_ylabel("Predicted", fontsize=11)
    ax.set_title("Confusion matrix (normalized)", fontsize=15, fontweight="bold", loc="left", pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_results_summary(run_dir, out_path):
    rows = list(csv.DictReader((run_dir / "results.csv").open()))
    for r in rows:
        for k in r:
            r[k] = float(r[k])
    epochs = [r["epoch"] for r in rows]

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), dpi=200)

    ax = axes[0]
    ax.plot(epochs, [r["metrics/mAP50(M)"] for r in rows], color=BLUE, linewidth=2, label="mAP50")
    ax.plot(epochs, [r["metrics/mAP50-95(M)"] for r in rows], color=ORANGE, linewidth=2, label="mAP50-95")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mask mAP")
    ax.set_ylim(0, 1)
    ax.set_title("Segmentation accuracy over training", fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=10)

    ax = axes[1]
    train_loss = [r["train/box_loss"] + r["train/seg_loss"] + r["train/cls_loss"] for r in rows]
    val_loss = [r["val/box_loss"] + r["val/seg_loss"] + r["val/cls_loss"] for r in rows]
    ax.plot(epochs, train_loss, color=BLUE, linewidth=2, label="Train loss")
    ax.plot(epochs, val_loss, color=ORANGE, linewidth=2, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (box + seg + cls)")
    ax.set_title("Loss convergence", fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=10)

    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.spines["left"].set_color(MUTED)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=MODELS_DIR / "train-4",
                    help="Ultralytics training run directory (contains weights/ and results.csv)")
    args = ap.parse_args()

    plot_confusion_matrix(args.run_dir, args.run_dir / "confusion_matrix_normalized_presentation.png")
    plot_results_summary(args.run_dir, args.run_dir / "results_presentation.png")

    scratch = RUNS_DIR / "segment" / "_val_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)


if __name__ == "__main__":
    main()
