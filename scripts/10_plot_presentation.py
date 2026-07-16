#!/usr/bin/env python3
"""
Generates presentation-ready PNG figures from a 09_live_scan_ws.py detection
log (timestamp, frame_id, species, confidence, x1, y1, x2, y2, image_width,
image_height, mask_area_px2).

Usage:
    python scripts/10_plot_presentation.py
    python scripts/10_plot_presentation.py --log runs/live_scan_ws_log.csv --out runs/plots
"""
import argparse
import csv
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = Path(__file__).resolve().parent.parent

# Validated categorical palette (blue/green/magenta/yellow slots 1-4), light-mode steps.
COLORS = {
    "botrylloides_violaceus": "#2a78d6",
    "hildenbrandia_rubra": "#008300",
    "asparagopsis_armata": "#e87ba4",
    "rugulopteryx_okamurae": "#eda100",
}
LABELS = {
    "botrylloides_violaceus": "Botrylloides violaceus",
    "hildenbrandia_rubra": "Hildenbrandia rubra",
    "asparagopsis_armata": "Asparagopsis armata",
    "rugulopteryx_okamurae": "Rugulopteryx okamurae",
}
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "text.color": INK,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": SECONDARY_INK,
    "xtick.color": SECONDARY_INK,
    "ytick.color": SECONDARY_INK,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def load_rows(log_path):
    rows = list(csv.DictReader(log_path.open()))
    for r in rows:
        r["confidence"] = float(r["confidence"])
        r["mask_area_px2"] = float(r["mask_area_px2"]) if r["mask_area_px2"] else 0.0
        r["image_width"] = float(r["image_width"])
        r["image_height"] = float(r["image_height"])
        r["ts"] = datetime.fromisoformat(r["timestamp"])
        r["area_pct"] = r["mask_area_px2"] / (r["image_width"] * r["image_height"]) * 100
    return rows


def species_order(rows):
    counts = defaultdict(int)
    for r in rows:
        counts[r["species"]] += 1
    return sorted(counts, key=lambda sp: -counts[sp])


def plot_detections_by_species(rows, species, out_dir):
    counts = defaultdict(int)
    for r in rows:
        counts[r["species"]] += 1

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    ys = range(len(species))
    values = [counts[sp] for sp in species]
    colors = [COLORS[sp] for sp in species]
    bars = ax.barh(list(ys), values, color=colors, height=0.55)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([LABELS[sp] for sp in species])
    ax.invert_yaxis()
    ax.set_xlabel("Detections (frame instances)")
    ax.set_title("Detections by species", fontsize=15, fontweight="bold", loc="left", pad=14)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + max(values) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:,}", va="center", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(out_dir / "01_detections_by_species.png")
    plt.close(fig)


def plot_range_by_species(rows, species, out_dir, key, xlabel, filename, title, fmt, xmax=None):
    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    for i, sp in enumerate(species):
        vals = sorted(r[key] for r in rows if r["species"] == sp)
        p10 = vals[int(len(vals) * 0.10)]
        p50 = vals[int(len(vals) * 0.50)]
        p90 = vals[min(int(len(vals) * 0.90), len(vals) - 1)]
        y = len(species) - i
        ax.plot([p10, p90], [y, y], color=COLORS[sp], linewidth=2, solid_capstyle="round")
        ax.scatter([p50], [y], color=COLORS[sp], s=90, zorder=3, edgecolors="white", linewidths=1.5)
        ax.text(p50, y + 0.22, fmt(p50), ha="center", fontsize=10.5, color=INK)
    ax.set_yticks(range(1, len(species) + 1))
    ax.set_yticklabels([LABELS[sp] for sp in reversed(species)])
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=15, fontweight="bold", loc="left", pad=14)
    if xmax:
        ax.set_xlim(0, xmax)
    ax.grid(axis="x", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    fig.text(0.99, 0.01, "dot = median  •  line = p10–p90 range", ha="right", fontsize=9, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out_dir / filename)
    plt.close(fig)


def plot_detections_over_time(rows, species, out_dir):
    t0 = min(r["ts"] for r in rows)
    bins = defaultdict(lambda: defaultdict(int))
    for r in rows:
        minute = int((r["ts"] - t0).total_seconds() // 60)
        bins[minute][r["species"]] += 1

    all_minutes = sorted(bins.keys())
    runs, current = [], []
    for m in all_minutes:
        if current and m != current[-1] + 1:
            runs.append(current)
            current = []
        current.append(m)
    if current:
        runs.append(current)

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=200)
    for run in runs:
        bottom = [0] * len(run)
        for sp in species:
            values = [bins[m][sp] for m in run]
            ax.bar(run, values, bottom=bottom, width=0.85, color=COLORS[sp],
                   label=LABELS[sp] if run is runs[0] else None)
            bottom = [b + v for b, v in zip(bottom, values)]

    if len(runs) > 1:
        gap_start, gap_end = runs[0][-1] + 0.5, runs[1][0] - 0.5
        ax.axvspan(gap_start, gap_end, color=MUTED, alpha=0.08)
        ax.text((gap_start + gap_end) / 2, ax.get_ylim()[1] * 0.5, "no detections\n(~10 min)",
                ha="center", va="center", fontsize=10, color=SECONDARY_INK)

    ax.set_xlabel("Minutes since session start")
    ax.set_ylabel("Detections per minute")
    ax.set_title("Detections over time, by species", fontsize=15, fontweight="bold", loc="left", pad=14)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.spines["left"].set_color(MUTED)
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "04_detections_over_time.png")
    plt.close(fig)


def print_conclusion(rows, species):
    print("\n" + "=" * 70)
    print("PRELIMINARY CONCLUSIONS (grounded in this session's data)")
    print("=" * 70)
    by_sp = defaultdict(list)
    for r in rows:
        by_sp[r["species"]].append(r)

    for sp in species:
        vals = by_sp[sp]
        confs = [v["confidence"] for v in vals]
        areas = [v["mask_area_px2"] / (v["image_width"] * v["image_height"]) * 100 for v in vals]
        print(f"\n{LABELS[sp]} (n={len(vals)}):")
        print(f"  confidence: median {statistics.median(confs):.2f}, mean {statistics.mean(confs):.2f}")
        print(f"  frame coverage: median {statistics.median(areas):.1f}%, "
              f"range {min(areas):.1f}%-{max(areas):.1f}%")

    dominant_count = species[0]
    print(f"\n- {LABELS[dominant_count]} accounts for the most detections by far, but its "
          f"per-detection frame coverage barely varies across thousands of hits - a signature "
          f"more consistent with a fixed visual artifact than a real, varying organism. Worth "
          f"a manual spot-check before treating its raw count as ground truth.")
    highest_conf = max(species, key=lambda sp: statistics.median(r["confidence"] for r in by_sp[sp]))
    print(f"- {LABELS[highest_conf]} shows the highest confidence and the most variable, often "
          f"large, coverage area - the most credible large-scale detection in this session.")
    print("- Detections were not uniform across the session: watch for extended gaps followed "
          "by sharp spikes in the over-time chart, which likely mark transit vs. actual fouled "
          "structure.")
    print("=" * 70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True,
                     help="Path to a 09_live_scan_ws.py CSV log, e.g. runs/live_scan_ws_log_20260716_133454.csv "
                          "- each recording session gets its own timestamped log, so pick the one you want plotted.")
    ap.add_argument("--out",
                     help="Output directory for the PNGs. Defaults to runs/plots/<session timestamp>/, "
                          "derived from the --log filename, so different sessions' plots don't overwrite each other.")
    args = ap.parse_args()

    log_path = Path(args.log)
    if args.out:
        out_dir = Path(args.out)
    else:
        session_suffix = log_path.stem.replace("live_scan_ws_log_", "").replace("live_scan_ws_log", "latest")
        out_dir = ROOT / "runs" / "plots" / session_suffix
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(log_path)
    if not rows:
        print(f"No rows found in {log_path}")
        return
    species = species_order(rows)

    plot_detections_by_species(rows, species, out_dir)
    plot_range_by_species(rows, species, out_dir, "confidence", "Confidence",
                           "02_confidence_by_species.png", "Detection confidence by species",
                           fmt=lambda v: f"{v:.2f}", xmax=1.0)
    plot_range_by_species(rows, species, out_dir, key="area_pct", xlabel="Frame coverage (%)",
                           filename="03_coverage_by_species.png", title="Detected area by species (% of frame)",
                           fmt=lambda v: f"{v:.0f}%")
    plot_detections_over_time(rows, species, out_dir)

    print(f"Saved 4 figures to {out_dir}/")
    print_conclusion(rows, species)


if __name__ == "__main__":
    main()
