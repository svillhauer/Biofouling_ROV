"""Read/write single instances in the YOLO segmentation label format.

One line per instance: ``<class_id> x1 y1 x2 y2 ...`` with coordinates
normalized to [0, 1] by image width/height.
"""
import numpy as np


def to_yolo_seg_line(poly_px, width, height, class_id) -> str:
    """Serialize a pixel-space polygon as a YOLO-seg label line (coords clamped to [0, 1])."""
    coords = []
    for x, y in poly_px:
        coords.append(min(max(x / width, 0.0), 1.0))
        coords.append(min(max(y / height, 0.0), 1.0))
    return str(class_id) + " " + " ".join(f"{c:.6f}" for c in coords)


def parse_yolo_seg_line(line, width, height) -> np.ndarray:
    """Parse a YOLO-seg label line back into an (N, 2) pixel-space polygon."""
    coords = [float(v) for v in line.split()[1:]]
    pts = [[coords[i] * width, coords[i + 1] * height] for i in range(0, len(coords), 2)]
    return np.array(pts, dtype=np.float32)
