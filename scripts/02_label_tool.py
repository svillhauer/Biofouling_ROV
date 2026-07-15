#!/usr/bin/env python3
"""
Interactive SAM-assisted segmentation labeling tool.

For each unlabeled image: left-click on the algae (positive point), right-click
to exclude a region (negative point), then press SPACE to run SAM and preview
the mask (shown in orange). Press 's' to accept the mask as one instance -
accepted instances stay drawn with a green outline for the rest of that image
and are written to disk immediately, so you can label several thalli in the
same photo by repeating click -> SPACE -> s. 'b' if the image has no visible
target of the current --class-id (confirmed negative for that class only),
'r' to reset points, 'u' to undo the last point, 'k' to skip (leave for
later), 'p' to go back to the previous image (e.g. you fat-fingered 'b' or
'n'), 'q' to quit (progress is already saved as you go).

An image can be safely revisited under a *different* --class-id (e.g. the
same rov_frames photo shows two species) - each class's instances are
preserved independently rather than overwriting one another, and any
already-labeled other-class instances are shown as a cyan outline for
spatial context.

Usage:
    python scripts/02_label_tool.py
    python scripts/02_label_tool.py --source rov_frames   # label ROV frames instead
    python scripts/02_label_tool.py --model mobile_sam.pt  # faster, lower quality
    python scripts/02_label_tool.py --source images/aa --class-id 1  # label AA instances
                                                                      # for the false-positive
                                                                      # check, not for training
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM

ROOT = Path(__file__).resolve().parent.parent
MAX_DISPLAY = (1280, 900)

state = {
    "points_disp": [],   # [(x, y)] in display coords
    "labels": [],        # 1 = positive, 0 = negative
    "mask_poly_orig": None,  # in-progress candidate polygon in ORIGINAL pixel coords
    "accepted_polys": [],    # polygons already accepted (s) for the current image
    "img_disp": None,
    "img_orig": None,
    "scale": 1.0,
    "class_id": 0,
}


def mouse_cb(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["points_disp"].append((x, y))
        state["labels"].append(1)
    elif event == cv2.EVENT_RBUTTONDOWN:
        state["points_disp"].append((x, y))
        state["labels"].append(0)


def resize_for_display(img):
    h, w = img.shape[:2]
    scale = min(MAX_DISPLAY[0] / w, MAX_DISPLAY[1] / h, 1.0)
    disp = cv2.resize(img, (int(w * scale), int(h * scale)))
    return disp, scale


def largest_contour_polygon(mask: np.ndarray):
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 30:
        return None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.002 * peri, True)
    if len(approx) < 3:
        approx = c
    return approx.reshape(-1, 2)


def draw_overlay():
    disp = state["img_disp"].copy()

    # Already-accepted instances for this image: persistent green outline,
    # stays visible so accepting one doesn't look like it vanished.
    if state["accepted_polys"]:
        overlay = disp.copy()
        for poly in state["accepted_polys"]:
            poly_disp = (poly * state["scale"]).astype(np.int32)
            cv2.fillPoly(overlay, [poly_disp], (0, 200, 0))
        disp = cv2.addWeighted(overlay, 0.3, disp, 0.7, 0)
        for poly in state["accepted_polys"]:
            poly_disp = (poly * state["scale"]).astype(np.int32)
            cv2.polylines(disp, [poly_disp], True, (0, 200, 0), 2)

    for (x, y), lab in zip(state["points_disp"], state["labels"]):
        color = (0, 255, 0) if lab == 1 else (0, 0, 255)
        cv2.circle(disp, (x, y), 5, color, -1)

    # Instances already saved under a *different* class for this image
    # (only when re-visiting the same image for a second species): cyan
    # outline, no fill, so it's visible for spatial context without
    # obscuring the current class's own overlay.
    if state.get("other_polys"):
        for poly in state["other_polys"]:
            poly_disp = (poly * state["scale"]).astype(np.int32)
            cv2.polylines(disp, [poly_disp], True, (255, 255, 0), 2)

    # In-progress candidate mask (not yet accepted): orange
    if state["mask_poly_orig"] is not None:
        poly_disp = (state["mask_poly_orig"] * state["scale"]).astype(np.int32)
        overlay = disp.copy()
        cv2.fillPoly(overlay, [poly_disp], (255, 200, 0))
        disp = cv2.addWeighted(overlay, 0.35, disp, 0.65, 0)
        cv2.polylines(disp, [poly_disp], True, (255, 200, 0), 2)

    cv2.putText(disp, f"LMB=+point RMB=-point SPACE=predict s=accept b=background  [class {state['class_id']}]",
                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(disp, "r=reset u=undo k=skip p=previous q=quit",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(disp, f"accepted instances (green) in this image: {len(state['accepted_polys'])}"
                       f"  |  other classes already labeled here (cyan): {len(state.get('other_polys') or [])}",
                (10, disp.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return disp


def to_yolo_seg_line(poly_orig, w, h, class_id):
    coords = []
    for x, y in poly_orig:
        coords.append(min(max(x / w, 0.0), 1.0))
        coords.append(min(max(y / h, 0.0), 1.0))
    return str(class_id) + " " + " ".join(f"{c:.6f}" for c in coords)


def parse_yolo_seg_line_poly(line, w, h):
    parts = line.split()
    coords = [float(v) for v in parts[1:]]
    pts = [[coords[i] * w, coords[i + 1] * h] for i in range(0, len(coords), 2)]
    return np.array(pts, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="images/all",
                    help="Subfolder under data/ to label (default: images/all)")
    ap.add_argument("--model", default="sam_b.pt",
                    help="SAM checkpoint: sam_b.pt (better) or mobile_sam.pt (faster)")
    ap.add_argument("--class-id", type=int, default=0,
                    help="Class index to write for every instance labeled this run. "
                         "0=rugulopteryx_okamurae is the only trained class (see data.yaml); "
                         "1=asparagopsis_armata is a look-alike used only for the "
                         "false-positive check (scripts/05_check_false_positives.py), "
                         "never fed into training")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    img_dir = ROOT / "data" / args.source
    label_dir = ROOT / "data" / "labels_yolo_seg"
    label_dir.mkdir(parents=True, exist_ok=True)
    progress_file = label_dir / "_progress.txt"

    done = set()
    if progress_file.exists():
        done = set(progress_file.read_text().splitlines())

    def class_key(name):
        # Compound "name::class_id" keys let the same image be revisited
        # under a *different* class-id (e.g. rov_frames with two species
        # in one photo) without being blocked by the plain-filename entries
        # written by earlier, single-class-per-image runs.
        return f"{name}::{args.class_id}"

    all_images = sorted([p for p in img_dir.iterdir()
                          if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    todo = [p for p in all_images if p.name not in done and class_key(p.name) not in done]
    random.Random(args.seed).shuffle(todo)

    if not todo:
        print(f"Nothing to label in {img_dir} (all {len(all_images)} already processed).")
        return

    print(f"{len(todo)} images left to label out of {len(all_images)} total. Class id: {args.class_id}")
    print(f"Loading SAM model {args.model} ...")
    sam = SAM(args.model)
    state["class_id"] = args.class_id

    cv2.namedWindow("label", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("label", mouse_cb)

    i = 0
    while i < len(todo):
        path = todo[i]
        img_orig = cv2.imread(str(path))
        if img_orig is None:
            print(f"Could not read {path}, skipping.")
            i += 1
            continue
        h, w = img_orig.shape[:2]
        disp, scale = resize_for_display(img_orig)

        label_path = label_dir / (path.stem + ".txt")
        # Lines already saved under a *different* class-id (from an earlier
        # pass over this same source dir) - preserved untouched underneath
        # whatever this session does for args.class_id.
        other_lines = []
        other_polys = []
        if label_path.exists():
            for line in label_path.read_text().splitlines():
                if not line.strip():
                    continue
                if line.split()[0] != str(args.class_id):
                    other_lines.append(line)
                    other_polys.append(parse_yolo_seg_line_poly(line, w, h))

        state.update(img_orig=img_orig, img_disp=disp, scale=scale,
                     points_disp=[], labels=[], mask_poly_orig=None, accepted_polys=[],
                     other_polys=other_polys)

        instances_this_image = 0
        lines = []

        def write_label():
            all_lines = other_lines + lines
            label_path.write_text("\n".join(all_lines) + "\n" if all_lines else "")

        print(f"[{i+1}/{len(todo)}] {path.name}")
        step = None  # +1 = next image, -1 = go back to previous image
        while step is None:
            cv2.imshow("label", draw_overlay())
            key = cv2.waitKey(20) & 0xFF

            if key == ord('q'):
                if lines:
                    write_label()
                progress_file.write_text("\n".join(sorted(done)) + "\n")
                cv2.destroyAllWindows()
                print(f"Saved progress: {len(done)} images done.")
                return

            elif key == ord('u') and state["points_disp"]:
                state["points_disp"].pop()
                state["labels"].pop()

            elif key == ord('r'):
                state["points_disp"], state["labels"] = [], []
                state["mask_poly_orig"] = None

            elif key == 32:  # SPACE -> run SAM
                if not state["points_disp"]:
                    continue
                pts_orig = [[int(x / scale), int(y / scale)] for x, y in state["points_disp"]]
                results = sam.predict(img_orig, points=[pts_orig], labels=[state["labels"]], verbose=False)
                r = results[0]
                if r.masks is None or len(r.masks.data) == 0:
                    print("  SAM returned no mask, try different points.")
                    continue
                mask = r.masks.data[0].cpu().numpy()
                poly = largest_contour_polygon(mask)
                state["mask_poly_orig"] = poly

            elif key == ord('s'):
                if state["mask_poly_orig"] is None:
                    print("  No mask to accept yet - press SPACE first.")
                    continue
                lines.append(to_yolo_seg_line(state["mask_poly_orig"], w, h, args.class_id))
                state["accepted_polys"].append(state["mask_poly_orig"])
                instances_this_image += 1
                write_label()  # autosave immediately
                print(f"  Instance {instances_this_image} saved to {label_path.name}.")
                state["points_disp"], state["labels"] = [], []
                state["mask_poly_orig"] = None

            elif key == ord('b'):
                # confirmed no instances of THIS class here. Other classes'
                # lines already saved for this image (if any, from an
                # earlier pass) are preserved, not wiped.
                lines.clear()
                write_label()
                done.add(class_key(path.name))
                progress_file.write_text("\n".join(sorted(done)) + "\n")
                step = 1

            elif key == ord('k'):
                step = 1  # skip without marking done -> reappears later

            elif key == ord('n'):
                already_saved = label_path.exists() and label_path.stat().st_size > 0
                if not lines and not other_lines and not already_saved:
                    print("  Nothing accepted yet - press 'b' if there's no algae here, "
                          "or click + SPACE + 's' to label it. Use 'k' to leave undecided.")
                    continue
                write_label()
                done.add(class_key(path.name))
                progress_file.write_text("\n".join(sorted(done)) + "\n")
                step = 1

            elif key == ord('p'):
                if i == 0:
                    print("  Already at the first image.")
                    continue
                step = -1

        i += step

    progress_file.write_text("\n".join(sorted(done)) + "\n")
    cv2.destroyAllWindows()
    print("All images processed.")


if __name__ == "__main__":
    main()
