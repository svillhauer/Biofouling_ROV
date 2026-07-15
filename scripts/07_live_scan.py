#!/usr/bin/env python3
"""
Real-time biofouling scan against a live video stream (e.g. BlueOS's RTSP
feed), run topside on a GPU machine rather than on the ROV's own companion
computer - no model export/downsizing needed, same weights as
05_scan_for_biofouling.py.

Shows a live annotated preview window (boxes/masks + confidence, like the
photo/video scans) and appends every detection to a CSV log (timestamp,
species, confidence) so there's a record to review after a dive without
having to re-watch the whole feed.

Find your BlueOS stream URL in the BlueOS web UI -> Video Manager - set up
an RTSP (or other cv2-readable) output for the camera you want, then pass
that URL as --source. Use --source 0 (or another webcam index) to test the
script against a local webcam first if you don't have the ROV connected.

Usage:
    python scripts/07_live_scan.py --source rtsp://192.168.2.2:8554/video
    python scripts/07_live_scan.py --source 0 --weights runs/segment/train-4/weights/best.pt
Press 'q' in the preview window (or Ctrl+C in the terminal) to stop.
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                     help="RTSP URL from BlueOS's Video Manager, or a webcam index (e.g. 0) for testing")
    ap.add_argument("--weights", default=str(ROOT / "runs/segment/train-4/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--log", default=str(ROOT / "runs" / "live_scan_log.csv"))
    ap.add_argument("--no-display", action="store_true",
                     help="Skip the live preview window - log detections only (for a headless run)")
    args = ap.parse_args()

    model = YOLO(args.weights)
    source = int(args.source) if args.source.isdigit() else args.source

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new_log = not log_path.exists()
    log_file = open(log_path, "a", newline="")
    writer = csv.writer(log_file)
    if is_new_log:
        writer.writerow(["timestamp", "species", "confidence"])

    if not args.no_display:
        cv2.namedWindow("Biofouling live scan", cv2.WINDOW_NORMAL)

    print(f"Streaming from {args.source} - press 'q' in the preview window "
          f"(or Ctrl+C here) to stop. Logging detections to {log_path}")

    try:
        results = model.predict(source=source, conf=args.conf, stream=True, verbose=False)
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                ts = datetime.now().isoformat(timespec="seconds")
                for cls, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    writer.writerow([ts, r.names[int(cls)], f"{conf:.3f}"])
                log_file.flush()

            if not args.no_display:
                cv2.imshow("Biofouling live scan", r.plot())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        log_file.close()
        cv2.destroyAllWindows()
        print(f"\nStopped. Detections logged to {log_path}")


if __name__ == "__main__":
    main()
