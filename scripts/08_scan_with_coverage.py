#!/usr/bin/env python3
"""
Same live-scan loop as 07_live_scan.py, plus a per-frame percent-coverage
metric using range from a Blue Robotics Ping1D echosounder (read directly
over its own connection via brping, not through ArduSub/MAVLink).

Percent coverage of the frame (detected-mask pixels / total frame pixels)
doesn't actually need range - it's already a scale-independent fraction of
whatever's currently in view. What the echosounder range adds:
  - converts that fraction into an absolute covered area in m^2, using the
    distance-to-wall and the camera's field of view (--hfov/--vfov)
  - logs the range alongside each reading, so frames taken from farther
    away (blurrier, less reliable detections) can be filtered out later

This is a single-frame metric, not a whole-wall survey number - overlapping
frames along a pass aren't de-duplicated here, so summing covered_area_m2
across frames will double-count anything seen more than once. Stitching
frames along a survey track into one non-overlapping estimate is a separate,
harder problem for later.

Usage:
    python scripts/08_scan_with_coverage.py --source rtsp://192.168.2.2:8554/video \\
        --ping-udp 192.168.2.2:9092 --hfov 80 --vfov 64
    python scripts/08_scan_with_coverage.py --source 0 --ping-serial /dev/ttyUSB0
Press 'q' in the preview window (or Ctrl+C in the terminal) to stop.
"""
import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO
from brping import Ping1D

ROOT = Path(__file__).resolve().parent.parent


def connect_ping(args) -> Ping1D:
    ping = Ping1D()
    if args.ping_udp:
        host, port = args.ping_udp.split(":")
        ping.connect_udp(host, int(port))
    else:
        ping.connect_serial(args.ping_serial, args.ping_baud)
    if not ping.initialize():
        raise RuntimeError(
            "Failed to initialize the Ping1D echosounder - check the "
            "--ping-udp/--ping-serial address and that nothing else is holding the port")
    return ping


def read_range_m(ping: Ping1D):
    data = ping.get_distance_simple()
    if data is None:
        return None, None
    return data["distance"] / 1000.0, data["confidence"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True,
                     help="RTSP URL from BlueOS's Video Manager, or a webcam index (e.g. 0) for testing")
    ap.add_argument("--weights", default=str(ROOT / "runs/segment/train-4/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--log", default=str(ROOT / "runs" / "live_scan_log.csv"),
                     help="Per-detection CSV, same format as 07_live_scan.py plus range_m")
    ap.add_argument("--coverage-log", default=str(ROOT / "runs" / "coverage_log.csv"),
                     help="Per-frame percent-coverage CSV")
    ap.add_argument("--no-display", action="store_true",
                     help="Skip the live preview window - log detections only (for a headless run)")

    ping_group = ap.add_mutually_exclusive_group(required=True)
    ping_group.add_argument("--ping-udp", metavar="HOST:PORT",
                             help="Echosounder reachable over UDP, e.g. 192.168.2.2:9092")
    ping_group.add_argument("--ping-serial", metavar="DEVICE",
                             help="Echosounder on a direct serial connection, e.g. /dev/ttyUSB0")
    ap.add_argument("--ping-baud", type=int, default=115200,
                     help="Baud rate for --ping-serial")
    ap.add_argument("--hfov", type=float, default=None,
                     help="Camera horizontal field of view in degrees. Needed to convert "
                          "percent coverage into an area in m^2 - without it, only the "
                          "percent-of-frame figure is logged.")
    ap.add_argument("--vfov", type=float, default=None,
                     help="Camera vertical field of view in degrees (see --hfov)")
    args = ap.parse_args()

    model = YOLO(args.weights)
    source = int(args.source) if args.source.isdigit() else args.source
    ping = connect_ping(args)

    det_log_path = Path(args.log)
    det_log_path.parent.mkdir(parents=True, exist_ok=True)
    det_is_new = not det_log_path.exists()
    det_file = open(det_log_path, "a", newline="")
    det_writer = csv.writer(det_file)
    if det_is_new:
        det_writer.writerow(["timestamp", "species", "confidence", "range_m"])

    cov_log_path = Path(args.coverage_log)
    cov_log_path.parent.mkdir(parents=True, exist_ok=True)
    cov_is_new = not cov_log_path.exists()
    cov_file = open(cov_log_path, "a", newline="")
    cov_writer = csv.writer(cov_file)
    if cov_is_new:
        cov_writer.writerow(["timestamp", "num_detections", "percent_coverage",
                              "range_m", "range_confidence", "covered_area_m2"])

    if not args.no_display:
        cv2.namedWindow("Biofouling live scan", cv2.WINDOW_NORMAL)

    print(f"Streaming from {args.source} - press 'q' in the preview window "
          f"(or Ctrl+C here) to stop. Logging detections to {det_log_path}, "
          f"coverage to {cov_log_path}")

    try:
        results = model.predict(source=source, conf=args.conf, stream=True, verbose=False)
        for r in results:
            ts = datetime.now().isoformat(timespec="seconds")
            range_m, range_conf = read_range_m(ping)

            num_dets = 0 if r.boxes is None else len(r.boxes)
            if num_dets > 0:
                for cls, conf in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                    det_writer.writerow([ts, r.names[int(cls)], f"{conf:.3f}",
                                          f"{range_m:.3f}" if range_m is not None else ""])
                det_file.flush()

            if r.masks is not None and len(r.masks.data) > 0:
                covered = r.masks.data.amax(dim=0) > 0.5  # union of all instance masks
                percent_coverage = covered.float().mean().item() * 100
            else:
                percent_coverage = 0.0

            covered_area_m2 = ""
            if range_m is not None and args.hfov and args.vfov:
                frame_w_m = 2 * range_m * math.tan(math.radians(args.hfov / 2))
                frame_h_m = 2 * range_m * math.tan(math.radians(args.vfov / 2))
                covered_area_m2 = f"{(percent_coverage / 100) * frame_w_m * frame_h_m:.4f}"

            cov_writer.writerow([ts, num_dets, f"{percent_coverage:.2f}",
                                  f"{range_m:.3f}" if range_m is not None else "",
                                  range_conf if range_conf is not None else "",
                                  covered_area_m2])
            cov_file.flush()

            if not args.no_display:
                cv2.imshow("Biofouling live scan", r.plot())
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        det_file.close()
        cov_file.close()
        cv2.destroyAllWindows()
        print(f"\nStopped. Detections logged to {det_log_path}, coverage to {cov_log_path}")


if __name__ == "__main__":
    main()
