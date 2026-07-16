#!/usr/bin/env python3
"""
Real-time biofouling scan that publishes detections over WebSocket for the
blueos-standoff-controller operator console AI overlay, instead of (or in
addition to) the local cv2 preview window in 07_live_scan.py. Same weights,
same inference loop - this just speaks the console's wire protocol
(protocol_version "1.0", messages: hello/heartbeat/detections) on
ws://<host>:<port>/api/v1/detections/ws so static/js/ai-overlay.js can render
boxes/masks over the live video.

RTSP source: found via the BlueOS web UI -> Video Manager -> stream endpoint.
Defaults to the vehicle's video_udp_stream_0 RTSP output below; override
with --source for a different stream, or --source 0 (or another webcam
index) to test the WebSocket pipeline without the ROV connected.

Usage:
    python scripts/09_live_scan_ws.py
    python scripts/09_live_scan_ws.py --source 0   # webcam smoke test

Then point blueos-standoff-controller at it (already the config.py default):
    AI_ENABLED=true
    AI_WEBSOCKET_URL=ws://localhost:8765/api/v1/detections/ws

Press Ctrl+C to stop.
"""
import argparse
import asyncio
import csv
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

# RTSP over UDP (OpenCV/FFmpeg's default) drops/reorders packets on the tether link
# and hangs with repeated "Waiting for stream 0" warnings; TCP transport is reliable.
# Must be set before cv2/ultralytics opens the capture.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import websockets
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
PROTOCOL_VERSION = "1.0"
DEFAULT_SOURCE = "rtsp://192.168.2.2:8554/video_udp_stream_0"
HEARTBEAT_INTERVAL_S = 1.0


class Broadcaster:
    """Tracks connected operator-console clients and fans messages out to all of them."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.clients: set[websockets.ServerConnection] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    async def handle_client(self, websocket):
        self.clients.add(websocket)
        peer = getattr(websocket, "remote_address", "?")
        print(f"[ws] console connected: {peer}")
        try:
            await websocket.send(json.dumps({
                "protocol_version": PROTOCOL_VERSION,
                "type": "hello",
                "timestamp": time.time(),
                "model": {"name": self.model_name},
            }))
            await websocket.wait_closed()
        finally:
            self.clients.discard(websocket)
            print(f"[ws] console disconnected: {peer}")

    def broadcast_threadsafe(self, message: dict):
        """Called from the inference thread; hops onto the asyncio loop to send."""
        if not self.clients or self.loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast(message), self.loop)

    async def _broadcast(self, message: dict):
        payload = json.dumps(message)
        for ws in list(self.clients):
            try:
                await ws.send(payload)
            except websockets.exceptions.ConnectionClosed:
                self.clients.discard(ws)


def polygon_area_px(points_norm, image_width, image_height):
    """Shoelace formula on normalized [0,1] polygon points, scaled to pixel area."""
    if not points_norm or len(points_norm) < 3:
        return None
    area = 0.0
    n = len(points_norm)
    for i in range(n):
        x1n, y1n = points_norm[i]
        x2n, y2n = points_norm[(i + 1) % n]
        area += x1n * y2n - x2n * y1n
    return abs(area) / 2 * image_width * image_height


def run_inference_loop(model, source, conf, broadcaster: Broadcaster, log_writer, log_file,
                        stop_event, record_path=None, record_fps=25):
    """Blocking loop (runs in a background thread) - mirrors 07_live_scan.py's
    predict(stream=True) loop but broadcasts each frame instead of cv2.imshow."""
    frame_id = 0
    window_start = time.time()
    frames_since_heartbeat = 0
    video_writer = None

    try:
        results = model.predict(source=source, conf=conf, stream=True, verbose=False)
        for r in results:
            if stop_event.is_set():
                break
            frame_id += 1
            now = time.time()
            image_height, image_width = r.orig_shape

            if record_path is not None and video_writer is None:
                record_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                video_writer = cv2.VideoWriter(str(record_path), fourcc, record_fps, (image_width, image_height))
                print(f"Recording annotated video to {record_path}")
            if video_writer is not None:
                video_writer.write(r.plot())

            detections = []
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.tolist()
                cls_list = boxes.cls.tolist()
                conf_list = boxes.conf.tolist()
                polys = r.masks.xyn if r.masks is not None else None
                for i, (x1, y1, x2, y2) in enumerate(xyxy):
                    class_id = int(cls_list[i])
                    det = {
                        "class_id": class_id,
                        "class_name": r.names[class_id],
                        "confidence": round(float(conf_list[i]), 4),
                        "bbox": {"format": "xyxy_pixels", "x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    }
                    mask_area = None
                    if polys is not None and i < len(polys):
                        points = polys[i].tolist()
                        det["polygon"] = {"points": points}
                        mask_area = polygon_area_px(points, image_width, image_height)
                    detections.append(det)
                    log_writer.writerow([
                        datetime.now().isoformat(timespec="seconds"), frame_id, det["class_name"], f"{det['confidence']:.3f}",
                        round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1),
                        image_width, image_height,
                        round(mask_area, 1) if mask_area is not None else "",
                    ])
                log_file.flush()

            broadcaster.broadcast_threadsafe({
                "protocol_version": PROTOCOL_VERSION,
                "type": "detections",
                "timestamp": now,
                "frame_id": frame_id,
                "image_width": int(image_width),
                "image_height": int(image_height),
                "detection_count": len(detections),
                "detections": detections,
            })

            frames_since_heartbeat += 1
            elapsed = now - window_start
            if elapsed >= HEARTBEAT_INTERVAL_S:
                broadcaster.broadcast_threadsafe({
                    "protocol_version": PROTOCOL_VERSION,
                    "type": "heartbeat",
                    "timestamp": now,
                    "fps": frames_since_heartbeat / elapsed,
                })
                frames_since_heartbeat = 0
                window_start = now
    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"Recording finalized: {record_path}")


async def main_async(args):
    model = YOLO(args.weights)
    source = int(args.source) if args.source.isdigit() else args.source
    broadcaster = Broadcaster(model_name=f"biofouling-{Path(args.weights).parent.parent.name}")
    broadcaster.loop = asyncio.get_running_loop()

    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new_log = not log_path.exists()
    log_file = open(log_path, "a", newline="")
    log_writer = csv.writer(log_file)
    if is_new_log:
        log_writer.writerow(["timestamp", "frame_id", "species", "confidence", "x1", "y1", "x2", "y2", "image_width", "image_height", "mask_area_px2"])

    stop_event = threading.Event()
    record_path = Path(args.record) if args.record else None
    inference_thread = threading.Thread(
        target=run_inference_loop,
        args=(model, source, args.conf, broadcaster, log_writer, log_file, stop_event, record_path, args.record_fps),
        daemon=True,
    )

    async with websockets.serve(broadcaster.handle_client, args.ws_host, args.ws_port):
        print(f"AI overlay WebSocket serving on ws://{args.ws_host}:{args.ws_port}{args.ws_path}")
        print(f"Point the operator console at this with AI_WEBSOCKET_URL "
              f"(ws://localhost:{args.ws_port}{args.ws_path} if it's the same machine).")
        inference_thread.start()
        try:
            while inference_thread.is_alive():
                await asyncio.sleep(0.5)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            # Give the inference thread a chance to break out of its loop and finalize
            # the video file (proper container trailer) before the process exits - a
            # daemon thread killed mid-write would leave the recording corrupted.
            stop_event.set()
            inference_thread.join(timeout=5)
            log_file.close()
            print(f"\nStopped. Detections logged to {log_path}")


def main():
    # One timestamp per run, shared by the log and the recording defaults, so a
    # session's CSV and video are paired by construction - never lands back in
    # the same file as a previous run's differently-shaped data.
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DEFAULT_SOURCE,
                     help="RTSP URL from BlueOS's Video Manager, or a webcam index (e.g. 0) for testing")
    ap.add_argument("--weights", default=str(ROOT / "runs/segment/train-4/weights/best.pt"))
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--log", default=str(ROOT / "runs" / f"live_scan_ws_log_{session_id}.csv"),
                     help="CSV detection log. Defaults to a timestamped file per run (matches --record's default).")
    ap.add_argument("--ws-host", default="0.0.0.0")
    ap.add_argument("--ws-port", type=int, default=8765)
    ap.add_argument("--ws-path", default="/api/v1/detections/ws",
                     help="Cosmetic only - the server accepts connections on any path at this host:port.")
    ap.add_argument("--record", nargs="?", const=str(ROOT / "runs" / "recordings" / f"live_scan_{session_id}.mp4"),
                     help="Save an annotated (boxes/masks drawn) MP4 of the session. "
                          "Pass a path, or use bare --record for a timestamped default under runs/recordings/ "
                          "(same timestamp as the default --log file).")
    ap.add_argument("--record-fps", type=float, default=25.0,
                     help="Playback fps for the saved video (doesn't need to match live processing rate).")
    args = ap.parse_args()

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
