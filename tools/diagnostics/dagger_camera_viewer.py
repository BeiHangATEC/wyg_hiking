#!/usr/bin/env python3
"""Local browser viewer for live DAGGER depth and optional backend records."""

from __future__ import annotations

import argparse
from collections import deque
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib
import io
import json
import os
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit

import cv2
import numpy as np

from dagger_camera_input import camera_info_dict, load_pipeline, stamp_ns, value_summary


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]


def color_depth(values, minimum, maximum, invalid=None):
    normalized = np.clip((values - minimum) / (maximum - minimum), 0., 1.)
    image = cv2.applyColorMap(np.uint8(np.nan_to_num(normalized) * 255), cv2.COLORMAP_TURBO)
    bad = ~np.isfinite(values)
    if invalid is not None:
        bad |= invalid
    image[bad] = 0
    return image


def png_bytes(image):
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise ValueError("cannot encode depth image")
    return encoded.tobytes()


def committed_record_line(path):
    # The recorder appends a line only after all input files are atomically saved.
    with path.open("rb") as stream:
        stream.seek(0, 2)
        size = stream.tell()
        start = max(0, size - 16384)
        stream.seek(start)
        data = stream.read()
    lines = data.splitlines(keepends=True)
    if start and lines:
        lines.pop(0)
    committed = [line for line in lines if line.endswith(b"\n") and line.strip()]
    return committed[-1].decode().strip() if committed else None


class BackendRecordReader:
    def __init__(self, directory, model_sha256, contract):
        self.directory = Path(directory).resolve()
        self.model_sha256 = model_sha256
        self.contract = contract
        self.frame = None
        self.error = None
        self._key = None

    def poll(self):
        metadata_path = self.directory / "capture.json"
        dataset_path = self.directory / "dataset.txt"
        if not metadata_path.is_file() or not dataset_path.is_file():
            self.frame, self.error, self._key = None, None, None
            return
        try:
            metadata = json.loads(metadata_path.read_text())
            if not isinstance(metadata, dict):
                raise ValueError("recorded metadata must be an object")
            if metadata.get("model_sha256") != self.model_sha256:
                raise ValueError("recorded model hash does not match installed DAGGER")
            declared = {item["name"]: (item["shape"], item["dtype"]) for item in metadata["inputs"]}
            if declared != self.contract:
                raise ValueError("recorded input contract does not match installed DAGGER")
            line = committed_record_line(dataset_path)
            if not line:
                self.frame, self.error, self._key = None, None, None
                return
            key = (line, metadata_path.stat().st_mtime_ns, dataset_path.stat().st_ino)
            if key == self._key:
                return
            paths = [Path(value).resolve() for value in line.split()]
            if len(paths) != len(metadata["inputs"]):
                raise ValueError("incomplete recorded sample")
            inputs = {}
            for spec, path in zip(metadata["inputs"], paths):
                if path.parent != self.directory or path.suffix != ".npy":
                    raise ValueError("recorded path is outside the model capture directory")
                if path.stat().st_size > 65536:
                    raise ValueError("recorded input file exceeds the DAGGER contract size")
                value = np.load(path, allow_pickle=False)
                if (list(value.shape), str(value.dtype)) != self.contract[spec["name"]]:
                    raise ValueError("recorded array does not match its declared contract")
                inputs[spec["name"]] = value
            depth = inputs["depth_data"]
            self.frame = {
                "sample": paths[0].name.split("_")[0],
                "updated_ns": max(path.stat().st_mtime_ns for path in paths),
                "model": metadata["model"], "backend": metadata["capture_backend"],
                "summary": {name: value_summary(value) for name, value in inputs.items()},
                "inputs": inputs, "depth": depth[0].copy(),
            }
            self.error, self._key = None, key
        except (OSError, ValueError, KeyError, TypeError) as exc:
            self.frame, self.error, self._key = None, str(exc), None


class CameraMonitor:
    def __init__(self, state, policy):
        self.state, self.policy = state, policy
        self.lock = threading.Lock()
        self.latest = None
        self.receipts = deque(maxlen=90)
        self.count = self.rejected = self.duplicate_stamps = self.backward_stamps = 0
        self.previous_stamp = None
        self.error = None
        package = state.__class__.__module__.rsplit(".", 1)[0]
        self.projection = importlib.import_module(f"{package}.depth_projection")

    def on_info(self, message):
        self.state.camera_info_callback(message)

    def on_image(self, message):
        received = time.monotonic()
        with self.lock:
            self.count += 1
            self.receipts.append(received)
            stamp = stamp_ns(message)
            if self.previous_stamp is not None:
                self.duplicate_stamps += int(stamp == self.previous_stamp)
                self.backward_stamps += int(stamp < self.previous_stamp)
            self.previous_stamp = stamp
        try:
            previous = self.state._latest_depth_frame_id
            self.state.depth_image_callback(message)
            if previous == self.state._latest_depth_frame_id:
                raise ValueError("waiting for valid CameraInfo/depth")
            raw = self.state._depth_msg_to_meters(message)
            tensor = self.policy._preprocess_depth(self.state._depth_rotated).copy()
            if raw is None or tensor.shape != (36, 36) or not np.isfinite(tensor).all():
                raise ValueError("invalid depth preprocessing result")
            info = self.state._camera_info
            if info.header.frame_id != message.header.frame_id:
                raise ValueError("depth and CameraInfo optical frames differ")
            _, crop = self.projection.project_depth(raw, info, self.state.projection)
            x, y, width, height = crop
            roi = raw[y:y + height, x:x + width]
            frame = {
                "seq": self.count, "received": received, "stamp_ns": stamp,
                "raw": raw, "rotated": self.state._depth_rotated.copy(), "tensor": tensor,
                "crop": crop, "info": camera_info_dict(info), "encoding": message.encoding,
                "invalid_percent": float(np.mean(~(np.isfinite(raw) & (raw > 0))) * 100),
                "crop_invalid_percent": float(np.mean(~(np.isfinite(roi) & (roi > 0))) * 100),
                "tensor_summary": value_summary(tensor),
            }
            with self.lock:
                self.latest, self.error = frame, None
        except (ValueError, TypeError) as exc:
            with self.lock:
                self.error = str(exc)
                self.rejected += 1

    def snapshot(self):
        with self.lock:
            times = list(self.receipts)
            fps = (len(times) - 1) / (times[-1] - times[0]) if len(times) > 1 and times[-1] > times[0] else None
            return self.latest, {"received": self.count, "rejected": self.rejected,
                                 "duplicate_stamps": self.duplicate_stamps,
                                 "backward_stamps": self.backward_stamps,
                                 "fps": fps, "error": self.error}


class ViewerCache:
    def __init__(self, monitor, reader, topics, domain):
        self.monitor, self.reader = monitor, reader
        self.topics, self.domain = list(topics), domain
        self.lock = threading.Lock()
        self.api = {"camera": {"state": "waiting"}, "actual": {"state": "waiting"}}
        self.images = {}
        self.arrays = {}

    def refresh(self):
        frame, stats = self.monitor.snapshot()
        self.reader.poll()
        now = time.monotonic()
        timeout = self.monitor.state.depth_timeout_sec
        camera = {**stats, "state": "waiting", "age_ms": None, "seq": None}
        images, arrays = {}, {}
        if frame is not None:
            age = max(0., now - frame["received"])
            camera.update({key: frame[key] for key in ("seq", "stamp_ns", "info", "encoding", "crop",
                                                      "invalid_percent", "crop_invalid_percent", "tensor_summary")})
            camera.update(state="invalid" if stats["error"] else ("stale" if age > timeout else "live"),
                          age_ms=age * 1000)
            minimum, maximum = self.monitor.policy.depth_range
            raw_image = color_depth(frame["raw"], minimum, maximum, frame["raw"] <= 0)
            x, y, width, height = frame["crop"]
            cv2.rectangle(raw_image, (x, y), (x + width - 1, y + height - 1), (0, 255, 255), 2)
            images["native"] = png_bytes(raw_image)
            images["preview"] = png_bytes(color_depth(frame["tensor"], -.5, .5))
            arrays = {"raw_depth_m": frame["raw"], "state_depth_rotated": frame["rotated"],
                      "tensor_preview": frame["tensor"][np.newaxis], "camera_k": np.asarray(frame["info"]["k"])}
        elif stats["error"]:
            camera["state"] = "invalid"
        actual = {"state": "invalid" if self.reader.error else "waiting",
                  "error": self.reader.error, "directory": str(self.reader.directory),
                  "sample": None, "age_ms": None}
        recorded = self.reader.frame
        if recorded is not None:
            age_ms = max(0., time.time_ns() - recorded["updated_ns"]) / 1e6
            finite = all(value["nonfinite_fraction"] == 0. for value in recorded["summary"].values())
            actual.update({key: recorded[key] for key in ("sample", "updated_ns", "model", "backend", "summary")})
            actual.update(age_ms=age_ms, state="invalid" if not finite else ("stale" if age_ms > timeout * 1000 else "live"))
            images["actual"] = png_bytes(color_depth(recorded["depth"], -.5, .5))
            arrays.update({f"backend_{name}": value for name, value in recorded["inputs"].items()})
        payload = {"camera": camera, "actual": actual, "topics": self.topics, "domain": self.domain,
                   "has_snapshot": bool(arrays), "display_range_m": list(self.monitor.policy.depth_range)}
        with self.lock:
            self.api, self.images, self.arrays = payload, images, arrays

    def response(self, path):
        with self.lock:
            if path == "/api/frame":
                return "application/json", json.dumps(self.api, allow_nan=False).encode()
            if path.startswith("/api/image/"):
                name = path.rsplit("/", 1)[-1].removesuffix(".png")
                if name not in self.images:
                    return None
                return "image/png", self.images[name]
            if path == "/api/snapshot.npz" and self.arrays:
                arrays = self.arrays.copy()
                metadata = json.dumps(self.api, allow_nan=False)
            else:
                return None
        output = io.BytesIO()
        np.savez_compressed(output, **arrays, metadata_json=np.asarray(metadata))
        return "application/octet-stream", output.getvalue()


def make_handler(cache):
    html = (HERE / "dagger_camera_viewer.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = urlsplit(self.path).path
            result = ("text/html; charset=utf-8", html) if path == "/" else cache.response(path)
            if result is None:
                self.send_error(404)
                return
            content_type, body = result
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if path == "/api/snapshot.npz":
                self.send_header("Content-Disposition", 'attachment; filename="dagger-input-snapshot.npz"')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format, *args):
            return

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--capture-root", type=Path, default=PROJECT / "logs/dagger-live-inputs")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")

    import rclpy
    from ament_index_python.packages import get_package_share_path
    from rclpy.qos import QoSProfile, qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    mod_root = get_package_share_path("bxi_example_py_elf3") / "mods/com.bxi.normal_depth"
    state, policy = load_pipeline(mod_root)
    package = state.__class__.__module__.rsplit(".", 1)[0]
    topics = (state.depth_image_topic, state.camera_info_topic)
    if not topics[0]:
        topics = importlib.import_module(f"{package}.camera_topics").resolve_camera_topics("hardware", state.camera_name)
    state.depth_image_topic, state.camera_info_topic = topics
    sha256 = hashlib.sha256((mod_root / "assets/dagger_new_position.onnx").read_bytes()).hexdigest()
    contract = {name: (list(value.shape), str(value.dtype)) for name, value in policy._policy_inputs.items()}
    reader = BackendRecordReader(args.capture_root / "dagger_new_position", sha256, contract)
    monitor = CameraMonitor(state, policy)
    cache = ViewerCache(monitor, reader, topics, os.environ.get("ROS_DOMAIN_ID", "0"))
    server = ThreadingHTTPServer((args.bind, args.port), make_handler(cache))
    server.daemon_threads = True
    stop = threading.Event()
    rclpy.init(args=[])
    node = rclpy.create_node("dagger_camera_viewer")
    state._bind_logger(node.get_logger())
    qos = QoSProfile(depth=1, reliability=qos_profile_sensor_data.reliability,
                     durability=qos_profile_sensor_data.durability)
    subscriptions = [node.create_subscription(CameraInfo, topics[1], monitor.on_info, qos),
                     node.create_subscription(Image, topics[0], monitor.on_image, qos)]

    def receive():
        while not stop.is_set() and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.1)

    def render():
        while not stop.is_set():
            try:
                cache.refresh()
            except Exception as exc:
                node.get_logger().error(f"viewer rendering failed: {exc}")
            stop.wait(.2)

    threads = [threading.Thread(target=receive, daemon=True), threading.Thread(target=render, daemon=True)]
    for thread in threads:
        thread.start()
    print(f"Read-only viewer: http://{args.bind}:{args.port}/", flush=True)
    print(f"Camera topics: {topics}; capture root: {args.capture_root}", flush=True)
    print("No camera driver, controller or motor command is started by this viewer.", flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=2)
        server.server_close()
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()
        policy.close()


if __name__ == "__main__":
    main()
