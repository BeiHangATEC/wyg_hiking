#!/usr/bin/env python3
"""Manual local depth/RGB recording until Ctrl+C, isolated from robot control."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import queue
import shutil
import signal
import threading
import time
from zoneinfo import ZoneInfo

import cv2
import numpy as np

from dagger_camera_input import load_pipeline, stamp_ns
from dagger_camera_viewer import BackendRecordReader, CameraMonitor, color_depth


PROJECT = Path(__file__).resolve().parents[2]


def local_time():
    return datetime.now(ZoneInfo("Asia/Shanghai"))


def write_json(path, value):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    os.replace(temporary, path)


class StateInfo:
    """Optional state annotations never control the recording lifetime."""

    def __init__(self, timeout=2.):
        self.timeout = timeout
        self.lock = threading.Lock()
        self.status = None
        self.received = None

    def update(self, payload, now=None):
        now = time.monotonic() if now is None else now
        try:
            value = json.loads(payload)
            if (not isinstance(value, dict) or not isinstance(value.get("current"), dict)
                    or not isinstance(value["current"].get("name"), str)
                    or not isinstance(value.get("in_transition"), bool)):
                raise ValueError("invalid state snapshot")
        except (ValueError, TypeError):
            value = None
        with self.lock:
            self.status, self.received = value, now

    def snapshot(self, now=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            if self.received is None or now - self.received > self.timeout:
                return None
            return self.status


def color_image(message, maximum_width=1280):
    channels = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}
    count = channels.get(message.encoding)
    if not count or not message.width or not message.height:
        raise ValueError(f"unsupported color encoding: {message.encoding}")
    if message.step < message.width * count or len(message.data) < message.step * message.height:
        raise ValueError("incomplete color image")
    array = np.ndarray((message.height, message.width, count), dtype=np.uint8,
                       buffer=bytes(message.data), strides=(message.step, count, 1)).copy()
    conversions = {"rgb8": cv2.COLOR_RGB2BGR, "rgba8": cv2.COLOR_RGBA2BGR,
                   "bgra8": cv2.COLOR_BGRA2BGR, "mono8": cv2.COLOR_GRAY2BGR}
    if message.encoding in conversions:
        array = cv2.cvtColor(array, conversions[message.encoding])
    if array.shape[1] > maximum_width:
        height = max(2, round(array.shape[0] * maximum_width / array.shape[1]) // 2 * 2)
        array = cv2.resize(array, (maximum_width, height), interpolation=cv2.INTER_AREA)
    return array


def video_writer(path, fps, size):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    if not writer.isOpened():
        writer.release()
        raise OSError(f"cannot open video writer: {path}")
    return writer


def depth_panel(frame, actual, elapsed, depth_range=(.2, 3.), actual_error=None):
    panel = np.full((480, 1280, 3), 238, dtype=np.uint8)
    native = color_depth(frame["raw"], *depth_range, frame["raw"] <= 0)
    x, y, width, height = frame["crop"]
    cv2.rectangle(native, (x, y), (x + width - 1, y + height - 1), (0, 255, 255), 2)
    target_height = min(360, max(1, round(native.shape[0] * 640 / native.shape[1])))
    native = cv2.resize(native, (640, target_height), interpolation=cv2.INTER_NEAREST)
    panel[64:64 + target_height, 12:652] = native
    preview = cv2.resize(color_depth(frame["tensor"], -.5, .5), (288, 288), interpolation=cv2.INTER_NEAREST)
    panel[64:352, 672:960] = preview
    if actual is not None:
        image = cv2.resize(color_depth(actual["depth"], -.5, .5), (288, 288), interpolation=cv2.INTER_NEAREST)
        panel[64:352, 980:1268] = image
    actual_state = "INVALID" if actual_error else "WAITING"
    if actual is not None:
        age = max(0., time.time_ns() - actual["updated_ns"]) / 1e9
        finite = all(item["nonfinite_fraction"] == 0 for item in actual["summary"].values())
        actual_state = "INVALID" if not finite else ("STALE" if age > .5 else "LIVE")
    labels = [(12, f"NATIVE DEPTH [{depth_range[0]}, {depth_range[1]} m]"), (672, "PREVIEW (not backend)"),
              (980, "BACKEND: " + actual_state)]
    for left, label in labels:
        cv2.putText(panel, label, (left, 30), cv2.FONT_HERSHEY_SIMPLEX, .62, (35, 40, 40), 1, cv2.LINE_AA)
    cv2.putText(panel, f"elapsed={elapsed:.2f}s  depth_stamp_ns={frame['stamp_ns']}",
                (12, 452), cv2.FONT_HERSHEY_SIMPLEX, .6, (35, 40, 40), 1, cv2.LINE_AA)
    if actual is not None:
        cv2.putText(panel, f"backend sample={actual['sample']} (file time; asynchronous)",
                    (672, 400), cv2.FONT_HERSHEY_SIMPLEX, .5, (35, 40, 40), 1, cv2.LINE_AA)
    return panel


@dataclass
class FramePacket:
    received: float
    depth: object
    info: object
    color: object
    color_info: object
    status: dict | None


class RecordingSession:
    def __init__(self, args, started, identity, topics):
        import rosbag2_py
        from rclpy.serialization import serialize_message

        self.args, self.started = args, started
        self.serialize = serialize_message
        self.rosbag = rosbag2_py
        self.topics = topics
        self.directory = args.output / (local_time().strftime("%Y%m%d-%H%M%S-%f") + "-manual")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.metadata = {"schema": 2, "recording_mode": "manual", "start_time": local_time().isoformat(),
                         "status": "recording", "fps_limit": args.fps, "segment_sec": args.segment_sec,
                         "max_session_sec": args.max_session_sec, "topics": topics, "model": identity,
                         "backend_capture_root": str(args.capture_root), "segments": [],
                         "notes": "AVI has nominal FPS; frames.jsonl is authoritative. Preview is not backend input. "
                                  "Backend file times, color and depth are asynchronous; no frame-alignment guarantee."}
        self.bag = self.depth_video = self.rgb_video = self.frames = None
        self.segment_started = None
        self.count = self.invalid = self.color_errors = 0
        self.color_size = None
        write_json(self.directory / "metadata.json", self.metadata)

    def _close_segment(self):
        for attribute in ("depth_video", "rgb_video"):
            resource = getattr(self, attribute)
            if resource is not None:
                resource.release()
                setattr(self, attribute, None)
        if self.bag is not None:
            self.bag.close()
            self.bag = None
        if self.frames is not None:
            self.frames.close()
            self.frames = None

    def _open_segment(self, now):
        self._close_segment()
        directory = self.directory / f"segment-{len(self.metadata['segments']) + 1:04d}"
        directory.mkdir()
        self.metadata["segments"].append(directory.name)
        self.bag = self.rosbag.SequentialWriter()
        self.bag.open(self.rosbag.StorageOptions(uri=str(directory / "depth_bag"), storage_id="sqlite3"),
                      self.rosbag.ConverterOptions("cdr", "cdr"))
        for name, topic in self.topics.items():
            msg_type = "std_msgs/msg/String" if name == "state" else "sensor_msgs/msg/CameraInfo" if name.endswith("info") else "sensor_msgs/msg/Image"
            # RGB video is saved separately; avoiding raw 1080p RGB in the bag bounds disk I/O.
            if name != "color":
                self.bag.create_topic(self.rosbag.TopicMetadata(name=topic, type=msg_type, serialization_format="cdr"))
        self.depth_video = video_writer(directory / "depth_input.avi", self.args.fps, (1280, 480))
        self.frames = (directory / "frames.jsonl").open("w", buffering=1)
        self.segment_directory, self.segment_started = directory, now
        self.color_size = None
        write_json(self.directory / "metadata.json", self.metadata)

    def write(self, packet, monitor, reader):
        from std_msgs.msg import String

        if self.segment_started is None or packet.received - self.segment_started >= self.args.segment_sec:
            self._open_segment(packet.received)
        timestamp = time.time_ns()
        for key, message in (("depth", packet.depth), ("info", packet.info), ("color_info", packet.color_info)):
            if message is not None:
                self.bag.write(self.topics[key], self.serialize(message), timestamp)
        if packet.status is not None:
            self.bag.write(self.topics["state"], self.serialize(String(data=json.dumps(packet.status))), timestamp)
        previous = monitor.count
        if packet.info is not None:
            monitor.on_info(packet.info)
        monitor.on_image(packet.depth)
        frame, stats = monitor.snapshot()
        valid = frame is not None and frame["seq"] > previous and stats["error"] is None
        reader.poll()
        actual = reader.frame
        record = {"record_time_ns": timestamp, "depth_stamp_ns": stamp_ns(packet.depth),
                  "elapsed_sec": packet.received - self.started, "preview_valid": valid,
                  "error": stats["error"], "state": packet.status,
                  "backend_sample": actual["sample"] if actual else None,
                  "backend_file_time_ns": actual["updated_ns"] if actual else None,
                  "backend_error": reader.error, "color_stamp_ns": None}
        if valid:
            self.depth_video.write(depth_panel(frame, actual, record["elapsed_sec"],
                                               self.metadata["model"].get("depth_range", (.2, 3.)), reader.error))
            if "camera_info" not in self.metadata and "info" in frame:
                self.metadata["camera_info"] = frame["info"]
                write_json(self.directory / "metadata.json", self.metadata)
            record.update(crop=list(frame["crop"]), tensor_summary=frame["tensor_summary"],
                          invalid_percent=frame["invalid_percent"])
        else:
            self.invalid += 1
        if packet.color is not None:
            try:
                rgb = color_image(packet.color)
                if self.rgb_video is None:
                    self.color_size = (rgb.shape[1], rgb.shape[0])
                    self.rgb_video = video_writer(self.segment_directory / "color.avi", self.args.fps, self.color_size)
                if (rgb.shape[1], rgb.shape[0]) != self.color_size:
                    rgb = cv2.resize(rgb, self.color_size, interpolation=cv2.INTER_AREA)
                self.rgb_video.write(rgb)
                record["color_stamp_ns"] = stamp_ns(packet.color)
            except (ValueError, cv2.error) as exc:
                self.color_errors += 1
                record["color_error"] = str(exc)
        self.frames.write(json.dumps(record, allow_nan=False) + "\n")
        self.count += 1

    def close(self, reason, dropped=0):
        self._close_segment()
        self.metadata.update(status="closed", end_time=local_time().isoformat(), reason=reason,
                             frames=self.count, invalid_previews=self.invalid, color_errors=self.color_errors,
                             queue_dropped=dropped)
        write_json(self.directory / "metadata.json", self.metadata)


class RecorderWorker:
    def __init__(self, args, state_info, monitor, reader, identity, topics, log=print):
        self.args, self.state_info, self.monitor, self.reader = args, state_info, monitor, reader
        self.identity, self.topics, self.log = identity, topics, log
        self.queue = queue.Queue(maxsize=8)
        self.session = None
        self.blocked = False
        self.dropped = 0
        self.entry_dropped = 0
        self.next_selected = None
        self.disk_checked = float("-inf")
        self.stop = threading.Event()

    def submit(self, depth, info, color=None, color_info=None, now=None):
        now = time.monotonic() if now is None else now
        if self.blocked or self.stop.is_set():
            return
        if self.next_selected is None:
            self.next_selected = now
        if now + 1e-9 < self.next_selected:
            return
        # Keep sampling phase across camera jitter rather than accumulating skipped intervals.
        missed = int(max(0., now - self.next_selected) * self.args.fps)
        self.next_selected += (missed + 1) / self.args.fps
        try:
            self.queue.put_nowait(FramePacket(now, depth, info, color, color_info, self.state_info.snapshot(now)))
        except queue.Full:
            self.dropped += 1

    def _finish(self, reason):
        if self.session is not None:
            directory = self.session.directory
            try:
                self.session.close(reason, self.dropped - self.entry_dropped)
            except Exception as exc:
                self.log(f"recording finalization failed: {exc}")
            self.session = None
            self.log(f"RECORDING STOP: {reason}; {directory}")

    def tick(self, now=None):
        now = time.monotonic() if now is None else now
        if self.blocked or self.stop.is_set():
            return
        try:
            if (self.session is not None and self.args.max_session_sec > 0
                    and now - self.session.started >= self.args.max_session_sec):
                self.blocked = True
                self._finish("session_time_limit")
                return
            if self.session is not None and now - self.disk_checked >= 1:
                self.disk_checked = now
                if shutil.disk_usage(self.args.output).free < self.args.min_free_gb * 1024 ** 3:
                    raise OSError("disk free space is below the configured recording reserve")
            try:
                packet = self.queue.get_nowait()
            except queue.Empty:
                return
            if self.session is None:
                if shutil.disk_usage(self.args.output).free < self.args.min_free_gb * 1024 ** 3:
                    raise OSError("disk free space is below the configured recording reserve")
                self.session = RecordingSession(self.args, packet.received, self.identity, self.topics)
                self.disk_checked = now
                self.log(f"RECORDING START: {self.session.directory}")
            self.session.write(packet, self.monitor, self.reader)
        except Exception as exc:
            self.blocked = True
            self.log(f"RECORDING ERROR (controller is unaffected): {exc}")
            self._finish("recording_error: " + str(exc))

    def run(self):
        try:
            while not self.stop.is_set():
                self.tick()
                self.stop.wait(.01)
        finally:
            self._finish("recorder_shutdown")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "logs/dagger-recordings")
    parser.add_argument("--capture-root", type=Path, default=PROJECT / "logs/dagger-live-inputs")
    parser.add_argument("--state-topic", default="/hardware/state_machine_info")
    parser.add_argument("--fps", type=float, default=10.)
    parser.add_argument("--segment-sec", type=float, default=60.)
    parser.add_argument("--max-session-sec", type=float, default=0., help="0 records until Ctrl+C (default).")
    parser.add_argument("--min-free-gb", type=float, default=2.)
    parser.add_argument("--state-timeout", type=float, default=2., help="Age limit for optional state annotations only.")
    args = parser.parse_args(argv)
    values = (args.fps, args.segment_sec, args.min_free_gb, args.state_timeout)
    if not all(np.isfinite(value) and value > 0 for value in values) or args.fps > 30:
        parser.error("limits must be finite and positive; --fps must not exceed 30")
    if not np.isfinite(args.max_session_sec) or args.max_session_sec < 0:
        parser.error("--max-session-sec must be finite and nonnegative (0 disables the time limit)")
    return args


def main():
    args = parse_args()
    args.output, args.capture_root = args.output.resolve(), args.capture_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)

    import rclpy
    from ament_index_python.packages import get_package_share_path
    from rclpy.qos import QoSProfile, qos_profile_sensor_data
    from rclpy.signals import SignalHandlerOptions
    from sensor_msgs.msg import CameraInfo, Image
    from std_msgs.msg import String

    mod_root = get_package_share_path("bxi_example_py_elf3") / "mods/com.bxi.normal_depth"
    cv2.setNumThreads(1)
    state, policy = load_pipeline(mod_root)
    package = state.__class__.__module__.rsplit(".", 1)[0]
    depth_topic, info_topic = importlib.import_module(f"{package}.camera_topics").resolve_camera_topics("hardware", state.camera_name)
    topics = {"depth": state.depth_image_topic or depth_topic, "info": state.camera_info_topic or info_topic,
              "color": f"/hardware/{state.camera_name}/color/image_raw",
              "color_info": f"/hardware/{state.camera_name}/color/camera_info", "state": args.state_topic}
    sha256 = hashlib.sha256((mod_root / "assets/dagger_new_position.onnx").read_bytes()).hexdigest()
    contract = {name: (list(value.shape), str(value.dtype)) for name, value in policy._policy_inputs.items()}
    reader = BackendRecordReader(args.capture_root / "dagger_new_position", sha256, contract)
    state_info, monitor = StateInfo(args.state_timeout), CameraMonitor(state, policy)
    rclpy.init(args=[], signal_handler_options=SignalHandlerOptions.NO)
    node = rclpy.create_node("dagger_camera_recorder")
    state._bind_logger(node.get_logger())
    identity = {"name": "dagger_new_position", "sha256": sha256, "installed_mod": str(mod_root),
                "depth_range": policy.depth_range, "projection": asdict(state.projection),
                "depth_uint16_scale": state.depth_uint16_scale, "inputs": contract}
    worker = RecorderWorker(args, state_info, monitor, reader, identity,
                            topics, log=lambda message: node.get_logger().info(message))
    latest = {"info": None, "color": None, "color_info": None, "color_received": float("-inf")}
    qos = QoSProfile(depth=1, reliability=qos_profile_sensor_data.reliability,
                     durability=qos_profile_sensor_data.durability)

    def on_state(message):
        publishers = node.count_publishers(args.state_topic)
        if publishers != 1 and latest.get("state_publishers") != publishers:
            node.get_logger().warning(f"state annotation unavailable: publishers={publishers}; recording continues")
        latest["state_publishers"] = publishers
        state_info.update(message.data if publishers == 1 else "null")

    def on_color(message):
        latest.update(color=message, color_received=time.monotonic())

    def on_depth(message):
        color = latest["color"] if time.monotonic() - latest["color_received"] <= .5 else None
        worker.submit(message, latest["info"], color, latest["color_info"] if color is not None else None)

    subscriptions = [node.create_subscription(String, args.state_topic, on_state, QoSProfile(depth=10)),
                     node.create_subscription(CameraInfo, topics["info"], lambda msg: latest.update(info=msg), qos),
                     node.create_subscription(CameraInfo, topics["color_info"], lambda msg: latest.update(color_info=msg), qos),
                     node.create_subscription(Image, topics["color"], on_color, qos),
                     node.create_subscription(Image, topics["depth"], on_depth, qos)]
    thread = threading.Thread(target=worker.run, name="local-camera-writer")
    thread.start()
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, lambda signum, frame: worker.stop.set())
    node.get_logger().info(f"RECORDER READY: mode=manual; output={args.output}; fps<={args.fps}; "
                           f"time_limit_sec={args.max_session_sec}; waiting for depth frames; "
                           "Ctrl+C stops recording; robot state is optional; no robot command is published")
    try:
        while rclpy.ok() and not worker.stop.is_set():
            rclpy.spin_once(node, timeout_sec=.1)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        worker.stop.set()
        thread.join()
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        policy.close()


if __name__ == "__main__":
    main()
