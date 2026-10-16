#!/usr/bin/env python3
"""Read-only ROS depth audit using the installed DAGGER preprocessing code."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np


def value_summary(values):
    array = np.asarray(values)
    finite = array[np.isfinite(array)]
    result = {"shape": list(array.shape), "dtype": str(array.dtype),
              "nonfinite_fraction": float(np.mean(~np.isfinite(array))) if array.size else None}
    if finite.size:
        result.update(zip(("min", "p05", "p50", "p95", "max"),
                          map(float, np.percentile(finite, [0, 5, 50, 95, 100]))))
    return result


def depth_summary(depth):
    result = value_summary(depth)
    valid = np.isfinite(depth) & (depth > 0)
    result["invalid_fraction"] = float(np.mean(~valid))
    result["positive_depth_m"] = value_summary(depth[valid]) if valid.any() else None
    return result


def sequence_summary(stamps, receipts, digests):
    stamps = np.asarray(stamps, dtype=np.int64)
    receipts = np.asarray(receipts, dtype=np.float64)
    result = {"frames": int(stamps.size)}
    if stamps.size < 2:
        return result
    deltas = np.diff(stamps)
    elapsed = float(receipts[-1] - receipts[0])
    result.update({
        "receipt_hz": float((stamps.size - 1) / elapsed) if elapsed > 0 else None,
        "stamp_interval_ms": value_summary(deltas.astype(np.float64) / 1e6),
        "duplicate_stamps": int(np.sum(deltas == 0)),
        "backward_stamps": int(np.sum(deltas < 0)),
        "identical_consecutive_images": sum(a == b for a, b in zip(digests, digests[1:])),
    })
    return result


def load_pipeline(mod_root):
    import yaml
    from bxi_example_py_elf3.framework.mod_api import ResourceLoadContext, StateBuildContext

    package_name = "_dagger_camera_audit_mod"
    package = types.ModuleType(package_name)
    package.__path__ = [str(mod_root)]
    sys.modules[package_name] = package
    plugin = importlib.import_module(f"{package_name}.plugin")
    manifest = yaml.safe_load((mod_root / "mod.yaml").read_text())
    context = StateBuildContext("camera_audit", 1,
                                manifest["states"]["normal_depth"]["params"])
    policy = plugin._load_origin_policy(ResourceLoadContext("com.bxi.normal_depth", mod_root))
    handle = types.SimpleNamespace(get=lambda: policy)
    state = plugin._build_normal_depth_state(context, handle, handle)
    context.finish()
    if state.mode != "origin_camera":
        policy.close()
        raise ValueError("this audit requires the installed origin_camera state")
    return state, policy


def stamp_ns(message):
    return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)


def camera_info_dict(info):
    return {"width": int(info.width), "height": int(info.height),
            "frame_id": info.header.frame_id, "stamp_ns": stamp_ns(info),
            "distortion_model": info.distortion_model,
            **{name: list(getattr(info, name)) for name in ("k", "d", "r", "p")}}


def probe_model(policy, tensor):
    inputs = {name: np.zeros_like(value) for name, value in policy._policy_inputs.items()}
    np.copyto(inputs[policy.depth_input_name], tensor[np.newaxis])
    outputs = policy._backend.run(inputs)
    if not all(np.isfinite(value).all() for value in outputs.values()):
        raise ValueError("nonfinite output from offline camera-depth probe")
    return {
        "scope": "real camera depth with ZERO SYNTHETIC obs_history/h_in; NOT robot proprioception",
        "backend": policy._backend.backend_name,
        "inputs": {name: value_summary(value) for name, value in inputs.items()},
        "outputs": {name: value_summary(value) for name, value in outputs.items()},
    }


def save_capture(directory, state, policy, frames, timings, errors, topics):
    import cv2
    from dataclasses import asdict

    last = frames[-1]
    raw, rotated, tensor = last["raw"], last["rotated"], last["tensor"]
    info = last["info"]
    projection = importlib.import_module(f"{state.__class__.__module__.rsplit('.', 1)[0]}.depth_projection")
    _, crop = projection.project_depth(raw, info, state.projection)
    x, y, width, height = crop
    roi = raw[y:y + height, x:x + width]
    report = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "camera-only preprocessing preview; NOT captured controller backend input",
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", "0"),
        "topics": topics, "camera_info": camera_info_dict(info),
        "projection": asdict(state.projection), "crop_xywh": list(crop),
        "depth_uint16_scale_m": state.depth_uint16_scale,
        "preprocessing_gaussian_available": policy.__class__._preprocess_depth.__globals__["_get_cv2"]() is not None,
        "received": sequence_summary(*timings),
        "accepted_frames": len(frames), "errors": errors,
        "stamp_age_at_receive_ms": value_summary(np.array([f["age_ms"] for f in frames])),
        "raw_invalid_fraction": value_summary(np.array([f["invalid"] for f in frames])),
        "raw_depth_last": depth_summary(raw), "crop_depth_last": depth_summary(roi),
        "tensor_preview_last": value_summary(tensor),
        "tensor_min_saturation_fraction": float(np.mean(tensor <= -.5)),
        "tensor_max_saturation_fraction": float(np.mean(tensor >= .5)),
        "offline_model_probe": probe_model(policy, tensor),
        "image_header": last["header"],
        "note": "Driver stamps ROS publish time, not device exposure time. Receipt age is not end-to-end latency.",
    }
    directory.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(directory / "depth_snapshot.npz", raw_depth_m=raw,
                        state_depth_rotated=rotated, tensor_preview=tensor[np.newaxis],
                        crop_xywh=np.asarray(crop), camera_k=np.asarray(info.k),
                        stamp_ns=np.asarray(last["header"]["stamp_ns"]))
    normalized = np.clip((raw - .2) / 2.8, 0., 1.)
    preview = cv2.applyColorMap(np.uint8(np.nan_to_num(normalized) * 255), cv2.COLORMAP_TURBO)
    preview[~(np.isfinite(raw) & (raw > 0))] = 0
    cv2.rectangle(preview, (x, y), (x + width - 1, y + height - 1), (0, 255, 255), 2)
    model = cv2.applyColorMap(np.uint8(np.clip(tensor + .5, 0, 1) * 255), cv2.COLORMAP_TURBO)
    model = cv2.resize(model, (360, 360), interpolation=cv2.INTER_NEAREST)
    for name, image in (("native_depth_with_crop.png", preview), ("tensor_preview_36x36.png", model)):
        if not cv2.imwrite(str(directory / name), image):
            raise RuntimeError(f"cannot save {name}")
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    print(f"Output: {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not 0 < args.seconds <= 60:
        parser.error("--seconds must be in (0,60]")
    output = args.output.expanduser().resolve()
    if output.exists():
        parser.error("--output must be a new directory")

    import rclpy
    from ament_index_python.packages import get_package_share_path
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image

    mod_root = get_package_share_path("bxi_example_py_elf3") / "mods/com.bxi.normal_depth"
    state, policy = load_pipeline(mod_root)
    camera_topics = importlib.import_module("_dagger_camera_audit_mod.camera_topics")
    topics = (state.depth_image_topic, state.camera_info_topic)
    if not topics[0]:
        topics = camera_topics.resolve_camera_topics("hardware", state.camera_name)
    state.depth_image_topic, state.camera_info_topic = topics
    rclpy.init(args=[])
    node = rclpy.create_node("dagger_camera_input_audit")
    state._bind_logger(node.get_logger())
    frames, stamps, receipts, digests, errors = [], [], [], [], []

    def image_callback(message):
        received = time.monotonic()
        received_ros_ns = node.get_clock().now().nanoseconds
        stamps.append(stamp_ns(message))
        receipts.append(received)
        digests.append(hashlib.blake2s(message.data).hexdigest())
        try:
            previous = state._latest_depth_frame_id
            state.depth_image_callback(message)
            if state._latest_depth_frame_id == previous:
                return
            raw = state._depth_msg_to_meters(message)
            if raw is None:
                raise ValueError("depth decoder did not accept this frame")
            tensor = policy._preprocess_depth(state._depth_rotated).copy()
            if tensor.shape != (36, 36) or not np.isfinite(tensor).all():
                raise ValueError("invalid tensor preview")
            if not np.all((tensor >= -.5) & (tensor <= .5)):
                raise ValueError("normalized depth preview out of range")
            # Keep only the final full image; per-frame scalars remain for statistics.
            if frames:
                for key in ("raw", "rotated", "tensor", "info"):
                    frames[-1].pop(key, None)
            frames.append({"raw": raw, "rotated": state._depth_rotated.copy(),
                           "tensor": tensor, "info": state._camera_info,
                           "invalid": float(np.mean(~(np.isfinite(raw) & (raw > 0)))),
                           "age_ms": (received_ros_ns - stamps[-1]) / 1e6,
                           "header": {"stamp_ns": stamps[-1], "frame_id": message.header.frame_id,
                                      "encoding": message.encoding, "width": message.width,
                                      "height": message.height, "step": message.step}})
        except Exception as exc:
            if len(errors) < 10:
                errors.append(str(exc))

    subscriptions = [
        node.create_subscription(CameraInfo, topics[1], state.camera_info_callback, qos_profile_sensor_data),
        node.create_subscription(Image, topics[0], image_callback, qos_profile_sensor_data),
    ]
    print(f"Camera-only subscriber: {topics}; no motor commands or services")
    try:
        end = time.monotonic() + args.seconds
        while rclpy.ok() and time.monotonic() < end:
            rclpy.spin_once(node, timeout_sec=min(.1, max(0., end - time.monotonic())))
        if not frames:
            print(f"No accepted depth frames: received={len(stamps)}, CameraInfo={state._camera_info is not None}, errors={errors}",
                  file=sys.stderr)
            return 1
        save_capture(output, state, policy, frames, (stamps, receipts, digests), errors, list(topics))
        return int(bool(errors))
    finally:
        for subscription in subscriptions:
            node.destroy_subscription(subscription)
        node.destroy_node()
        rclpy.shutdown()
        policy.close()


if __name__ == "__main__":
    raise SystemExit(main())
