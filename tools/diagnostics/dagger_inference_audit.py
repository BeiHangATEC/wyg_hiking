#!/usr/bin/env python3
"""Opt-in, tool-side correlation of control state and actual backend inputs."""

from __future__ import annotations

import argparse
import atexit
from collections.abc import Mapping
from datetime import datetime
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
BENCHMARK = PROJECT / "tools/benchmark"
if str(BENCHMARK) not in sys.path:
    sys.path.insert(0, str(BENCHMARK))
from calibration_capture import CaptureConfig, CaptureManager, CalibrationDatasetWriter, _model_identity

ACTIVE_ENV = "BXI_DAGGER_INPUT_AUDIT"
ROOT_ENV = "BXI_DAGGER_INPUT_AUDIT_ROOT"
EVERY_ENV = "BXI_DAGGER_INPUT_AUDIT_EVERY"
MAX_ENV = "BXI_DAGGER_INPUT_AUDIT_MAX"
SKIP_ENV = "BXI_DAGGER_INPUT_AUDIT_SKIP"
TARGET = "com.bxi.normal_depth/normal_depth"
MODEL = "dagger_new_position"
CONTRACT = {"obs_history": (1, 960), "depth_data": (1, 36, 36), "h_in": (1, 1, 256)}
CONTROL = threading.local()


def bind_control_cycle(original):
    @wraps(original)
    def update(controller, *args, **kwargs):
        previous = getattr(CONTROL, "controller", None)
        CONTROL.controller = controller
        try:
            return original(controller, *args, **kwargs)
        finally:
            CONTROL.controller = previous
    return update


def control_context():
    controller = getattr(CONTROL, "controller", None)
    if controller is None:
        return {"scope": "outside_control_cycle", "state": None, "depth_frame_id": None}
    try:
        state = controller.state_machine.snapshot()
        frame = controller.inference_frame
        frame_id = frame.depth_frame_id
        return {
            "scope": "control_cycle",
            "state": state,
            "depth_frame_id": int(frame_id) if frame_id is not None else None,
        }
    except Exception as error:
        return {"scope": "context_unavailable", "state": None, "error": str(error)}


def array_summary(value):
    array = np.asarray(value)
    finite = np.isfinite(array)
    valid = array[finite].astype(np.float64, copy=False)
    return {
        "shape": list(array.shape), "dtype": str(array.dtype),
        "finite": bool(finite.all()),
        "min": float(valid.min()) if valid.size else None,
        "max": float(valid.max()) if valid.size else None,
        "std": float(valid.std()) if valid.size else None,
    }


class AuditBackend:
    def __init__(self, backend, writer, *, every, skip):
        self._backend = backend
        self._writer = writer
        self._every = every
        self._skip = skip
        self._calls = 0
        self._captured = 0
        self._disabled = False
        self._last_message = 0.0
        self._last_label = None

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def _disable(self, error):
        if not self._disabled:
            self._disabled = True
            print(f"[input-audit] DISABLED: {error}; robot backend remains unchanged", file=sys.stderr, flush=True)

    def _save(self, inputs, record):
        try:
            record["input_summary"] = {name: array_summary(value) for name, value in inputs.items()}
            if not self._writer.capture(inputs, annotation=record):
                return
            self._captured += 1
            state = record["context"].get("state") or {}
            name = state.get("current", {}).get("name", "unavailable")
            label = (name, state.get("in_transition"), record["success"])
            now = time.monotonic()
            if label != self._last_label or now - self._last_message >= 1.0:
                self._last_message, self._last_label = now, label
                print(
                    f"[input-audit] QUEUED model={self._writer.identity.name} "
                    f"sha256={self._writer.identity.sha256} state={name} "
                    f"transition={state.get('in_transition')} success={record['success']} "
                    f"depth_frame={record['context'].get('depth_frame_id')} samples={self._captured}",
                    flush=True,
                )
        except Exception as error:
            self._disable(error)

    def run(self, inputs):
        self._calls += 1
        eligible = self._calls - self._skip
        selected = (
            not self._disabled and self._captured < self._writer.max_samples
            and eligible > 0 and (eligible - 1) % self._every == 0
        )
        if not selected:
            return self._backend.run(inputs)
        try:
            # Snapshot before run: backend/recurrent buffers can be reused in-place.
            snapshot = {name: np.array(value, order="C", copy=True) for name, value in inputs.items()}
            record = {
                "schema": 1, "backend_call": self._calls,
                "call_started_unix_ns": time.time_ns(), "context": control_context(),
            }
        except Exception as error:
            self._disable(error)
            return self._backend.run(inputs)
        started = time.perf_counter_ns()
        try:
            outputs = self._backend.run(inputs)
        except Exception as error:
            record.update(success=False, error=str(error), run_ns=time.perf_counter_ns() - started)
            self._save(snapshot, record)
            raise
        record.update(success=True, run_ns=time.perf_counter_ns() - started)
        try:
            record["output_summary"] = {name: array_summary(value) for name, value in outputs.items()}
        except Exception as error:
            self._disable(error)
            return outputs
        self._save(snapshot, record)
        return outputs

    def warmup(self, inputs, runs=1):
        for _ in range(max(0, int(runs))):
            self.run(inputs)


class AuditManager(CaptureManager):
    def __init__(self, config, *, every, skip):
        super().__init__(config)
        self.every, self.skip = every, skip

    def wrap(self, backend, spec):
        identity = _model_identity(spec, backend.backend_name)
        key = (identity.name, identity.sha256)
        writer = self._writers.get(key)
        if writer is None:
            writer = CalibrationDatasetWriter(self.config, identity)
            self._writers[key] = writer
        return AuditBackend(backend, writer, every=self.every, skip=self.skip)


def install_from_environment():
    if os.environ.get(ACTIVE_ENV) != "1":
        return None
    try:
        from bxi_example_py_elf3.framework.inference.runtime import InferenceRuntime
        from bxi_example_py_elf3.framework.runtime.controller import RobotControlFramework
    except ModuleNotFoundError:
        return None
    existing = getattr(InferenceRuntime, "_bxi_input_audit", None)
    if existing is not None:
        return existing
    every, maximum, skip = (int(os.environ[key]) for key in (EVERY_ENV, MAX_ENV, SKIP_ENV))
    if every <= 0 or maximum <= 0 or skip < 0:
        raise ValueError("invalid inference audit sampling configuration")
    manager = AuditManager(
        CaptureConfig(Path(os.environ[ROOT_ENV]), 1, maximum, 0, 16), every=every, skip=skip,
    )
    original_open = InferenceRuntime.open_backend

    @wraps(original_open)
    def open_backend(runtime, spec, *, backend=None):
        selected = original_open(runtime, spec, backend=backend)
        try:
            return manager.wrap(selected, spec)
        except Exception as error:
            print(f"[input-audit] cannot open recorder: {error}; using unwrapped backend", file=sys.stderr, flush=True)
            return selected

    RobotControlFramework.update = bind_control_cycle(RobotControlFramework.update)
    InferenceRuntime.open_backend = open_backend
    InferenceRuntime._bxi_input_audit = manager
    atexit.register(manager.close)
    print("[input-audit] ENABLED: controller state + actual backend input; no state request is sent", flush=True)
    return manager


def inspect_recording(root, expected_model):
    expected_hash = hashlib.sha256(expected_model.read_bytes()).hexdigest()
    report = {"root": str(root), "expected_model_sha256": expected_hash, "models": [], "confirmed": False}
    for metadata_path in sorted(root.glob("*/capture.json")):
        directory = metadata_path.parent
        metadata = json.loads(metadata_path.read_text())
        item = {
            "model": metadata["model"], "source": metadata["model_source"],
            "sha256": metadata["model_sha256"], "backend": metadata["capture_backend"],
            "committed_samples": 0, "running_states": {}, "confirmed_dagger_samples": 0,
            "rejected_samples": 0, "depth_frame_ids": set(), "depth_hashes": set(),
            "depth_min": None, "depth_max": None,
        }
        contract_ok = {value["name"]: (tuple(value["shape"]), value["dtype"]) for value in metadata["inputs"]} == {
            name: (shape, "float32") for name, shape in CONTRACT.items()
        }
        dataset = directory / "dataset.txt"
        if not dataset.exists():
            continue
        # dataset.txt commits only after arrays and the annotation are complete.
        for line in dataset.read_text().splitlines():
            if not line.strip():
                continue
            item["committed_samples"] += 1
            try:
                paths = [Path(value) for value in line.split()]
                sample = int(paths[0].name.split("_", 1)[0])
                audit = json.loads((directory / f"{sample:06d}.audit.json").read_text())["inference"]
                state = audit["context"].get("state") or {}
                current = state.get("current", {}).get("name", "unavailable")
                item["running_states"][current] = item["running_states"].get(current, 0) + 1
                outputs_ok = bool(audit.get("output_summary")) and all(
                    value["finite"] for value in audit["output_summary"].values()
                )
                if not (
                    metadata["model"] == MODEL and metadata["model_sha256"] == expected_hash
                    and contract_ok and current == TARGET and state.get("in_transition") is False
                    and audit["context"]["scope"] == "control_cycle" and audit["success"] and outputs_ok
                    and isinstance(audit["context"].get("depth_frame_id"), int)
                    and audit["context"]["depth_frame_id"] > 0
                ):
                    continue
                arrays = {}
                for info, path in zip(metadata["inputs"], paths):
                    if path.resolve().parent != directory.resolve():
                        raise ValueError("input path outside capture directory")
                    array = np.load(path, allow_pickle=False)
                    if array.dtype != np.float32 or array.shape != CONTRACT[info["name"]] or not np.isfinite(array).all():
                        raise ValueError("invalid saved tensor")
                    arrays[info["name"]] = array
                if len(arrays) != len(CONTRACT) or len(paths) != len(CONTRACT):
                    raise ValueError("missing or additional saved tensor")
                depth = arrays["depth_data"]
                item["confirmed_dagger_samples"] += 1
                frame_id = audit["context"].get("depth_frame_id")
                if frame_id is not None:
                    item["depth_frame_ids"].add(frame_id)
                item["depth_hashes"].add(hashlib.sha256(depth.tobytes()).hexdigest())
                low, high = float(depth.min()), float(depth.max())
                item["depth_min"] = low if item["depth_min"] is None else min(item["depth_min"], low)
                item["depth_max"] = high if item["depth_max"] is None else max(item["depth_max"], high)
            except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
                item["rejected_samples"] += 1
                item["last_rejection"] = str(error)
        item["unique_depth_frame_ids"] = len(item.pop("depth_frame_ids"))
        item["unique_depth_tensors"] = len(item.pop("depth_hashes"))
        report["models"].append(item)
        report["confirmed"] |= item["confirmed_dagger_samples"] > 0
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, help="read saved evidence only; does not launch hardware")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--every", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--skip-first", type=int, default=5)
    parser.add_argument("--expected-model", type=Path, default=PROJECT / f"install/share/bxi_example_py_elf3/mods/com.bxi.normal_depth/assets/{MODEL}.onnx")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.report is not None:
        report = inspect_recording(args.report.resolve(), args.expected_model)
        print(json.dumps(report, indent=2))
        return 0 if report["confirmed"] else 2
    if args.every <= 0 or args.max_samples <= 0 or args.skip_first < 0:
        parser.error("every/max-samples must be positive; skip-first must be non-negative")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a launch command is required after --")
    output = (args.output or PROJECT / "logs/dagger-input-audit" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error("output must be a new or empty directory; do not mix different tests")
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    hook = PROJECT / "tools/diagnostics/inference_audit_sitecustomize"
    environment.update({
        ACTIVE_ENV: "1", ROOT_ENV: str(output), EVERY_ENV: str(args.every),
        MAX_ENV: str(args.max_samples), SKIP_ENV: str(args.skip_first),
        "PYTHONPATH": os.pathsep.join([str(hook), str(Path(__file__).parent), str(BENCHMARK), environment.get("PYTHONPATH", "")]),
    })
    print(f"[input-audit] output={output}; every={args.every}; max-samples/model={args.max_samples}", flush=True)
    print("[input-audit] this command starts hardware; select policies manually; sample limit does not stop robot", flush=True)
    try:
        return subprocess.run(command, env=environment, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
