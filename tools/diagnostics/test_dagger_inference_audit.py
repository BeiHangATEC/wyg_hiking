import hashlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dagger_inference_audit as audit
from calibration_capture import CaptureConfig, CalibrationDatasetWriter, ModelIdentity


def inputs():
    return {name: np.full(shape, .125, dtype=np.float32) for name, shape in audit.CONTRACT.items()}


def controller(name=audit.TARGET, transition=False):
    snapshot = {"current": {"name": name}, "in_transition": transition}
    return types.SimpleNamespace(
        state_machine=types.SimpleNamespace(snapshot=lambda: snapshot),
        inference_frame=types.SimpleNamespace(depth_frame_id=np.int64(17)),
    )


class Backend:
    backend_name = "onnxruntime"

    def __init__(self, error=None, mutate=False):
        self.calls = 0
        self.error = error
        self.mutate = mutate
        self.outputs = {"actions": np.ones((1, 29), dtype=np.float32),
                        "h_out": np.ones((1, 1, 256), dtype=np.float32)}

    def run(self, values):
        self.calls += 1
        if self.error:
            raise self.error
        if self.mutate:
            values["h_in"].fill(9.)
        return self.outputs


@pytest.fixture
def recording(tmp_path):
    model = tmp_path / "dagger_new_position.onnx"
    model.write_bytes(b"test model identity, not a deployed camera capture")
    identity = ModelIdentity(audit.MODEL, str(model), hashlib.sha256(model.read_bytes()).hexdigest(), "onnxruntime")
    writer = CalibrationDatasetWriter(CaptureConfig(tmp_path / "records", 1, 100, 0, 16), identity)
    yield writer, model
    writer.close()
    audit.CONTROL.controller = None


def test_control_context_is_thread_local_and_restored_after_exception():
    first, second = controller(), controller("com.bxi.basic_actions/normal")
    audit.CONTROL.controller = first

    @audit.bind_control_cycle
    def update(current):
        assert audit.control_context()["state"]["current"]["name"] == current.state_machine.snapshot()["current"]["name"]
        raise ValueError("backend failure")

    with pytest.raises(ValueError, match="backend failure"):
        update(second)
    assert audit.CONTROL.controller is first
    audit.CONTROL.controller = None
    assert audit.control_context()["scope"] == "outside_control_cycle"


def test_actual_inputs_are_copied_before_backend_reuses_buffers(recording):
    writer, model = recording
    backend = Backend(mutate=True)
    wrapped = audit.AuditBackend(backend, writer, every=1, skip=0)
    audit.CONTROL.controller = controller()
    values = inputs()
    returned = wrapped.run(values)
    writer.close()
    assert returned is backend.outputs
    assert np.all(values["h_in"] == 9.)
    assert np.all(np.load(writer.directory / "000000_02_h_in.npy") == .125)
    annotation = json.loads((writer.directory / "000000.audit.json").read_text())
    assert annotation["inference"]["context"]["depth_frame_id"] == 17
    assert annotation["inference"]["success"] is True
    result = audit.inspect_recording(model.parent / "records", model)
    assert result["confirmed"]
    assert result["models"][0]["confirmed_dagger_samples"] == 1
    assert result["models"][0]["unique_depth_tensors"] == 1


@pytest.mark.parametrize("name,transition,scope", [
    ("com.bxi.basic_actions/normal", False, "cycle"),
    ("com.bxi.basic_actions/pd_brake", False, "cycle"),
    (audit.TARGET, True, "cycle"),
    (audit.TARGET, False, "outside"),
])
def test_preparation_transition_and_warmup_are_not_confirmed(recording, name, transition, scope):
    writer, model = recording
    audit.CONTROL.controller = controller(name, transition) if scope == "cycle" else None
    audit.AuditBackend(Backend(), writer, every=1, skip=0).run(inputs())
    writer.close()
    assert not audit.inspect_recording(model.parent / "records", model)["confirmed"]


def test_model_failure_is_recorded_and_original_exception_preserved(recording):
    writer, model = recording
    failure = ValueError("original model error")
    backend = Backend(error=failure)
    audit.CONTROL.controller = controller()
    with pytest.raises(ValueError) as caught:
        audit.AuditBackend(backend, writer, every=1, skip=0).run(inputs())
    assert caught.value is failure
    writer.close()
    record = json.loads((writer.directory / "000000.audit.json").read_text())
    assert record["inference"]["success"] is False
    assert not audit.inspect_recording(model.parent / "records", model)["confirmed"]


def test_recorder_failure_does_not_change_backend_calls_or_result(recording, monkeypatch):
    writer, _ = recording
    backend = Backend()
    wrapped = audit.AuditBackend(backend, writer, every=1, skip=0)

    def fail(*args, **kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(writer, "capture", fail)
    assert wrapped.run(inputs()) is backend.outputs
    assert wrapped.run(inputs()) is backend.outputs
    assert wrapped._disabled
    assert backend.calls == 2


def test_sampling_does_not_skip_any_real_inference_call(recording):
    writer, _ = recording
    backend = Backend()
    wrapped = audit.AuditBackend(backend, writer, every=3, skip=5)
    for _ in range(12):
        wrapped.run(inputs())
    writer.close()
    assert backend.calls == 12
    calls = [json.loads(path.read_text())["inference"]["backend_call"]
             for path in sorted(writer.directory.glob("*.audit.json"))]
    assert calls == [6, 9, 12]


def test_existing_capture_without_annotations_remains_compatible(recording):
    writer, _ = recording
    assert writer.capture(inputs())
    writer.close()
    assert len(writer.dataset_path.read_text().splitlines()) == 1
    assert not list(writer.directory.glob("*.audit.json"))
    assert np.load(writer.directory / "000000_01_depth_data.npy").shape == (1, 36, 36)


@pytest.mark.parametrize("damage", ["hash", "array", "annotation", "nonfinite_output"])
def test_corrupt_or_wrong_model_evidence_cannot_confirm_dagger(recording, damage):
    writer, model = recording
    audit.CONTROL.controller = controller()
    audit.AuditBackend(Backend(), writer, every=1, skip=0).run(inputs())
    writer.close()
    if damage == "hash":
        model.write_bytes(b"a different installed model")
    elif damage == "array":
        np.save(writer.directory / "000000_01_depth_data.npy", np.zeros((1, 48, 36), dtype=np.float32))
    elif damage == "annotation":
        (writer.directory / "000000.audit.json").unlink()
    else:
        path = writer.directory / "000000.audit.json"
        record = json.loads(path.read_text())
        record["inference"]["output_summary"]["actions"]["finite"] = False
        path.write_text(json.dumps(record))
    assert not audit.inspect_recording(model.parent / "records", model)["confirmed"]


def test_opt_in_hook_with_real_onnx_and_stub_control_cycle(tmp_path):
    output = tmp_path / "subprocess-audit"
    model = audit.PROJECT / "install/share/bxi_example_py_elf3/mods/com.bxi.normal_depth/assets/dagger_new_position.onnx"
    code = f"""
import numpy as np
from types import SimpleNamespace
from bxi_example_py_elf3.framework.inference import InferenceRuntime, ModelSpec
from bxi_example_py_elf3.framework.runtime.controller import RobotControlFramework
spec = ModelSpec.portable_onnx({str(model)!r}, input_names=(), output_names=())
backend = InferenceRuntime().open_backend(spec, backend='onnxruntime')
values = {{'obs_history': np.zeros((1,960), dtype=np.float32),
          'depth_data': np.zeros((1,36,36), dtype=np.float32),
          'h_in': np.zeros((1,1,256), dtype=np.float32)}}
control = RobotControlFramework.__new__(RobotControlFramework)
control._closed = False
control._inference_frame = SimpleNamespace(depth_frame_id=12)
control._set_observation = lambda value: None
control._apply_pending_state_requests = lambda: None
control.current_cmd_vel = np.zeros(3, dtype=np.float32)
control.loop_count = 0
control.state_machine = SimpleNamespace(
    snapshot=lambda: {{'current': {{'name': {audit.TARGET!r}}}, 'in_transition': False}},
    update=lambda dt, events: False,
    update_current_state=lambda dt: backend.run(values))
control.update(None, [], .02)
backend.close()
"""
    assert audit.main(["--output", str(output), "--every", "1", "--skip-first", "0", "--",
                       sys.executable, "-c", code]) == 0
    result = audit.inspect_recording(output, model)
    assert result["confirmed"]
    assert result["models"][0]["confirmed_dagger_samples"] == 1
    assert result["models"][0]["unique_depth_frame_ids"] == 1

