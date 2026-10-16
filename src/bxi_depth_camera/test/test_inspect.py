import math
from types import SimpleNamespace

import pytest

from bxi_depth_camera.inspect import CameraIdentity, Intrinsics, _matching
from bxi_depth_camera import inspect as camera_inspect


def test_intrinsics_compute_horizontal_and_vertical_fov():
    intrinsics = Intrinsics(
        fx=320.0,
        fy=240.0,
        cx=319.5,
        cy=239.5,
        width=640,
        height=480,
    )

    assert intrinsics.hfov_deg == pytest.approx(90.0)
    assert intrinsics.vfov_deg == pytest.approx(90.0)


def test_invalid_intrinsics_do_not_produce_non_finite_fov():
    intrinsics = Intrinsics(0.0, math.nan, 0.0, 0.0, 640, 480)

    assert intrinsics.hfov_deg == 0.0
    assert intrinsics.vfov_deg == 0.0


def test_serial_filter_keeps_backend_identity():
    first = CameraIdentity("realsense", "123", "D435", "usb-1")
    second = CameraIdentity("orbbec", "ABC", "Gemini 335", "usb-2")
    devices = {first.key: first, second.key: second}

    assert _matching(devices, "123") == {first.key: first}
    assert _matching(devices, "") == devices


@pytest.mark.parametrize("available", [False, True])
def test_depth_preflight_only_opens_selected_camera(monkeypatch, available):
    old = CameraIdentity("realsense", "old", "D435I")
    new = CameraIdentity("realsense", "new", "D435")
    identities = {old.key: old}
    if available:
        identities[new.key] = new
    monkeypatch.setattr(camera_inspect, "_load_backends", lambda: (object(), object(), None, None))
    monkeypatch.setattr(camera_inspect, "_scan", lambda *_: (identities, []))
    opened = []

    def inspect_selected(identity, *_):
        opened.append(identity.serial)
        return camera_inspect.CameraReport(
            identity,
            streams=[camera_inspect.StreamInfo("depth", 848, 480, 30)],
            intrinsics={"depth": Intrinsics(428., 428., 424., 240., 848, 480)},
            depth_scale_m=.001,
            depth_frame_received=True,
        )

    monkeypatch.setattr(camera_inspect, "_inspect", inspect_selected)
    result = camera_inspect._run_once(SimpleNamespace(serial="new", require_depth=True))
    assert result == (0 if available else 1)
    assert opened == (["new"] if available else [])


@pytest.mark.parametrize("require_depth, expected", [(False, 0), (True, 1)])
def test_depth_preflight_rejects_busy_camera(monkeypatch, require_depth, expected):
    identity = CameraIdentity("realsense", "new", "D435")
    monkeypatch.setattr(camera_inspect, "_load_backends", lambda: (object(), object(), None, None))
    monkeypatch.setattr(camera_inspect, "_scan", lambda *_: ({identity.key: identity}, []))
    monkeypatch.setattr(camera_inspect, "_inspect", lambda *_: camera_inspect.CameraReport(identity))
    assert camera_inspect._run_once(SimpleNamespace(serial="new", require_depth=require_depth)) == expected


@pytest.mark.parametrize("arguments", [["--require-depth"], ["--require-depth", "--serial", "new", "--watch"]])
def test_depth_preflight_requires_one_explicit_serial(arguments):
    with pytest.raises(SystemExit) as result:
        camera_inspect.main(arguments)
    assert result.value.code == 2


def test_depth_preflight_rejects_open_pipeline_without_frames(monkeypatch):
    identity = CameraIdentity("realsense", "new", "D435")
    report = camera_inspect.CameraReport(
        identity, streams=[camera_inspect.StreamInfo("depth", 848, 480, 30)],
        intrinsics={"depth": Intrinsics(428., 428., 424., 240., 848, 480)},
        depth_scale_m=.001,
    )
    monkeypatch.setattr(camera_inspect, "_load_backends", lambda: (object(), object(), None, None))
    monkeypatch.setattr(camera_inspect, "_scan", lambda *_: ({identity.key: identity}, []))
    monkeypatch.setattr(camera_inspect, "_inspect", lambda *_: report)
    assert camera_inspect._run_once(SimpleNamespace(serial="new", require_depth=True)) == 1


@pytest.mark.parametrize("frame_arrives", [True, False])
def test_realsense_inspection_checks_frames_and_always_closes_pipeline(monkeypatch, frame_arrives):
    stopped = []

    def wait(timeout_ms):
        assert timeout_ms == 3000
        if not frame_arrives:
            raise RuntimeError("Frame did not arrive")
        return SimpleNamespace(get_depth_frame=lambda: True)

    pipeline = SimpleNamespace(wait_for_frames=wait, stop=lambda: stopped.append(True))
    device = SimpleNamespace(get_info=lambda _: "new")
    rs = SimpleNamespace(camera_info=SimpleNamespace(serial_number="serial"),
                         context=lambda: SimpleNamespace(query_devices=lambda: [device]))
    profile = SimpleNamespace(get_streams=lambda: [])
    monkeypatch.setattr(camera_inspect, "_realsense_depth_scale", lambda *_: .001)
    monkeypatch.setattr(camera_inspect, "_start_realsense_pipeline", lambda *_: (pipeline, profile, []))
    report = camera_inspect._inspect_realsense(rs, CameraIdentity("realsense", "new", "D435"))
    assert report.depth_frame_received == frame_arrives
    assert stopped == [True]
