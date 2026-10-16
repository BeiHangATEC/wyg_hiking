import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import types

import cv2
import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo, Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dagger_camera_recorder import RecorderWorker, StateInfo, color_image, parse_args


def test_recorded_hardware_entrypoint_only_delegates_to_original(tmp_path):
    script_root = Path(__file__).resolve().parents[2] / "script"
    temporary_scripts = tmp_path / "script"
    temporary_scripts.mkdir()
    wrapper = temporary_scripts / "start_dagger_recorded_hw_test.sh"
    shutil.copyfile(script_root / wrapper.name, wrapper)
    original = temporary_scripts / "start_dagger_hw_test.sh"
    original.write_text("#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\nexit 17\n")
    result = subprocess.run(["bash", str(wrapper), "254522074724"],
                            cwd=tmp_path, text=True, capture_output=True)
    assert result.returncode == 17
    assert result.stdout == "254522074724\n"
    assert result.stderr == ""


def state(name="com.bxi.basic_actions/normal", transition=False):
    return json.dumps({"current": {"name": name}, "in_transition": transition,
                       "cmd_vel": {"x": 0., "y": 0., "yaw": 0.}})


@pytest.mark.parametrize("payload", ["oops", "null", "[]", '{"current":{},"in_transition":false}',
                                     '{"current":{"name":"test"},"in_transition":"false"}'])
def test_bad_state_data_is_only_an_unavailable_annotation(payload):
    annotations = StateInfo()
    annotations.update(payload, 1)
    assert annotations.snapshot(1) is None


def test_state_annotations_expire_without_controlling_recording():
    annotations = StateInfo(timeout=2.)
    assert annotations.snapshot(0) is None
    annotations.update(state(transition=True), 1)
    assert annotations.snapshot(1)["in_transition"]
    assert annotations.snapshot(3.1) is None
    annotations.update(state("com.bxi.basic_actions/zero_torque"), 4)
    assert annotations.snapshot(4)["current"]["name"].endswith("/zero_torque")


def test_default_manual_recording_has_no_time_limit():
    args = parse_args([])
    assert args.max_session_sec == 0
    assert args.fps == 10
    assert parse_args(["--max-session-sec", "1200"]).max_session_sec == 1200


@pytest.mark.parametrize("arguments", [
    ["--max-session-sec", "-1"], ["--max-session-sec", "nan"],
    ["--max-session-sec", "inf"], ["--fps", "0"], ["--fps", "31"],
    ["--segment-sec", "0"], ["--min-free-gb", "0"], ["--state-timeout", "0"],
])
def test_invalid_recording_limits_are_rejected(arguments):
    with pytest.raises(SystemExit):
        parse_args(arguments)


def test_padded_rgb_is_decoded_without_swapping_or_reading_padding():
    image = Image(width=2, height=2, encoding="rgb8", step=8,
                  data=bytes([255, 0, 0, 0, 255, 0, 99, 99] * 2))
    decoded = color_image(image)
    np.testing.assert_array_equal(decoded[0], [[0, 0, 255], [0, 255, 0]])
    image.step = 1
    with pytest.raises(ValueError):
        color_image(image)


@pytest.fixture
def worker(tmp_path):
    args = types.SimpleNamespace(output=tmp_path, capture_root=tmp_path / "inputs", fps=10.,
                                 segment_sec=60., max_session_sec=0., min_free_gb=.001)
    image = Image(width=848, height=480, encoding="16UC1", step=1696,
                  data=np.full((480, 848), 1200, dtype=np.uint16).tobytes())
    image.header.frame_id = "camera_optical"
    image.header.stamp.sec = 1234
    info = CameraInfo(width=848, height=480, k=[428., 0., 424., 0., 428., 245., 0., 0., 1.])
    info.header = image.header
    monitor = types.SimpleNamespace(count=0, latest=None, on_info=lambda msg: None)

    def on_image(message):
        monitor.count += 1
        monitor.latest = {"seq": monitor.count, "raw": np.full((480, 848), 1.2, dtype=np.float32),
                          "crop": [246, 2, 356, 475], "tensor": np.zeros((36, 36), dtype=np.float32),
                          "stamp_ns": 1234000000000, "tensor_summary": {"min": 0., "max": 0.},
                          "invalid_percent": 0.}
    monitor.on_image = on_image
    monitor.snapshot = lambda: (monitor.latest, {"error": None})
    reader = types.SimpleNamespace(frame=None, error=None, poll=lambda: None)
    topics = {"depth": "/test/depth", "info": "/test/info", "color": "/test/color",
              "color_info": "/test/color_info", "state": "/test/state"}
    logs = []
    recorder = RecorderWorker(args, StateInfo(), monitor, reader, {"sha256": "test"}, topics, log=logs.append)
    yield recorder, image, info, logs
    recorder._finish("test_cleanup")


def test_waiting_for_camera_does_not_create_empty_recordings(worker):
    recorder, image, info, logs = worker
    recorder.tick()
    assert recorder.session is None
    assert not list(recorder.args.output.iterdir())


def test_video_and_lossless_rosbag_record_without_state_and_close_on_stop(worker):
    import rosbag2_py
    from rclpy.serialization import deserialize_message

    recorder, image, info, logs = worker
    now = time.monotonic()
    color = Image(width=64, height=48, encoding="bgr8", step=192,
                  data=np.full((48, 64, 3), 150, dtype=np.uint8).tobytes())
    recorder.submit(image, info, color, now=now)
    recorder.tick(now)
    directory = recorder.session.directory
    recorder.stop.set()
    recorder.run()
    assert recorder.session is None
    metadata = json.loads((directory / "metadata.json").read_text())
    assert metadata["status"] == "closed" and metadata["frames"] == 1
    assert metadata["recording_mode"] == "manual"
    assert metadata["reason"] == "recorder_shutdown"
    segment = directory / "segment-0001"
    for name, size in [("depth_input.avi", (1280, 480)), ("color.avi", (64, 48))]:
        video = cv2.VideoCapture(str(segment / name))
        success, frame = video.read()
        video.release()
        assert success and frame.shape[:2] == size[::-1]
        assert np.std(frame) > 0 if name.startswith("depth") else True
    bag = rosbag2_py.SequentialReader()
    bag.open(rosbag2_py.StorageOptions(uri=str(segment / "depth_bag"), storage_id="sqlite3"),
             rosbag2_py.ConverterOptions("cdr", "cdr"))
    found_depth = False
    while bag.has_next():
        topic, data, timestamp = bag.read_next()
        if topic == "/test/depth":
            restored = deserialize_message(data, Image)
            assert bytes(restored.data) == bytes(image.data)
            assert restored.header.stamp.sec == 1234
            found_depth = True
        assert topic != "/test/state"
    assert found_depth
    record = json.loads((segment / "frames.jsonl").read_text().strip())
    assert record["preview_valid"] and record["backend_sample"] is None
    assert record["state"] is None


def test_policy_changes_and_missing_status_do_not_stop_or_split_recording(worker):
    recorder, image, info, logs = worker
    now = time.monotonic()
    directory = None
    for index, (name, transition) in enumerate([
        ("com.bxi.basic_actions/zero_torque", False),
        ("com.bxi.basic_actions/normal", True),
        ("com.bxi.basic_actions/normal", False),
        ("com.bxi.normal_depth/normal_depth", False),
        ("com.bxi.basic_actions/pd_brake", False),
    ]):
        received = now + index * .2
        recorder.state_info.update(state(name, transition), received)
        recorder.submit(image, info, now=received)
        recorder.tick(received)
        directory = directory or recorder.session.directory
        assert recorder.session.directory == directory
    recorder.submit(image, info, now=now + 10)
    recorder.tick(now + 10)
    assert recorder.session.directory == directory
    assert recorder.session.count == 6
    recorder._finish("operator_stop")
    records = [json.loads(line) for line in (directory / "segment-0001/frames.jsonl").read_text().splitlines()]
    assert records[1]["state"]["in_transition"]
    assert records[-1]["state"] is None


def test_segments_are_finalized_at_rotation(worker):
    recorder, image, info, logs = worker
    recorder.args.segment_sec = .15
    now = time.monotonic()
    recorder.submit(image, info, now=now)
    recorder.tick(now)
    recorder.submit(image, info, now=now + .2)
    recorder.tick(now + .2)
    assert len(recorder.session.metadata["segments"]) == 2
    assert (recorder.session.directory / "segment-0001/depth_bag/metadata.yaml").is_file()


def test_queue_is_bounded_and_never_blocks_receiver(worker):
    recorder, image, info, logs = worker
    now = time.monotonic()
    for index in range(20):
        recorder.submit(image, info, now=now + index * .1)
    assert recorder.queue.qsize() == 8
    assert recorder.dropped > 0


def test_sampling_phase_does_not_accumulate_camera_jitter(worker):
    recorder, image, info, logs = worker
    now = time.monotonic()
    for offset in (0., .11, .205):
        recorder.submit(image, info, now=now + offset)
    assert recorder.queue.qsize() == 3
    assert recorder.next_selected == pytest.approx(now + .3)


def test_default_duration_does_not_stop_after_ten_minutes(worker):
    recorder, image, info, logs = worker
    now = time.monotonic()
    recorder.submit(image, info, now=now)
    recorder.tick(now)
    directory = recorder.session.directory
    recorder.submit(image, info, now=now + 3600)
    recorder.tick(now + 3600)
    assert recorder.session.directory == directory
    assert recorder.session.count == 2
    assert not recorder.blocked


def test_optional_duration_limit_closes_and_does_not_reopen(worker):
    recorder, image, info, logs = worker
    recorder.args.max_session_sec = .2
    now = time.monotonic()
    recorder.submit(image, info, now=now)
    recorder.tick(now)
    directory = recorder.session.directory
    recorder.tick(now + .3)
    assert recorder.session is None and recorder.blocked
    recorder.submit(image, info, now=now + .4)
    recorder.tick(now + .4)
    assert recorder.session is None
    assert json.loads((directory / "metadata.json").read_text())["reason"] == "session_time_limit"


def test_disk_reserve_failure_does_not_open_or_crash_worker(worker, monkeypatch):
    recorder, image, info, logs = worker
    recorder.submit(image, info)
    monkeypatch.setattr("dagger_camera_recorder.shutil.disk_usage", lambda path: types.SimpleNamespace(free=0))
    recorder.tick()
    assert recorder.session is None
    assert recorder.blocked
    assert "controller is unaffected" in logs[-1]


def test_manual_restart_creates_a_new_directory_without_overwriting(worker):
    recorder, image, info, logs = worker
    recorder.submit(image, info)
    recorder.tick()
    original = recorder.session.directory
    recorder._finish("operator_stop")
    second = RecorderWorker(recorder.args, StateInfo(), recorder.monitor, recorder.reader,
                            recorder.identity, recorder.topics, log=logs.append)
    try:
        second.submit(image, info)
        second.tick()
        assert second.session.directory != original
        assert json.loads((original / "metadata.json").read_text())["status"] == "closed"
    finally:
        second._finish("operator_stop")
