from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dagger_camera_input import depth_summary, load_pipeline, probe_model, sequence_summary, value_summary


def test_empty_summary_is_json_safe():
    assert value_summary(np.array([]))["nonfinite_fraction"] is None


def test_invalid_depth_not_counted_as_positive():
    result = depth_summary(np.array([0., -1., np.nan, np.inf, 1., 2.], dtype=np.float32))
    assert result["invalid_fraction"] == pytest.approx(4 / 6)
    assert result["positive_depth_m"]["p50"] == 1.5


def test_timestamp_and_content_counters_are_independent():
    result = sequence_summary([100_000_000, 100_000_000, 90_000_000], [0., .1, .2], ["a", "b", "b"])
    assert result["receipt_hz"] == 10.
    assert result["duplicate_stamps"] == 1
    assert result["backward_stamps"] == 1
    assert result["identical_consecutive_images"] == 1


def test_single_frame_does_not_report_a_rate():
    assert sequence_summary([1], [1.], ["a"]) == {"frames": 1}


def test_real_preprocessing_and_onnx_probe():
    from sensor_msgs.msg import CameraInfo, Image
    mod_root = Path(__file__).resolve().parents[2] / "src/bxi_example_py_elf3/mods/com.bxi.normal_depth"
    state, policy = load_pipeline(mod_root)
    state._bind_logger(type("Logger", (), {"warning": lambda self, message: None})())
    try:
        info = CameraInfo(width=848, height=480, k=[428., 0., 424., 0., 428., 245., 0., 0., 1.])
        state.camera_info_callback(info)
        values = np.tile(np.arange(480, dtype=np.uint16)[:, None] + 700, (1, 848))
        message = Image(width=848, height=480, encoding="16UC1", step=1696, data=values.tobytes())
        state.depth_image_callback(message)
        tensor = policy._preprocess_depth(state._depth_rotated)
        assert tensor.shape == (36, 36)
        assert tensor.dtype == np.float32
        assert np.ptp(tensor) > 0
        # The clockwise state rotation must be undone before the asymmetric row crop.
        assert np.all(np.diff(tensor[:, 18]) > 0)
        result = probe_model(policy, tensor)
        assert result["inputs"]["depth_data"]["shape"] == [1, 36, 36]
        assert result["inputs"]["obs_history"]["shape"] == [1, 960]
        assert result["inputs"]["h_in"]["shape"] == [1, 1, 256]
        assert result["outputs"]["actions"]["shape"] == [1, 29]
        assert result["outputs"]["h_out"]["shape"] == [1, 1, 256]
        assert all(value["nonfinite_fraction"] == 0. for value in result["outputs"].values())
    finally:
        policy.close()
