import io
import json
from pathlib import Path
import sys
import threading
import time
import types
from urllib.error import HTTPError
from urllib.request import urlopen

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dagger_camera_viewer import BackendRecordReader, ViewerCache, color_depth, committed_record_line, make_handler
from http.server import ThreadingHTTPServer


CONTRACT = {"obs_history": ([1, 960], "float32"), "depth_data": ([1, 36, 36], "float32"),
            "h_in": ([1, 1, 256], "float32")}


def recorded_sample(directory):
    directory.mkdir()
    metadata = {"model": "dagger_new_position", "model_sha256": "test-hash", "capture_backend": "onnxruntime",
                "inputs": [{"name": name, "shape": shape, "dtype": dtype} for name, (shape, dtype) in CONTRACT.items()]}
    (directory / "capture.json").write_text(json.dumps(metadata))
    paths = []
    for index, (name, (shape, dtype)) in enumerate(CONTRACT.items()):
        path = directory / f"000001_{index:02d}_{name}.npy"
        np.save(path, np.zeros(shape, dtype=dtype))
        paths.append(path)
    (directory / "dataset.txt").write_text(" ".join(map(str, paths)) + "\n")
    return BackendRecordReader(directory, "test-hash", CONTRACT)


def test_partial_last_record_is_not_displayed(tmp_path):
    path = tmp_path / "dataset.txt"
    path.write_text("complete sample\npartial sample")
    assert committed_record_line(path) == "complete sample"


def test_reads_actual_sample_and_rejects_another_model(tmp_path):
    reader = recorded_sample(tmp_path / "model")
    reader.poll()
    assert reader.frame["sample"] == "000001"
    assert reader.frame["depth"].shape == (36, 36)
    metadata_path = reader.directory / "capture.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["model_sha256"] = "another-model"
    metadata_path.write_text(json.dumps(metadata))
    reader.poll()
    assert reader.frame is None
    assert "hash" in reader.error


def test_rejects_shape_mismatch(tmp_path):
    reader = recorded_sample(tmp_path / "model")
    np.save(reader.directory / "000001_01_depth_data.npy", np.zeros((1, 48, 36), dtype=np.float32))
    reader.poll()
    assert reader.frame is None
    assert "contract" in reader.error


def test_empty_dataset_clears_previous_sample(tmp_path):
    reader = recorded_sample(tmp_path / "model")
    reader.poll()
    assert reader.frame is not None
    (reader.directory / "dataset.txt").write_text("")
    reader.poll()
    assert reader.frame is None
    assert reader.error is None


def test_malformed_metadata_rejected(tmp_path):
    reader = recorded_sample(tmp_path / "model")
    (reader.directory / "capture.json").write_text("[]")
    reader.poll()
    assert reader.frame is None
    assert "metadata" in reader.error


def test_record_paths_cannot_escape_directory(tmp_path):
    reader = recorded_sample(tmp_path / "model")
    dataset = reader.directory / "dataset.txt"
    original = dataset.read_text()
    outside = tmp_path / "outside.npy"
    np.save(outside, np.zeros((1, 960), dtype=np.float32))
    paths = original.split()
    paths[0] = str(outside)
    dataset.write_text(" ".join(paths) + "\n")
    reader.poll()
    assert reader.frame is None
    assert "outside" in reader.error


def test_invalid_depth_is_black_and_tensor_contrast_visible():
    values = np.array([[np.nan, -np.inf, 0.], [.2, 1.5, 3.]], dtype=np.float32)
    image = color_depth(values, .2, 3., values <= 0)
    np.testing.assert_array_equal(image[0], np.zeros((3, 3), dtype=np.uint8))
    assert not np.array_equal(image[1, 0], image[1, 2])


def make_cache(reader, received=None):
    tensor = np.linspace(-.3, .2, 1296, dtype=np.float32).reshape(36, 36)
    frame = {"seq": 9, "stamp_ns": time.time_ns(), "received": time.monotonic() if received is None else received,
             "raw": np.ones((48, 64), dtype=np.float32), "rotated": np.ones((36, 48), dtype=np.float32),
             "tensor": tensor, "crop": (10, 0, 36, 48), "info": {"k": [400., 0., 32., 0., 400., 24., 0., 0., 1.]},
             "encoding": "16UC1", "invalid_percent": 0., "crop_invalid_percent": 0., "tensor_summary": {"min": -.3}}
    monitor = types.SimpleNamespace(state=types.SimpleNamespace(depth_timeout_sec=.5),
                                    policy=types.SimpleNamespace(depth_range=[.2, 3.]),
                                    snapshot=lambda: (frame, {"error": None, "fps": 30.}))
    return ViewerCache(monitor, reader, ["/depth", "/info"], "42")


def test_waiting_actual_is_not_a_preview_and_stale_camera_is_marked(tmp_path):
    reader = BackendRecordReader(tmp_path / "missing", "test-hash", CONTRACT)
    cache = make_cache(reader, received=time.monotonic() - 2)
    cache.refresh()
    api = json.loads(cache.response("/api/frame")[1])
    assert api["camera"]["state"] == "stale"
    assert api["actual"]["state"] == "waiting"
    assert cache.response("/api/image/actual.png") is None
    assert cache.response("/api/image/preview.png")[0] == "image/png"


def test_actual_input_and_snapshot_are_separate_from_preview(tmp_path):
    cache = make_cache(recorded_sample(tmp_path / "model"))
    cache.refresh()
    assert cache.api["actual"]["state"] == "live"
    actual = cv2.imdecode(np.frombuffer(cache.images["actual"], dtype=np.uint8), cv2.IMREAD_COLOR)
    assert actual.shape == (36, 36, 3)
    snapshot = np.load(io.BytesIO(cache.response("/api/snapshot.npz")[1]), allow_pickle=False)
    assert snapshot["tensor_preview"].shape == snapshot["backend_depth_data"].shape == (1, 36, 36)
    assert not np.array_equal(snapshot["tensor_preview"], snapshot["backend_depth_data"])


def test_backend_stale_and_nonfinite_status(tmp_path):
    reader = recorded_sample(tmp_path / "model")
    reader.poll()
    cache = make_cache(reader)
    reader.frame["updated_ns"] = time.time_ns() - 2_000_000_000
    cache.refresh()
    assert cache.api["actual"]["state"] == "stale"
    reader.frame["summary"]["h_in"]["nonfinite_fraction"] = 1.
    cache.refresh()
    assert cache.api["actual"]["state"] == "invalid"


def test_http_readonly_endpoints(tmp_path):
    cache = make_cache(BackendRecordReader(tmp_path / "missing", "test-hash", CONTRACT))
    cache.refresh()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(cache))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base) as response:
            assert "DAGGER" in response.read().decode()
        with urlopen(base + "/api/frame") as response:
            assert json.load(response)["domain"] == "42"
        with urlopen(base + "/api/snapshot.npz") as response:
            assert "attachment" in response.headers["Content-Disposition"]
            assert np.load(io.BytesIO(response.read()), allow_pickle=False)["tensor_preview"].shape == (1, 36, 36)
        with pytest.raises(HTTPError) as error:
            urlopen(base + "/../capture.json")
        assert error.value.code == 404
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
