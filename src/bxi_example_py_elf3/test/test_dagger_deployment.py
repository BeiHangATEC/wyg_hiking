import importlib
from pathlib import Path
import sys
import time
import types

import numpy as np
import pytest
from sensor_msgs.msg import CameraInfo, Image
import yaml

from bxi_example_py_elf3.framework.inference import InferenceFrame
from bxi_example_py_elf3.framework.joints import JointStateView
from bxi_example_py_elf3.framework.mod_api import ResourceLoadContext
from bxi_example_py_elf3.policies.joints import ELF3_ISAAC_PARAMETERS as PARAMETERS


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MOD_ROOT = PACKAGE_ROOT / "mods" / "com.bxi.normal_depth"
PACKAGE = "_test_dagger_mod"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(MOD_ROOT)]
sys.modules.setdefault(PACKAGE, package)
depth = importlib.import_module(f"{PACKAGE}.depth")
state_module = importlib.import_module(f"{PACKAGE}.state")
projection = importlib.import_module(f"{PACKAGE}.depth_projection")
plugin = importlib.import_module(f"{PACKAGE}.plugin")


@pytest.fixture
def policy():
    result = depth.HumanoidGaitOriginCameraPolicyIsaaclab(
        str(MOD_ROOT / "assets/dagger_new_position.onnx"), backend="onnxruntime",
    )
    yield result
    result.close()


def sensor_frame():
    return InferenceFrame(
        joints=JointStateView(PARAMETERS.layout, PARAMETERS.default_position.copy(), np.zeros(29, dtype=np.float32)),
        quat_wxyz=np.array([1., 0., 0., 0.]), angular_velocity=np.zeros(3),
        command=np.zeros(3, dtype=np.float32), depth=np.full((36, 48), 1.5, dtype=np.float32),
        depth_frame_id=1,
    )


def make_state(policy):
    state = state_module.NormalDepthState(
        "com.bxi.normal_depth/normal_depth", 1, types.SimpleNamespace(get=lambda: policy),
        mode="origin_camera", camera_name="head_depth_camera",
        depth_image_topic="", camera_info_topic="", depth_uint16_scale=.001,
        depth_timeout_sec=.5,
        projection=projection.ProjectionSpec(36, 48, 45.2, 58.0616969, .2, 3.),
    )
    state._bind_logger(types.SimpleNamespace(
        info=lambda msg: None, warning=lambda msg: None, error=lambda msg: None,
    ))
    return state


def test_recurrent_contract_and_preview_reset(policy):
    frame = sensor_frame()
    policy.reset(frame)
    assert {name: value.shape for name, value in policy._policy_inputs.items()} == {
        "obs_history": (1, 960), "depth_data": (1, 36, 36), "h_in": (1, 1, 256),
    }
    first = policy.step(frame, .02, advance=False).joints.position.copy()
    np.testing.assert_array_equal(policy._recurrent_state, np.zeros((1, 1, 256)))
    np.testing.assert_array_equal(policy._previous_action, np.zeros(29))
    second = policy.step(frame, .02, advance=False).joints.position.copy()
    np.testing.assert_array_equal(first, second)
    policy.step(frame, .02)
    assert np.linalg.norm(policy._recurrent_state) > 0
    policy.reset(frame)
    np.testing.assert_array_equal(policy._recurrent_state, np.zeros((1, 1, 256)))


def test_real_model_finite_with_repeated_depth_frames(policy):
    frame = sensor_frame()
    policy.reset(frame)
    for tick in range(100):
        frame.depth_frame_id = tick // 3 + 1
        frame.command[0] = .3 if 20 <= tick < 60 else 0
        output = policy.step(frame, .02)
        assert np.isfinite(output.joints.position).all()
        assert np.isfinite(policy._recurrent_state).all()
    np.testing.assert_array_equal(output.joints.kp, PARAMETERS.kp)
    np.testing.assert_array_equal(output.joints.kd, PARAMETERS.kd)


def test_projection_rotation_and_normalization(policy):
    state = make_state(policy)
    info = CameraInfo(width=640, height=480, k=[400., 0., 320., 0., 400., 240., 0., 0., 1.])
    state.camera_info_callback(info)
    values = np.full((480, 640), 1500, dtype=np.uint16)
    image = Image(width=640, height=480, encoding="16UC1", step=1280, data=values.tobytes())
    state.depth_image_callback(image)
    assert state.is_available(None)
    assert state._depth_rotated.shape == (36, 48)
    model_depth = policy._preprocess_depth(state._depth_rotated)
    assert model_depth.shape == (36, 36)
    np.testing.assert_allclose(model_depth, (1.5 - .2) / 2.8 - .5, atol=1e-6)


def test_camera_missing_or_stale_denies_entry(policy):
    state = make_state(policy)
    assert not state.is_available(None)
    state._camera_info = CameraInfo(width=1, height=1)
    state._depth_rotated = np.ones((36, 48), dtype=np.float32)
    state._last_depth_time = time.monotonic() - 1
    assert not state.is_available(None)
    assert state._is_depth_timed_out()
    state._last_depth_time = time.monotonic()
    frame = sensor_frame()
    frame.quat_wxyz[:] = 0
    assert not state.is_available(types.SimpleNamespace(inference_frame=frame))


@pytest.mark.parametrize("bad_source", ["depth", "quaternion"])
def test_nonfinite_input_rejected(policy, bad_source):
    frame = sensor_frame()
    if bad_source == "depth":
        frame.depth[0, 0] = np.nan
    else:
        frame.quat_wxyz[0] = np.nan
    with pytest.raises(ValueError):
        policy.step(frame, .02)


def test_depth_timeout_uses_zero_torque_not_automatic_ga4b(policy):
    state = make_state(policy)
    state._depth_enter_time = time.monotonic() - 1
    calls, frames = [], []
    frame = sensor_frame()
    ctx = types.SimpleNamespace(
        inference_frame=frame, current_quat_xyzw=np.array([0., 0., 0., 1.]),
        is_orientation_unsafe=lambda quat: False, robot_layout=frame.joints.layout,
        request_state=lambda *args, **kwargs: calls.append((args, kwargs)),
        set_motor_target=lambda value: frames.append(value),
    )
    state.on_update(ctx, .02)
    assert calls[0][0] == ("com.bxi.basic_actions/zero_torque",)
    assert calls[0][1]["force"]
    np.testing.assert_array_equal(frames[-1].kp, np.zeros(29))
    np.testing.assert_array_equal(frames[-1].kd, np.zeros(29))


def test_model_path_button_and_emergency_routes():
    context = ResourceLoadContext("com.bxi.normal_depth", MOD_ROOT)
    policy = plugin._load_origin_policy(context)
    try:
        assert policy.recurrent_input_name == "h_in"
        assert policy.recurrent_output_name == "h_out"
    finally:
        policy.close()
    manifest = yaml.safe_load((MOD_ROOT / "mod.yaml").read_text())
    assert manifest["events"]["activate"] == {"slot": "btn_10", "value": 7}
    routes = manifest["routes"]
    assert any(item["from"] == "com.bxi.basic_actions/pd_brake" and item["to"] == "normal_depth" for item in routes)
    assert any(item["from"] == "normal_depth" and item["to"] == "com.bxi.basic_actions/pd_brake" for item in routes)
    assert any(item["from"] == "normal_depth" and item["to"] == "com.bxi.basic_actions/zero_torque" for item in routes)


def test_complete_mod_loader_and_state_graph(policy):
    from bxi_example_py_elf3.framework.runtime.mod_loader import load_mod_runtime
    from bxi_example_py_elf3.framework.runtime.state_builder import build_robot_states
    from bxi_example_py_elf3.framework.runtime.state_machine import RobotStateMachine

    runtime = load_mod_runtime(
        yaml.safe_load((PACKAGE_ROOT / "config/elf3_state_machine.yaml").read_text()),
        built_in_root=PACKAGE_ROOT / "mods",
    )
    try:
        assert not runtime.unavailable_mods
        states = build_robot_states(runtime.config, runtime.state_factories)
        logger = types.SimpleNamespace(info=lambda msg: None, warning=lambda msg: None, error=lambda msg: None)
        RobotStateMachine(types.SimpleNamespace(), runtime.config, states, logger=logger, enter_initial=False)
        assert "com.bxi.normal_depth/normal_depth" in states
        dagger = states["com.bxi.normal_depth/normal_depth"]
        assert dagger.speed_profile_name == "com.bxi.normal_depth/dagger"
        profiles = runtime.config["speed_profiles"]
        assert profiles["com.bxi.basic_actions/normal"]["vx_max"] == .5
        ctx = types.SimpleNamespace(
            speed_profiles=profiles,
            current_raw_cmd_vel=np.array([0., 1., -1.], dtype=np.float32),
            current_cmd_vel=np.zeros(3, dtype=np.float32),
        )
        frame = sensor_frame()
        for raw, expected in ((-10., -.4), (-1., -.4), (-.5, -.2),
                              (0., 0.), (.5, .2), (1., .4), (10., .4)):
            ctx.current_raw_cmd_vel[0] = raw
            frame.command[:] = dagger.get_cmd_vel(ctx)
            np.testing.assert_allclose(frame.command, [expected, .5, -1.05])
            policy.step(frame, .02, advance=False)
            np.testing.assert_allclose(policy.cmd, [max(expected, 0.), .5, -1.05])
        ctx.current_raw_cmd_vel[:2] = 0.
        for raw, expected in ((-10., -1.05), (-1., -1.05), (-.5, -.525),
                              (0., 0.), (.5, .525), (1., 1.05), (10., 1.05)):
            ctx.current_raw_cmd_vel[2] = raw
            frame.command[:] = dagger.get_cmd_vel(ctx)
            np.testing.assert_allclose(frame.command, [0., 0., expected])
            policy.step(frame, .02, advance=False)
            np.testing.assert_allclose(policy.cmd, frame.command)
    finally:
        runtime.close()
