import importlib
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import yaml

from bxi_example_py_elf3.framework.inference.api import InferenceFrame
from bxi_example_py_elf3.framework.joints import JointStateView
from bxi_example_py_elf3.policies.joints import ELF3_ISAAC_PARAMETERS
from bxi_example_py_elf3.framework.mod_api import ResourceLoadContext


MOD_ROOT = Path(__file__).resolve().parents[1] / "mods" / "com.bxi.basic_actions"
PACKAGE = "_test_ga4b_mod"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(MOD_ROOT)]
sys.modules.setdefault(PACKAGE, package)
deployment = importlib.import_module(f"{PACKAGE}.ga4b_policy")


class Backend:
    def __init__(self):
        self.action = np.full((1, 29), 0.4, dtype=np.float32)
        self.obs = None

    def input_shape(self, name):
        return (1, 960)

    def output_shape(self, name):
        return (1, 29)

    def run(self, inputs):
        self.obs = inputs["obs"].copy()
        return {"actions": self.action}


def make_policy(settings=None):
    backend = Backend()
    runtime = types.SimpleNamespace(open_backend=lambda *args, **kwargs: backend)
    policy = deployment.GA4BPolicy("unused.onnx", runtime=runtime, settings=settings)
    p = ELF3_ISAAC_PARAMETERS
    frame = InferenceFrame(
        joints=JointStateView(p.layout, p.default_position.copy(), np.zeros(29, dtype=np.float32)),
        quat_wxyz=np.asarray([1., 0., 0., 0.]), angular_velocity=np.zeros(3),
        command=np.zeros(3, dtype=np.float32), timestamp_ns=1,
    )
    policy.reset(frame)
    return policy, backend, frame


def test_installed_settings_preserve_zero_speed_targets_and_gains():
    settings = deployment.DeploymentSettings(**yaml.safe_load((MOD_ROOT / "ga4b_deployment.yaml").read_text()))
    assert settings.target_filter_tau_sec == 0
    assert settings.command_limits[0] == 0.5
    policy, backend, frame = make_policy(settings)
    output = policy.step(frame, 0.02)
    p = ELF3_ISAAC_PARAMETERS
    np.testing.assert_array_equal(output.joints.kp, p.kp)
    np.testing.assert_array_equal(output.joints.kd, p.kd)
    np.testing.assert_allclose(output.joints.position, p.default_position + backend.action[0] * p.action_scale)
    np.testing.assert_array_equal(policy._applied_command, np.zeros(3))


def test_real_resource_context_loads_configuration_outside_assets():
    plugin = importlib.import_module(f"{PACKAGE}.plugin")
    policy = plugin._load_normal_policy(ResourceLoadContext("com.bxi.basic_actions", MOD_ROOT))
    try:
        assert policy.settings.command_limits[0] == 0.5
    finally:
        policy.close()


def test_deadband_saturations_acceleration_and_center_return():
    policy, backend, frame = make_policy()
    frame.command[:] = [100., -100., 100.]
    policy.step(frame, 0.02)
    np.testing.assert_allclose(policy._applied_command, [0.008, -0.006, 0.02])
    for _ in range(200):
        policy.step(frame, 0.02)
    np.testing.assert_allclose(policy._applied_command, [0.5, -0.5, 1.5])
    frame.command[:] = [0.01, -0.01, 0.01]
    for _ in range(26):
        policy.step(frame, 0.02)
    np.testing.assert_array_equal(policy._applied_command, [0., 0., 0.])
    np.testing.assert_array_equal(backend.obs.reshape(10, 96)[-1, 6:9], [0., 0., 0.])


def test_preview_does_not_advance_conditioning_or_raw_action_history():
    settings = deployment.DeploymentSettings(target_filter_tau_sec=0.01)
    policy, backend, frame = make_policy(settings)
    frame.command[:] = [.6, .5, 1.5]
    command = policy._applied_command.copy()
    target = policy._filtered_target.copy()
    previous = policy._previous_action.copy()
    first = policy.step(frame, .02, advance=False).joints.position.copy()
    second = policy.step(frame, .02, advance=False).joints.position.copy()
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(command, policy._applied_command)
    np.testing.assert_array_equal(target, policy._filtered_target)
    np.testing.assert_array_equal(previous, policy._previous_action)
    policy.step(frame, .02)
    np.testing.assert_array_equal(policy._previous_action, backend.action[0])
    alpha = 1 - np.exp(-2)
    np.testing.assert_allclose(policy._filtered_target, target + alpha * .4 * ELF3_ISAAC_PARAMETERS.action_scale, atol=1e-7)


def test_reversal_brakes_to_zero_before_accelerating_in_opposite_direction():
    policy, backend, frame = make_policy()
    frame.command[0] = .6
    for _ in range(100):
        policy.step(frame, .02)
    frame.command[0] = -.6
    for _ in range(21):
        policy.step(frame, .02)
        assert policy._applied_command[0] >= 0
    assert policy._applied_command[0] == 0
    policy.step(frame, .02)
    assert policy._applied_command[0] == pytest.approx(-.008)


def test_quaternion_normalization_and_observation_clipping():
    policy, backend, frame = make_policy()
    frame.quat_wxyz[:] = [1.1, 0., 0., 0.]
    frame.joints.velocity[:] = 101
    policy.step(frame, .02)
    latest = backend.obs.reshape(10, 96)[-1]
    np.testing.assert_array_equal(latest[3:6], [0., 0., -1.])
    assert np.max(np.abs(backend.obs)) <= 100


@pytest.mark.parametrize("field", ["command", "quat_wxyz", "angular_velocity"])
def test_nonfinite_input_is_rejected_before_inference(field):
    policy, backend, frame = make_policy()
    getattr(frame, field)[0] = np.nan
    with pytest.raises(ValueError):
        policy.step(frame, .02)
    assert backend.obs is None


@pytest.mark.parametrize("value", [np.nan, np.inf, 101.])
def test_invalid_output_is_not_returned_to_motor_state(value):
    policy, backend, frame = make_policy()
    backend.action[0, 0] = value
    with pytest.raises(ValueError):
        policy.step(frame, .02)


def test_diagnostics_expose_zero_command_and_tracking_error():
    policy, backend, frame = make_policy()
    for _ in range(51):
        policy.step(frame, .02)
    message = policy.take_diagnostics()
    assert "cmd_applied=[0.0, 0.0, 0.0]" in message
    assert "peak_tracking_error=" in message
    assert "peak_feedback_repeat=" in message
    assert policy.take_diagnostics() is None


def test_inference_failure_requests_zero_torque_and_clears_current_output():
    module = importlib.import_module(f"{PACKAGE}.normal_state")
    policy, backend, frame = make_policy()
    backend.action[:] = np.nan
    state = module.NormalState("normal", 1, types.SimpleNamespace(get=lambda: policy))
    state._bind_logger(types.SimpleNamespace(error=lambda msg: None))
    requests, frames = [], []
    ctx = types.SimpleNamespace(
        inference_frame=frame, current_quat_xyzw=[0, 0, 0, 1],
        current_cmd_vel=np.zeros(3), is_orientation_unsafe=lambda quat: False,
        request_state=lambda *args, **kwargs: requests.append((args, kwargs)),
        set_motor_target=lambda motor: frames.append(motor),
        robot_layout=frame.joints.layout,
    )
    state.on_update(ctx, .02)
    assert requests[0][0] == ("com.bxi.basic_actions/zero_torque",)
    assert len(frames) == 1
    assert np.isfinite(frames[0].qpos).all()
    np.testing.assert_array_equal(frames[0].kp, np.zeros(29))
    np.testing.assert_array_equal(frames[0].kd, np.zeros(29))
