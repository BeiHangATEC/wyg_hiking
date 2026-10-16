"""Opt-in deployment conditioning for the GA4B normal policy."""

from dataclasses import dataclass, replace
import math
import time

import numpy as np

from bxi_example_py_elf3.policies import HumanoidGaitPolicyLiteIsaaclab


@dataclass(frozen=True)
class DeploymentSettings:
    command_limits: tuple = (0.5, 0.5, 1.5)
    command_deadband: tuple = (0.018, 0.025, 0.075)
    command_acceleration: tuple = (0.4, 0.3, 1.0)
    command_deceleration: tuple = (1.2, 1.0, 3.0)
    target_filter_tau_sec: float = 0.0
    observation_clip: float = 100.0
    action_guard: float = 100.0
    diagnostics_interval_sec: float = 1.0

    def __post_init__(self):
        for name in ("command_limits", "command_deadband", "command_acceleration", "command_deceleration"):
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise ValueError(f"{name} must contain three finite numbers")
            if np.any(values < 0) or (name != "command_deadband" and np.any(values == 0)):
                raise ValueError(f"invalid {name}: {values}")
        if np.any(np.asarray(self.command_deadband) >= self.command_limits):
            raise ValueError("command_deadband must be smaller than command_limits")
        for name in ("target_filter_tau_sec", "observation_clip", "action_guard", "diagnostics_interval_sec"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name != "target_filter_tau_sec" and value == 0):
                raise ValueError(f"invalid {name}: {value}")


class GA4BPolicy(HumanoidGaitPolicyLiteIsaaclab):
    def __init__(self, model, *, settings=None, **kwargs):
        self.settings = settings or DeploymentSettings()
        self._limits = np.asarray(self.settings.command_limits, dtype=np.float32)
        self._deadband = np.asarray(self.settings.command_deadband, dtype=np.float32)
        self._acceleration = np.asarray(self.settings.command_acceleration, dtype=np.float32)
        self._deceleration = np.asarray(self.settings.command_deceleration, dtype=np.float32)
        self._applied_command = np.zeros(3, dtype=np.float32)
        self._next_command = np.zeros(3, dtype=np.float32)
        self._normalized_quat = np.zeros(4, dtype=np.float64)
        super().__init__(model, **kwargs)
        if self.num_obs != 960 or self._action.size != 29:
            raise ValueError("GA4B deployment requires obs[1,960] -> actions[1,29]")
        self._filtered_target = self._target.copy()
        self._raw_target = self._target.copy()
        self._diagnostics = None
        self._clear_statistics()

    def _clear_statistics(self):
        self._statistics_elapsed = 0.0
        self._peaks = np.zeros(7, dtype=np.float64)
        self._last_step_at = None
        self._last_feedback_stamp = None
        self._feedback_changed_at = None
        self._error_joint = "-"

    @staticmethod
    def validate_frame(frame):
        for name, values in (("position", frame.joints.position), ("velocity", frame.joints.velocity),
                             ("quaternion", frame.quat_wxyz), ("omega", frame.angular_velocity),
                             ("command", frame.command)):
            if values is None or not np.isfinite(values).all():
                raise ValueError(f"invalid GA4B {name}")
        quat = np.asarray(frame.quat_wxyz)
        if quat.shape != (4,) or not 0.5 <= np.linalg.norm(quat) <= 1.5:
            raise ValueError("invalid GA4B quaternion norm")

    def _build_single_observation(self, qj, dqj, quat, omega, cmd_vel):
        norm = float(np.linalg.norm(quat))
        if not math.isfinite(norm) or not 0.5 <= norm <= 1.5:
            raise ValueError("invalid GA4B quaternion norm")
        np.divide(quat, norm, out=self._normalized_quat)
        return super()._build_single_observation(qj, dqj, self._normalized_quat, omega, cmd_vel)

    def _refresh_observation_buffer(self):
        super()._refresh_observation_buffer()
        np.clip(self._obs, -self.settings.observation_clip, self.settings.observation_clip, out=self._obs)

    def _update_observation(self, *args, **kwargs):
        super()._update_observation(*args, **kwargs)
        np.clip(self._obs, -self.settings.observation_clip, self.settings.observation_clip, out=self._obs)

    def reset(self, frame):
        self.validate_frame(frame)
        self._applied_command.fill(0.0)
        self._next_command.fill(0.0)
        super().reset(replace(frame, command=self._applied_command))
        np.copyto(self._filtered_target, self._target)
        self._diagnostics = None
        self._clear_statistics()

    def step(self, frame, dt, *, advance=True):
        self.validate_frame(frame)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("GA4B dt must be positive and finite")
        started = time.monotonic()
        np.clip(frame.command, -self._limits, self._limits, out=self._next_command)
        self._next_command[np.abs(self._next_command) < self._deadband] = 0.0
        reversing = self._next_command * self._applied_command < 0
        self._next_command[reversing] = 0.0
        delta = self._next_command - self._applied_command
        slowing = np.abs(self._next_command) < np.abs(self._applied_command)
        rates = np.where(slowing, self._deceleration, self._acceleration)
        self._next_command[:] = self._applied_command + np.clip(delta, -rates * dt, rates * dt)
        self._next_command[np.abs(self._next_command) < 1e-6] = 0.0
        output = super().step(replace(frame, command=self._next_command), dt, advance=advance)
        if not np.isfinite(self._action).all() or not np.isfinite(self._target).all():
            raise ValueError("non-finite GA4B inference output")
        if np.max(np.abs(self._action)) > self.settings.action_guard:
            raise ValueError("GA4B raw action exceeds deployment guard")
        np.copyto(self._raw_target, self._target)
        tau = self.settings.target_filter_tau_sec
        alpha = 1.0 if tau == 0 else -math.expm1(-dt / tau)
        # Keep raw previous_action/history semantics; filtering is actuator-side.
        if tau == 0:
            np.copyto(self._target, self._raw_target)
        else:
            self._target[:] = self._filtered_target + alpha * (self._raw_target - self._filtered_target)
        if advance:
            joints = self.bind_joints(frame)
            error = np.abs(self._target - joints.position)
            index = int(np.argmax(error))
            if error[index] > self._peaks[3]:
                self._error_joint = joints.layout.names[index]
            period_ms = 0.0 if self._last_step_at is None else (started - self._last_step_at) * 1000
            if self._last_feedback_stamp != frame.timestamp_ns or self._feedback_changed_at is None:
                self._feedback_changed_at = started
            self._last_feedback_stamp = frame.timestamp_ns
            repeat_ms = (started - self._feedback_changed_at) * 1000 if frame.timestamp_ns else 0.0
            values = (np.max(np.abs(self._raw_target - self._filtered_target)),
                      np.max(np.abs(self._target - self._filtered_target)), np.max(np.abs(joints.velocity)),
                      error[index], np.max(np.abs(frame.angular_velocity)), period_ms, repeat_ms)
            np.maximum(self._peaks, values, out=self._peaks)
            self._last_step_at = started
            np.copyto(self._filtered_target, self._target)
            np.copyto(self._applied_command, self._next_command)
            self._statistics_elapsed += dt
            if self._statistics_elapsed >= self.settings.diagnostics_interval_sec:
                p = self._peaks
                self._diagnostics = (
                    f"GA4B cmd_raw={np.round(frame.command, 3).tolist()} "
                    f"cmd_applied={np.round(self._applied_command, 3).tolist()} "
                    f"peak_raw_target_delta={p[0]:.3f}rad peak_filtered_target_delta={p[1]:.3f}rad "
                    f"peak_dq={p[2]:.2f}rad/s peak_tracking_error={p[3]:.3f}rad({self._error_joint}) "
                    f"peak_omega={p[4]:.2f}rad/s peak_period={p[5]:.1f}ms "
                    f"peak_feedback_repeat={p[6]:.1f}ms step={(time.monotonic()-started)*1000:.2f}ms"
                )
                self._peaks.fill(0.0)
                self._statistics_elapsed = 0.0
        return output

    def take_diagnostics(self):
        message, self._diagnostics = self._diagnostics, None
        return message
