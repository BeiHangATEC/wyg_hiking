# dagger_new_position Hardware Deployment Preparation

## Offline-Only Scope

The robot is in use by another operator. This work prepares source, installed
packages, launch arguments and offline tests only. Do not start the helper
or change a camera/controller process until the robot is available and the
test is coordinated. No live controller or camera was started or stopped.

Camera mounting references and the calibration/adaptation plan are in
[DAGGER_CAMERA_MOUNTING.md](DAGGER_CAMERA_MOUNTING.md). The head pose and a
commented body-camera "new position" coexist in this workspace; neither is
confirmed as this ONNX's training pose. Physical mounting remains pending.

## Installed Integration

Archive: `com.bxi.normal_depth(1).zip`. Source files and all three ONNX assets
were extracted into `src/bxi_example_py_elf3/mods/com.bxi.normal_depth`.
Old SDK logs, bytecode caches and RKNN build lock files were not installed.
The original archive and GA4B 75000 assets remain unchanged.

The active origin-camera loader now selects `dagger_new_position.onnx`, not
the archive's default `dagger2.onnx`. This is a separate depth state, not a
replacement for the 75000 normal policy.

Model contract, batch size 1:

```text
obs_history: [1, 960] float32
depth_data:  [1, 36, 36] float32
h_in:        [1, 1, 256] float32
actions:     [1, 29] float32
h_out:       [1, 1, 256] float32
```

SHA256: `21031ed16330e59bb3cc6c7ecdafe8235987c94df4c18c06b91091592f00d3ab`.
The ONNX has no custom metadata proving its training configuration.

The supplied policy's observation layout is angular velocity, projected
gravity, command, joint-position error, joint velocity and previous action,
with 10 oldest-to-newest frames. KP/KD, poses and action scales still use
the framework's ELF3_ISAAC_PARAMETERS. Hidden state is zeroed on entry and
updated only on advancing inference calls. Entry performs a non-advancing
warmup using the actual depth frame.

## Camera and Command Assumptions

The supplied origin-camera preprocessing was retained:

- CameraInfo-based center crop with horizontal/vertical FOV 45.2/58.0616969 deg.
- Resize to 36x48 before clockwise rotation; state cache is 36x48 after rotation.
- Policy restores orientation, applies optional OpenCV Gaussian blur, clips
  depth to 0.2-3.0 m, normalizes to [-0.5,0.5], and crops rows [2,38).
- Resulting model depth tensor is 36x36. Depth update interval is 0.05 s;
  on the 0.02 s control grid its cached-image replacement is quantized.

These are archive defaults, NOT a verified camera extrinsic configuration
for a model named "new_position". Before motor testing, confirm the training
camera's position, orientation, height, FOV and preprocessing against the
physical mount. Camera topic names do not establish physical equivalence.
Provide matching env.yaml/agent.yaml or the original deployment config.

On 2026-10-10 the user requested a lower forward speed. DAGGER now uses its
own speed profile: forward stick scale 0.4, bounds +/-0.4 m/s, lateral
scale 0.5 m/s and yaw scale 1.5 rad/s. The supplied depth policy additionally
clips forward commands to [0,0.8], so its effective forward range is
0-0.4 m/s: negative forward stick commands request zero, not reverse.
GA4B 75000 retains its separate +/-0.5 m/s forward profile. The DAGGER
half-forward stick command is 0.2 m/s and the full-forward command is 0.4.
The GA4B command ramp/filter is not automatically applied to this separate
policy. Zero command is not proof of learned standstill behavior.

## Controls and Failure Behavior

- RB+A: zero torque.
- RB+B: PD_BRAKE; also returns from the depth state.
- RT+X: enter dagger_new_position from PD_BRAKE or GA4B normal.
- RB+X: GA4B 75000; from the depth state this returns to GA4B.

Use RT (right trigger), not RB (right shoulder). Other shoulder/trigger
buttons must be released for the RT+X mapping.

Entry requires fresh depth, valid CameraInfo and finite valid proprioception.
Depth entry uses a 0.02 s hold followed by a 0.3 s gain ramp. While running,
depth receipt timeout (>0.5 s), unsafe orientation or inference failure
requests immediate zero torque and writes zero gains for the current cycle.
There is no automatic fallback into a different walking policy on camera
loss. This changes the uploaded Mod's original timeout behavior.

This is not a complete hardware safety system. In particular, independent
IMU/joint freshness watchdogs, calibrated mechanical target bounds, physical
latency, actuator dynamics and training-parameter equivalence remain to be
verified. Keep the safety tether and emergency stop ready; zero torque can
cause a supported robot to fall.

## Current Camera Binding (2026-10-15)

The policy camera is now the added RealSense D435, SDK serial `151222073444`
(USB/ASIC serial `153123024178`). The hardware launch and startup helper default
to this SDK serial. The original D435i, SDK serial `254522074724` (USB/ASIC serial
`260243062575`), is reserved for its other owner and is not opened automatically.
USB/ASIC serials are not interchangeable with `camera_info.serial_number`.

Only the selected device is mapped to `head_depth_camera` in ROS domain 42;
the model, depth preprocessing, buttons, gains and speed limits are unchanged.
CameraInfo comes from the selected device, so old-camera intrinsics are not
copied into the new feed. Changed mounting/extrinsics still need independent
verification. Keep other camera feeds on different topics or ROS domains.

The helper probes only the selected camera before hardware launch. A driver for
another physical camera no longer blocks startup; a missing/busy selected camera
fails the stream-open check. No process is killed and no lock file is created.

## Historical Camera Conflict (2026-10-10)

Inspection on 2026-10-10 found two RealSense USB devices and an existing
official camera process (PID 15092 at inspection time). It was configured
for ROS_DOMAIN_ID=22 and serial `254522074724`, under logical camera name
`head_depth_camera`. The new hardware test uses domain 42, so this existing
feed is not automatically usable. Its logical name does not prove that its
mount matches this model's training camera.

No camera or controller process was stopped, and no motor test was started.
Stop the owning camera launch deliberately before opening the same device
in domain 42, or arrange a verified compatible feed in domain 42 separately.
Do not launch a second driver on an occupied device. Do not guess which of
the two physical cameras belongs to the training view.

## Start Later, After Robot Availability and Confirmation

The helper defaults to the selected D435 and accepts an optional SDK serial:

```bash
cd /home/bxi/wyg_hiking
sudo bash script/start_dagger_hw_test.sh
```

To override the device, append its verified SDK serial, not its USB/ASIC serial.
The helper sources the installed SDK and this workspace, selects domain 42, enables
the camera and remote node, and starts in zero torque. Keep the stick
centered, verify RB+B, then press RT+X only with a clear, safe test area.
The camera and input programs must use the same ROS domain as the controller.

Each second, DAGGER log lines show effective commands, cached depth frame ID,
latest received depth age, and maximum absolute policy target angle. These
are not recordings of final hardware torque or proof of depth-image content
freshness. Record the full console output for a controlled retest.

## Verification and Known Test Gaps

- bxi_example_py_elf3 and bxi_depth_camera built successfully.
- 23 focused tests passed: new recurrent inference, reset/preview semantics,
  projection/normalization, depth timeout behavior, real ResourceLoadContext,
  complete Mod discovery/state graph and the previous GA4B regressions.
- Four Mods loaded; the full graph validates with 14 states/41 transitions.
- Real-model 200-step synthetic inference stayed finite. Measured step
  p99 approximately 0.602 ms and max 0.621 ms, without real hardware load.
- A pre-existing camera default-config test fails because default.yaml uses
  explicit 848x480x30 depth and 1920x1080x30 color profiles, while its test
  expects auto profiles. Those defaults were not changed.
- The repository also retains a test for an absent Pico Mod; a complete
  all-package regression is therefore not claimed.
- The previous GA4B startup config lookup was corrected to use mod_root:
  ResourceLoadContext.asset() only accepts files inside assets/. A new test
  uses the real context to catch this integration error.

Software preparation is complete. Camera-mount confirmation, matching
training configuration, closed-loop simulation and physical validation are
still outstanding. Do not treat offline finite outputs as safe locomotion.
