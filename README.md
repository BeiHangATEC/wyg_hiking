# wyg_hiking

DAGGER depth-policy deployment and manual camera recording for the BXI ELF3
robot, extracted from the working `bxi_wyg2` workspace on 2026-10-16.

This is a focused ROS 2 workspace based on
[bxi_rl_controller_ros2_example](https://github.com/bxirobotics/bxi_rl_controller_ros2_example).
It includes the controller framework and basic actions required by the depth
Mod, remote input, the depth-camera package with its existing SDK fallbacks,
recording, the browser viewer, and actual inference-input auditing. The unrelated
SONIC and forward-flip Mods, build products, recordings and runtime logs are not
included. The original workspace is not modified by this export.

## Current Deployment

| Item | Value |
| --- | --- |
| Model | `src/bxi_example_py_elf3/mods/com.bxi.normal_depth/assets/dagger_new_position.onnx` |
| Executing state | `com.bxi.normal_depth/normal_depth` |
| ONNX inputs | `obs_history [1,960]`, `depth_data [1,36,36]`, `h_in [1,1,256]` |
| ONNX outputs | `actions [1,29]`, `h_out [1,1,256]` |
| Policy camera | RealSense D435, **SDK serial `151222073444`** |
| USB/ASIC serial | `153123024178`, not the launch serial |
| Other camera | D435i SDK serial `254522074724`, not selected by the deployment launcher |
| ROS domain | `42` |
| Depth topic | `/hardware/head_depth_camera/depth/image_rect_raw` |
| Calibration topic | `/hardware/head_depth_camera/depth/camera_info` |
| Control period | `0.02 s` |
| Depth timeout | `0.5 s`, safety transition to zero torque |
| Forward speed | Maximum `0.4 m/s`; the policy clamps negative forward commands to zero |
| Lateral command scale | `0.5 m/s` |
| Yaw speed | `-1.05` to `+1.05 rad/s` |

CameraInfo is obtained from the selected device. It does not establish correct
camera mounting/extrinsics. Check [camera mounting](docs/DAGGER_CAMERA_MOUNTING.md)
and [deployment notes](docs/DAGGER_NEW_POSITION_DEPLOYMENT.md).

## Build

Baseline: Ubuntu 22.04, ROS 2 Humble, CPython 3.10, and the official BXI SDK
available at `/opt/bxi/bxi_ros2_pkg`. This repository does not distribute the
robot hardware SDK, machine-specific `/opt/bxi/robot_config.yaml`, ROS itself,
or training code. Existing third-party license notices remain with their files.
No new blanket license is asserted for upstream files or policy weights.

```bash
cd /home/bxi/wyg_hiking
source /opt/bxi/bxi_ros2_pkg/setup.bash
# On a new machine only, install/check the host dependencies as appropriate:
# bash deploy_environment.sh
bash build.sh
source install/setup.bash
```

Remote input uses the existing official-style PowerA/CRSF configuration. The
optional device setup files are `script/bxi-dev.rules` and
`script/bxi-battle-dragon-link`. Do not overwrite machine-wide configuration or
stop another user's services without checking the robot's current owner.

## Hardware And Actual Input Audit

**These commands start real robot hardware.** Arrange physical protection,
keep the emergency stop accessible, use a local operator, and keep the sticks
centered before starting. Do not run a second hardware controller or a second
driver for the selected camera. A foreground SSH session can block or terminate
on connection loss; use a reliable local terminal for hardware tests.

Terminal 1:

```bash
cd /home/bxi/wyg_hiking
sudo bash script/start_dagger_input_audit.sh
```

The helper checks an actual depth frame from the selected camera before starting
hardware. It starts in zero torque; it never automatically enters DAGGER.

| Button | Action |
| --- | --- |
| RB+B | PD_BRAKE |
| RT+X | DAGGER, from PD_BRAKE or normal |
| RB+X | GA4B 75000 normal, not the depth policy |

For deployment without actual-input sampling, use
`sudo bash script/start_dagger_hw_test.sh` instead, not concurrently. Both
helpers accept an optional **SDK serial** argument to override the camera.

The audit writes to `logs/dagger-input-audit/<session>/`. Inspect a session with:

```bash
python3 tools/diagnostics/dagger_inference_audit.py \
  --report 'logs/dagger-input-audit/<session>'
```

See [the audit guide](tools/diagnostics/DAGGER_INFERENCE_AUDIT.md) for the
difference between warmup, successful running inference and camera-only previews.

## Manual Recording

Terminal 2, while Terminal 1 remains running:

```bash
cd /home/bxi/wyg_hiking
sudo bash script/start_dagger_camera_recording.sh
```

`RECORDER READY` means the subscriber is ready. `RECORDING START` confirms that
frames have arrived and a new recording session has begun. Press Ctrl+C in
Terminal 2 and wait for it to finish saving; this does not stop the robot.
Recording is manual and does not depend on entering normal or DAGGER.

Output: `logs/dagger-recordings/<session>/`, including depth-preview AVI,
optional RGB AVI, timestamps, metadata and a lossless depth ROS bag. Default
recording is up to 10 FPS, with 60-second segments and a 2 GiB free-space reserve.
There is no default time limit. Never replay bags into a running robot controller.
See [the recording guide](tools/diagnostics/DAGGER_LOCAL_RECORDING.md).

For **camera-only** recording, start this in Terminal 1 instead of a controller:

```bash
cd /home/bxi/wyg_hiking
source /opt/bxi/bxi_ros2_pkg/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42 ROS_LOCALHOST_ONLY=0
ros2 launch bxi_depth_camera cameras.launch.py serial_no:=151222073444
```

Do not also run this camera-only launch alongside a hardware helper that opens
the same camera. The user and ROS environment of the recording subscriber must
be able to discover the camera publisher.

## Browser Viewer

With an existing camera feed, run:

```bash
sudo bash script/start_dagger_camera_viewer.sh
```

The viewer listens on the robot's loopback address, port `8766`. On a remote PC:

```bash
ssh -N -L 8766:127.0.0.1:8766 bxi@ROBOT_IP
```

Open `http://127.0.0.1:8766` locally. The camera preview is not proof of actual
controller input; point the viewer at a compatible audit session with
`--capture-root logs/dagger-input-audit/<session>` to inspect saved real inputs.

## Offline Verification

After building, use only offline tests (no robot launch):

```bash
source /opt/bxi/bxi_ros2_pkg/setup.bash
source install/setup.bash
python3 -m pytest -q \
  src/bxi_depth_camera/test/test_inspect.py \
  src/bxi_depth_camera/test/test_common.py \
  src/bxi_depth_camera/test/test_supervisor.py \
  tools/diagnostics/test_dagger_*.py \
  src/bxi_example_py_elf3/test/test_dagger_deployment.py
sha256sum -c MODEL_SHA256SUMS
```

Successful offline tests or a camera-only probe do not establish safe standing,
walking stability, correct extrinsics, or real-time execution on the robot.
