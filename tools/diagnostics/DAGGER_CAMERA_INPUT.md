# DAGGER Camera Input Audit

Offline, manual local video/raw-depth recording and the two-terminal hardware
startup/stop workflow are documented in [DAGGER_LOCAL_RECORDING.md](DAGGER_LOCAL_RECORDING.md).

For combined executing-state/model verification and actual backend input capture,
use [DAGGER_INFERENCE_AUDIT.md](DAGGER_INFERENCE_AUDIT.md). It wraps a fresh hardware
launch and is not a standalone camera subscriber or automatic policy switch.

This is a standalone subscriber. It does not start camera drivers, create motor
publishers, call hardware services, enter a walking state or change parameters.
Start the intended camera separately and use the same ROS domain as its driver.
No package rebuild or new dependencies are required in the current workspace.

```bash
cd /home/bxi/wyg_hiking
source /opt/bxi/bxi_ros2_pkg/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
python3 tools/diagnostics/dagger_camera_input.py --seconds 12 \
  --output "logs/camera-input-$(date +%Y%m%d-%H%M%S)"
```

The script reads the INSTALLED Mod manifest and uses its policy loader, state
decoder, FOV projection, rotation and policy preprocessing without modifying
them. Each output directory must be new. It saves:

- `report.json`: CameraInfo, received-frame/timestamp statistics, invalid-depth
  fractions, crop geometry, normalization statistics and offline ONNX results.
- `depth_snapshot.npz`: final native depth in metres, rotated state cache,
  preprocessed tensor preview `[1,36,36]`, crop, intrinsics and image timestamp.
- `native_depth_with_crop.png`: depth pseudocolor, invalid pixels black and the
  production FOV crop outlined in yellow; not an RGB image.
- `tensor_preview_36x36.png`: normalized tensor pseudocolor, nearest enlarged.

The tensor is a CAMERA-ONLY PREVIEW, not captured from a running controller.
The optional interface check performed by this script uses real camera depth
but ZERO SYNTHETIC `obs_history` and `h_in`. Its outputs are never published.
This checks model input compatibility and finite output, not standing/gait
stability or equivalence to training camera geometry.

Published depth is already filtered by the camera driver. Its ROS timestamps
represent publish time, not exposure time. Received-frame rate/gaps cannot
locate losses uniquely in the sensor, driver, DDS or subscriber. Unique stamps
and different image bytes are useful diagnostics, not proof of zero latency.

## Inspection on 2026-10-10

Camera driver SN `254522074724` was started by the user. No robot controller was
running. Final 12-second audit: `logs/camera-input-20261010-live-verified/`.

324 frames were accepted at 27.22 received frames/s. No duplicate/backward
timestamps, identical consecutive images or preprocessing errors were observed.
Median stamp interval was 33.37 ms, P95 66.51 ms, maximum 159.90 ms.
The final native frame was `848x480`, `16UC1`, optical frame
`head_depth_camera_depth_optical_frame`.

Native K: fx=fy=428.1398926, cx=424.0101013, cy=244.7706909; D is five zeros.
The deployed crop is `[246,2,356,475]` (x,y,width,height). The final native
invalid fraction was 2.36%, crop invalid fraction 0.21%. Tensor range was
[-0.26237,0.01802], finite with no endpoint saturation. ONNX returned finite
`actions [1,29]` and `h_out [1,1,256]` using the synthetic-proprioception probe.

Tests: 28 passed, including audit helpers and existing DAGGER/GA4B deployment
regressions. No motor test, calibration fit or measured-pose closed loop was
performed. This audit does not establish current physical camera extrinsics.

## Live Browser Window

Keep the existing camera running. Start the read-only viewer in another terminal:

```bash
cd /home/bxi/wyg_hiking
bash script/start_dagger_camera_viewer.sh
```

Open `http://127.0.0.1:8766/`. No sudo, rebuild, desktop display or additional
installation is needed. The helper sources the installed ROS workspace and uses
domain 42. It does not start a camera driver, controller or motor command. Stop
only the viewer with Ctrl+C in its terminal; this does not stop the camera.

For a remote browser, forward port 8766 in the editor, or use an SSH tunnel:

```bash
ssh -N -L 8766:127.0.0.1:8766 bxi@<robot-ip>
```

The default bind is loopback only. Optional `--port 8767` changes the port. Do not
expose `--bind 0.0.0.0` outside a trusted network: the viewer has no authentication.

The window separates three data sources:

- Native depth: metres, invalid pixels black, production FOV crop in yellow.
- Preprocessing preview: the installed decoder/projection/policy preprocessing,
  normalized `[1,36,36]`. This is not the controller's cached inference input.
- Actual backend input: `depth_data` from the existing external recorder, with
  `obs_history`/`h_in` shape and finite-value checks. Without compatible records
  this panel waits; it never substitutes the camera-only preview.

The browser updates at up to 5 Hz; receipt Hz reflects the ROS subscriber, not
the display rate. Camera age uses monotonic receipt time; actual-input age uses
record file modification time, NOT exposure time or an inference timestamp.
The deployed 0.5-second timeout labels stale data. The panels are asynchronous
and are not asserted to describe the same source frame or control cycle.
The viewer checks the installed model SHA256 and exact input names/shapes/dtypes
before showing actual records. Camera extrinsics and sim2real gap are not
validated by the window.

Pause freezes the displayed images/metrics only; subscriptions and the robot
continue. Download saves a browser-side NPZ with native depth, state cache,
preview, intrinsics and JSON metadata, plus actual backend inputs when present.
Snapshot download is disabled while paused to avoid exporting a different frame.

### Optional Actual-Input Recording

An ordinary controller launch does not produce these records. During a separate,
user-operated safe hardware test, wrap the controller with the existing recorder.
The command below DOES start the hardware controller; it is not a viewer command.
Do not run it alongside another controller. Keep the already-running camera and
use `enable_cameras:=false` to avoid opening the same camera twice:

```bash
sudo bash -c '
cd /home/bxi/wyg_hiking
source /opt/bxi/bxi_ros2_pkg/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
exec python3 tools/benchmark/collect_calibration.py \
  --output logs/dagger-live-inputs --every 5 --max-samples 10000 --skip-first 10 \
  -- ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py enable_cameras:=false
'
```

The default viewer watches `logs/dagger-live-inputs/dagger_new_position`. Override
the parent root with `--capture-root /absolute/path` if recording elsewhere. The
recorder captures backend calls, not robot-state transitions; `--skip-first` skips
startup calls, not a formal state boundary. Old records remain marked stale.
Recording is bounded but adds copies/disk I/O, and is not a real-time guarantee.
When `--max-samples` is reached the records stop updating; the viewer shows stale.
This inference-dataset "calibration" tool is NOT camera-extrinsic calibration.
No hardware controller or actual-input recording was started when implementing
and testing the viewer.

### Viewer Verification on 2026-10-10

39 tests passed across the viewer, camera audit and DAGGER/GA4B deployment
regressions. Existing Chromium was used to check 1440x1000 desktop and 390x844
mobile layouts, live image updates, pause/resume, browser NPZ download,
disconnect/reconnect and zero JavaScript exceptions. No new browser or package
was installed. Live native depth was 848x480 and preview 36x36; the actual-input
panel correctly waited with no controller running. Screenshots are saved as
`logs/dagger-camera-viewer-desktop.png` and
`logs/dagger-camera-viewer-mobile.png`. Actual backend integration was checked
with recorded-data fixtures, not a running robot controller.
