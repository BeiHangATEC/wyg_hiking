# Export Notes

Source: `/home/bxi/bxi_wyg2`, snapshot prepared 2026-10-16.
Destination: `BeiHangATEC/wyg_hiking`.

The source workspace had no `.git` metadata, so an upstream commit cannot be
reliably claimed. Attribution is to the official BXI ROS 2 example project;
existing source headers and third-party notices are preserved.

Included:

- `bxi_example_py_elf3`: framework, configuration, launch files, simulation data,
  depth Mod and basic actions required for PD_BRAKE/zero torque/normal routes.
- Current `dagger_new_position.onnx`, legacy depth model used by the Mod's
  alternate implementation, and supporting basic-action assets.
- `bxi_depth_camera`: selected-serial deployment, actual-frame preflight, vendor
  fallback SDKs with their existing notices, and camera tests.
- `remote_controller`: current input mapping/configuration and its tests.
- Manual recorder, browser viewer, camera-only probe, real-input audit and the
  calibration tools used by the audit; deployment helpers and related docs.

Excluded:

- SONIC and forward-flip Mods, unrelated BMS/input-demo packages and Pico tests.
- The unused historical `dagger2.onnx` model and uploaded ZIP archive.
- `build/`, `install/`, `log/`, `logs/`, videos, bags, cached Python files and
  machine credentials. There are no compiled controller build products.
- Machine-wide hardware SDK/configuration, service changes and robot settings.

Only this export's documentation paths were adapted from `bxi_wyg2` to
`wyg_hiking`. Policy parameters, model bytes, safety behavior, gains and input
mappings were not changed. The original deployment remains in `bxi_wyg2`.

Offline verification of this exported workspace on 2026-10-16:

- All three included ROS 2 packages built successfully with `--merge-install`.
- 122 Python tests passed for camera preflight, depth input/recording/viewing,
  inference auditing and DAGGER/GA4B deployment behavior.
- Both C++ remote-controller parser/control-rule tests passed.
- All 22 bundled model/motion assets passed `MODEL_SHA256SUMS` verification.
- No robot controller or camera driver was started for export verification.

Known limits from recent testing:

- On 2026-10-15 the newly connected D435 initially suffered USB enumeration
  errors and later one camera-driver run timed out before yielding frames.
  After reconnecting/restarting, the camera-only probe accepted 233 frames in
  10 seconds and produced a finite `[1,36,36]` tensor with real CameraInfo.
  This is not a long-duration stability or running-controller validation.
- A previous foreground run experienced a long scheduling gap and eventually
  a depth-timeout safety transition. That incident was not fixed by exporting
  this repository; do not interpret this snapshot as a validated safety release.
- Historical documents refer to local diagnostic captures that are deliberately
  not published. Their absence is not evidence of a successful current test.
- The full upstream test suite is not claimed: this focused export omits
  unrelated Mods, and the inherited default-camera-config test expects automatic
  profiles while the deployment config selects explicit stream profiles.
