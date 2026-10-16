# DAGGER State And Actual Input Audit

This opt-in diagnostic verifies the executing state/model and samples the real
backend inputs in the same control cycle. It does not request a state, change
policy outputs, gains, speed limits or button routes. No new dependency or
package rebuild is needed. Unlike the manual camera recorder, this wraps the
controller launch and therefore MUST NOT run beside another hardware controller.

## Hardware Test

Before starting, provide reliable physical support, an accessible emergency stop
and an operator at the local remote. This command starts real camera/hardware in
zero torque, using the unchanged original startup script and ROS domain 42:

```bash
cd /home/bxi/wyg_hiking
sudo bash script/start_dagger_input_audit.sh
```

The launcher prints a new `logs/dagger-input-audit/YYYYMMDD-HHMMSS-us/` directory.
The default policy camera is D435 SDK serial `151222073444` (USB/ASIC serial
`153123024178`). The old D435i SDK serial `254522074724` is not selected.
An optional SDK serial argument overrides the default. Only the selected camera
is probed; an unrelated camera driver is not a reason to stop other programs.
An unavailable/busy camera fails preflight before hardware launch.
Press RB+B for PD_BRAKE, then RT+X for DAGGER only when safe to do so. RB+X is
GA4B 75000 normal and does not consume camera depth. There is no automatic entry.

During sampled backend calls the terminal prints `[input-audit] QUEUED` with
model, full source-model SHA256, actual state, transition flag, success, depth
frame id and accepted sample count. QUEUED is NOT a claim of durable files.
Preparation and warmup are distinguished from steady running using the actual
in-process controller, not ROS state-topic delivery or policy filenames alone.

Default sampling skips five calls per backend, then copies every tenth call
(about 5 samples/s at 50 Hz) with a bounded queue of 16. It saves up to 2000
samples per backend, then stops sampling without stopping the robot. Copies and
disk I/O add load; this is not a hard-real-time guarantee. Queue overflow drops
selected samples; disk/write failures disable diagnostic saving and report it
without changing inference results. Original backend exceptions and safety
handling are preserved.

The existing independent manual camera recorder can still run in another terminal
with `sudo bash script/start_dagger_camera_recording.sh`. It is not necessary for
these two checks. Never start a second camera driver or hardware controller.

With the robot supported, use its original control/emergency-stop procedure
before Ctrl+C in the hardware terminal. Wait for writer shutdown. An unexpected
kill/power loss can leave a partial last sample; the report reads committed lines.

## Saved Evidence And Read-Only Report

Each model subdirectory contains the unchanged capture.json/dataset.txt/NPY
format, plus `000000.audit.json` for each accepted sample. The sidecar records
the actual state-machine snapshot, whether the call was inside a control cycle,
depth frame id, backend-call ordinal/time/duration, success or error and input/
output summaries. The three tensors are copied BEFORE backend.run, before any
subsequent recurrent-state update. Sidecars and arrays are written before their
dataset.txt commit line. Model identity is in capture.json, not inferred from a
startup message. Each invocation uses a fresh directory to avoid stale evidence.

Run this in another terminal while testing or after clean shutdown, substituting
the output directory printed by the launcher. This command never starts hardware:

```bash
cd /home/bxi/wyg_hiking
python3 tools/diagnostics/dagger_inference_audit.py \
  --report 'logs/dagger-input-audit/<session-directory>'
```

`confirmed: true` requires at least one committed sample with ALL of:

- Model name dagger_new_position and SHA256 matching the currently installed ONNX.
- Current state com.bxi.normal_depth/normal_depth, outside any transition, inside
  a real framework control cycle (not background warmup or preparation).
- Successful backend return with finite output summaries and a positive accepted
  depth-frame counter. This is the controller's counter, not a camera timestamp.
- Readable, finite float32 obs_history [1,960], depth_data [1,36,36], h_in [1,1,256].

The report also gives confirmed sample count, observed states, unique source
depth frame ids, unique saved depth tensors and depth range. A positive result
confirms these two checks, not source-frame freshness, camera extrinsics,
training preprocessing equivalence, motor execution or visual-policy quality.
Few unique tensors in an unchanged scene are not alone proof of a frozen camera.

The existing browser viewer can read the captured NPY format if started with
`--capture-root logs/dagger-input-audit/<session-directory>` (use sudo when the
camera/controller runs as root). Its actual-input panel is independent of the
camera-only preview and does not itself confirm executing state.

Offline verification uses stub control cycles and real ONNX inference, never
motor commands. Those test fixtures must not be presented as real-camera or
hardware verification. Hardware confirmation still requires the operator's run.

## Offline Verification

81 tests passed across this audit, manual recording, the browser viewer, camera
input checks and existing DAGGER/GA4B deployment regressions. The opt-in hook was
tested in a separate Python process using the real installed ONNX and a stub
framework control cycle, without any ROS node or motor command. Tests also cover
pre-run copies of reusable buffers, preserved backend outputs/exceptions,
recorder failure isolation, sampling, legacy recording compatibility and rejection
of warmup, transitions, wrong model hashes, corrupt tensors and non-finite outputs.
Python compilation and shell syntax checks passed. This is not hardware evidence.
