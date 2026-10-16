# Manual Local Camera Recording

Recording is controlled only by starting the recorder and pressing Ctrl+C in
its terminal. No normal/DAGGER entry, state snapshot, browser or network is
required. The recorder subscribes to the existing camera; it does not start
a driver, publish robot commands or switch a strategy. Existing script names
and output paths are retained. No rebuild or new dependency is needed.

## Two-Terminal Workflow

Both terminals use sudo, ROS domain 42 and the same installed workspace.
This avoids the observed cross-user Fast DDS shared-memory discovery problem.
Use local terminals on the robot for testing without a network. Foreground
processes started through SSH are not guaranteed to survive disconnection.

### Terminal 1: Camera And Robot

Confirm physical support, a working local remote and accessible emergency stop
before starting real hardware. The original script starts in zero torque.

If the old background hardware test is still active, stop it once, with the
robot physically supported, before opening another controller:

```bash
sudo systemctl stop bxi-wyg2-dagger-test.service
```

Then keep this terminal open:

```bash
cd /home/bxi/wyg_hiking
sudo bash script/start_dagger_hw_test.sh
```

This selects only D435 SDK serial `151222073444`, then starts the camera, remote
input and controller. The old D435i `254522074724` may retain its separate owner;
it is never selected automatically. Do not run a second driver for the selected
camera or a second hardware controller. If this camera was deliberately started separately,
keep the previous launch using `enable_cameras:=false` instead of opening it
twice. `RB+B` selects PD_BRAKE; `RB+X` selects normal. Policy selection is
independent of recording and is NOT needed to record camera data.

### Terminal 2: Manual Recorder

The old automatic recorder must not run concurrently. If its transient service
still exists, stop it once:

```bash
sudo systemctl stop bxi-wyg2-dagger-recorder.service
```

Start recording in this terminal:

```bash
cd /home/bxi/wyg_hiking
sudo bash script/start_dagger_camera_recording.sh
```

`RECORDER READY: mode=manual` means the subscriber is ready. The first received
depth frame creates a new session and prints `RECORDING START: <directory>`.
No camera frames means no empty clip directory and no recording success claim.
Fresh RGB, if available, is saved alongside depth. There is no default duration
limit; changing strategy or losing state messages does not stop recording.

Press Ctrl+C IN TERMINAL 2 to finalize AVI/bag files and metadata. Wait for
`RECORDING STOP: recorder_shutdown` and the shell prompt. This stops only
recording, not the robot in Terminal 1. Run the command again to create a new
session without overwriting previous files. Do not use SIGKILL in normal use.

With the robot supported, use its original control/emergency-stop procedure
before stopping Terminal 1. Closing the recorder is not an emergency stop.
No robot control was changed to implement manual recording.

## Local Outputs

```text
/home/bxi/wyg_hiking/logs/dagger-recordings/
  YYYYMMDD-HHMMSS-microseconds-manual/
    metadata.json
    segment-0001/
      depth_input.avi
      color.avi                 # only if fresh supported RGB is available
      frames.jsonl
      depth_bag/
        metadata.yaml
        *.db3
    segment-0002/ ...
```

Metadata schema 2 labels `recording_mode: manual`. It does not invent a robot
state. Frame-level `state` is null if no recent, valid, unambiguous state message
is available. State messages annotate recordings only and never gate start/stop.

`depth_input.avi` contains native depth pseudocolor with the FOV crop, a
camera-only DAGGER preprocessing preview, and actual backend depth ONLY if a
compatible external backend recording is available. Normal does not consume
that preview; camera recording is not proof of DAGGER inference. Actual input
is never replaced by the preview; its panel labels WAITING/STALE/INVALID.

`color.avi` saves camera RGB, capped at 1280 pixels wide. RGB and depth are not
asserted to be aligned or synchronized. CameraInfo is not rescaled to this video.
The lossless depth bag preserves selected original depth Image bytes, encoding
and header, depth/color CameraInfo and available state annotations. Raw RGB is
NOT in the bag. Existing backend-input capture remains optional and separate.

Defaults: selected depth <=10 Hz, 60-second segments, no session time limit,
2 GiB free-space reserve and a bounded queue of eight selected frames. Raw
848x480 16-bit depth at 10 Hz is about 0.5 GB/minute before bag/video overhead.
Low disk or write failure stops saving and reports an error; it does not stop
the robot. Old clips are never deleted automatically. Dropped and invalid-frame
counts are recorded. AVI has nominal fixed FPS, so use frames.jsonl timestamps
for actual timing. Driver header stamps are not claimed to be exposure times.

Limits are optional, for example:

```bash
sudo bash script/start_dagger_camera_recording.sh --fps 5 --min-free-gb 5
```

`--max-session-sec 1200` explicitly enables a 20-minute cap; the default 0 means
no cap. `--state-timeout` affects only optional annotations. If the camera stops
sending frames, recording waits for it without inventing frames or stopping
due to robot state. Ctrl+C still finalizes the session.

Review videos locally or transfer them once networking returns. For bag
inspection, use `ros2 bag info <segment>/depth_bag`. Never replay state/depth
bags into a running hardware controller; use a separate ROS domain/remapping.
Forced kill or power loss may leave the current segment unfinished; previously
closed segments remain available. Metadata marked recording is not proof that
the recorder is currently running.

## Verification

66 offline tests passed across manual recording, the browser viewer, camera
audit and existing DAGGER/GA4B deployment regressions. They cover recording
without any state, continuation through policy changes and stale/invalid state,
default unlimited duration, optional time cap, bounded sampling/queue, segment
rotation, readable AVI/bag files, clean manual stop, disk reserve and restart
without overwriting. Python compilation passed.

A real-camera manual smoke test on 2026-10-11 used the already running D435I
254522074724, with no policy switch or robot command. At 5 Hz it saved 160 depth
frames over 32 seconds, then Ctrl+C closed the session cleanly. Readback decoded
160 depth-preview frames and 158 RGB frames (RGB is optional/asynchronous),
with no invalid previews, RGB conversion errors or queue drops. The bag contained
160 original depth frames with header timestamps matching frames.jsonl.
Evidence: `logs/manual-recording-check-20261011/20261011-005721-214613-manual/`.

Earlier state-gated synthetic evidence under
`logs/camera-recorder-integration-a6ldymzw/` predates this manual workflow;
it is not a real-camera or current manual-recording validation. No policy,
gain, speed limit or button mapping was changed. Recording does not establish
standing stability, camera extrinsic calibration or safe sim2real behavior.
