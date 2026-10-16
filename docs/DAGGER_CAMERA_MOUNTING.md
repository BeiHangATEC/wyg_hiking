# DAGGER Camera Mounting Reference and Calibration Plan

## Status and Evidence

Offline preparation only. Do not stop another user's camera/controller or
open an occupied camera. No physical mount, camera stream or simulated
camera pose was changed as part of this preparation.

The deployed depth Mod subscribes to `head_depth_camera`. The archive does
not provide a training camera transform, and the ONNX has no corresponding
metadata. The existing workspace also contains a commented body-camera
"new position". Its name is a clue, NOT proof that dagger_new_position was
trained with that pose. Confirm the training scene/config before machining
a bracket or selecting a camera topic.

Source references, relative to this workspace:

- `src/bxi_example_py_elf3/mods/com.bxi.normal_depth/mod.yaml`
- `src/bxi_example_py_elf3/data/mujoco_simulation/elf3.xml`
- `src/bxi_example_py_elf3/data/mujoco_simulation/sensors/head_depth_camera_camera.xml`
- `src/bxi_example_py_elf3/data/mujoco_simulation/sensors/body_depth_camera_camera.xml`
- `src/bxi_example_py_elf3/data/mujoco_simulation/sensors/head_depth_camera_sensor.xml`

## Coordinate and Mechanical Conventions

Positions below refer to the simulated camera origin, not a case corner,
mounting screw or bracket. Use the physical imager's optical reference and
the manufacturer's mechanical drawing to obtain a bracket offset.
Parent robot frames use +X forward, +Y left, +Z up. Positions are in metres.
MuJoCo quaternion order is w,x,y,z; its camera looks along local -Z, with
local +Y upward in the image. ROS optical frames instead use +Z forward,
+X image-right and +Y image-down. Do not copy a MuJoCo quaternion directly
into a ROS optical-frame transform.

The directions and depression angles below were computed from the XML
quaternions using view_direction = -R[:,2]. They are relative to an upright
parent frame, not a leaning robot or the ground on all poses.

## Current Head Reference

Parent: `head_y_link`.

```text
position_m:        [0.0628, 0.0175, -0.048]
position_mm:       [62.8, 17.5, -48.0]
mujoco_quat_wxyz:  [0.5, 0.5, -0.5, -0.5]
optical_rpy_deg:   [-90, 0, -90]
view_direction:   [1, 0, 0]
depression_deg:   0
```

The optical RPY describes coordinate-frame conversion; it does NOT mean
the physical camera must point 90 degrees downward. The view is level and
forward when both head joints are zero. In the current elf3.xml, head_z_link
is 0.2995 m above torso_link, with no additional head_y_link translation.
With both head joints zero, the camera position relative to torso_link is
`[0.0628, 0.0175, 0.2515]` m. Head movement changes that transform.

Ground height depends on posture. The XML's initial torso world position
is not evidence of the camera height in the training standing pose.

## Body References: Active Old Pose and Commented New Pose

Both are relative to `torso_link`, not head_y_link or the pelvis origin.

Active old XML pose:

```text
position_m:        [0.1152, 0.0175, -0.1358]
mujoco_quat_wxyz:  [0.6963642386, 0.1227878136, -0.1227878136, -0.6963642386]
view_direction:   [0.34202017, 0, -0.93969261]
depression_deg:   70
```

Commented "new position", currently NOT enabled:

```text
position_m:        [0.08599, 0.0175, -0.20651]
position_mm:       [85.99, 17.5, -206.51]
mujoco_quat_wxyz:  [0.6903455271, 0.1530459187, -0.1530459187, -0.6903455271]
view_direction:   [0.42261826, 0, -0.90630779]
depression_deg:   65
```

If training confirms this new body pose, mount the optical origin 85.99 mm
forward, 17.5 mm left and 206.51 mm below torso_link, with the view pointing
forward/downward at 65 degrees below the torso's horizontal plane. Confirm
image roll as well as viewing direction against the quaternion; pointing
the lens in the correct direction alone does not fix image orientation.

The new body pose is 29.21 mm further back and 70.71 mm lower than the old
body pose, with 5 degrees less depression. Neither is currently selected
by the depth Mod's logical camera name. Do not activate either on a guess.

## Stream and Network View

The head simulation sensor's enabled reference is depth 848x480 at 30 Hz,
vertical FOV 58.605976 degrees; color is 1280x720. The file's measured
intrinsics refer to another device serial, not a verified current camera.

The depth policy uses a CameraInfo-based central crop with horizontal FOV
45.2 degrees and vertical FOV 58.0616969 degrees, then resize/rotation,
Gaussian blur if OpenCV is available, depth clip 0.2-3.0 m and normalization
to [-0.5,0.5]. The subsequent asymmetric row crop [2,38) produces 36x36.
Consequently the final model view is not described solely by either raw
camera FOV or the crop FOV. Retain the complete preprocessing chain.

Verify the actual published CameraInfo for the actual resolution and depth
alignment mode. Aligned depth and native depth may have different optical
reference frames/intrinsics. A topic called head_depth_camera is not proof
of a head mount or native-depth optical frame.

## If the Physical Mount Cannot Match

1. Obtain the training camera transform and preprocessing first. Record
   actual camera serial/model, parent robot link, resolution, alignment,
   FPS and rigid mount. Measure the optical origin and bracket offset.
2. Verify intrinsic calibration (fx, fy, cx, cy and distortion) from the
   device and published CameraInfo. If accuracy is insufficient, calibrate
   using multiple checkerboard/ChArUco views in the intended stream mode.
   Do not reuse calibration across resolutions/alignment modes blindly.
3. Determine the robot-link-to-camera optical extrinsic transform. Use a
   surveyed target with a known pose relative to the robot and solvePnP,
   or a suitable multi-pose robot/camera calibration with known robot poses.
   Checkerboard intrinsics alone do not determine this robot-frame offset.
   Hand-eye calibration requires multiple suitable poses and reliable
   robot kinematics; it is not always needed for a fixed body bracket.
4. Verify depth metric scale/bias using targets at known distances, floor
   plane geometry, invalid-pixel behavior, timestamps, latency and drops.
   Save raw depth, CameraInfo, joint/IMU feedback and processed 36x36 input.
5. Represent the measured physical camera pose/intrinsics in the training
   or simulation environment. Compare processed images and run closed-loop
   sim2sim, including camera latency/noise, before supported hardware tests.
6. If the changed view materially changes terrain visibility, fine-tune or
   retrain the student with that view (including renewed DAgger collection
   if applicable), then export and regress the new model. Calibration
   measures the mismatch; it does not make the existing policy invariant.

Cropping/rotation may correct some image-layout or FOV differences. They
cannot generally restore unseen/occluded regions from a translated camera.
Do not claim a 2-D image adjustment removes a 3-D mounting gap.

Before physical locomotion, also verify training KP/KD, joint ordering,
action scales, observation normalization, control period and sensor timing.
Camera calibration does not resolve these independent sim2real gaps.

## Reusing Navigation Records

Navigation already has a provisional reinforced pelvis-camera calibration:
`/home/bxi/bxi_wsm/elf_nav/runs/hip-stool-reinforced-20261010/`.
Its STATUS.md records a manually measured COLOR optical-centre height of
approximately 0.745 m and a pelvis-shell mount below the waist joint.
Do not treat that measurement as current if the mount has since changed.

`calib_reinforced_candidate.json` describes the color optical camera relative
to the navigation head-IMU corrected base frame, not torso_link. The record
explicitly reports missing head/waist feedback and residual uncertainty;
it does not establish a calibrated camera-to-torso transform during gait.
Convert the coordinate chain with the actual robot pose and color/depth
optical transform before comparing against training extrinsics. Do not
substitute absent head-joint feedback with zero.

The navigation RGB-D pipeline uses depth aligned to COLOR and color
CameraInfo. DAGGER currently consumes NATIVE depth/image_rect_raw and
depth/camera_info. Reuse the matching actual native-depth CameraInfo, or
deliberately revalidate the policy view before changing to aligned depth.

NormalDepthState does not look up TF: it passes projected/cropped depth to
the policy. Publishing a corrected navigation TF alone therefore cannot
change the policy's pixels or make its view match training. Prefer matching
the confirmed training mount. If that is not feasible, simulate the actual
mount/intrinsics and revalidate or adapt the policy. Depth-based virtual
view reprojection is only an approximation: it can leave holes at occlusions
and cannot recover scene areas never observed by the physical camera.

## Primary References

- ROS frame conventions: https://raw.githubusercontent.com/ros-infrastructure/rep/master/rep-0103.rst
- MuJoCo camera frames: https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-camera
- OpenCV calibration, solvePnP and hand-eye: https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html
