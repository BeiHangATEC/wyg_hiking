#!/usr/bin/env bash
set -eo pipefail

dagger_project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dagger_camera_serial="${1:-151222073444}"
if [[ $# -gt 1 ]]; then
    echo "usage: sudo bash script/start_dagger_hw_test.sh [CAMERA_SDK_SERIAL]" >&2
    echo "Confirm the camera mount matches dagger_new_position training first." >&2
    exit 2
fi
cd "$dagger_project_dir"
source /opt/bxi/bxi_ros2_pkg/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
echo "Checking selected policy camera (SDK serial): $dagger_camera_serial"
# Other cameras may have independent owners; only probe the selected device.
if ! ros2 run bxi_depth_camera cameras-inspect --serial "$dagger_camera_serial" --require-depth; then
    echo "error: the selected camera is unavailable or cannot open a depth stream; hardware was not started." >&2
    echo "Check its USB connection and SDK serial; do not stop unrelated camera drivers." >&2
    exit 1
fi
echo "DAGGER test: starts in zero_torque; RB+B -> PD_BRAKE, RT+X -> dagger_new_position."
exec ros2 launch bxi_example_py_elf3 example_demo_hw.launch.py \
    enable_cameras:=true depth_camera_serial_no:="$dagger_camera_serial" DEBUG:=true
