#!/usr/bin/env bash
set -eo pipefail

dagger_viewer_project="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$dagger_viewer_project"
source /opt/bxi/bxi_ros2_pkg/setup.bash
source install/setup.bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=0
exec python3 tools/diagnostics/dagger_camera_viewer.py "$@"
