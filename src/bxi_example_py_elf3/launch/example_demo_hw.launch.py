import os

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from bxi_example_py_elf3.framework.mod_api.hardware_launch import (
    declare_hardware_launch_arguments,
    hardware_node_from_context,
)

def generate_launch_description():
    state_machine_config = os.path.join(
        get_package_share_path("bxi_example_py_elf3"),
        "config/elf3_state_machine.yaml",
    )
    cameras_launch = os.path.join(
        get_package_share_path("bxi_depth_camera"),
        "launch/cameras.launch.py",
    )
    remote_launch = os.path.join(
        get_package_share_path("remote_controller"),
        "launch/remote_controller.launch.py",
    )

    return LaunchDescription(
        declare_hardware_launch_arguments()
        + [
            DeclareLaunchArgument(
                "enable_cameras",
                default_value="false",
                description="Start depth-camera supervision.",
            ),
            DeclareLaunchArgument(
                "start_remote_controller",
                default_value="true",
                description="Start the configured PowerA/CRSF input node.",
            ),
            DeclareLaunchArgument(
                "depth_camera_serial_no",
                default_value="151222073444",
                description="Explicit camera serial for a depth-policy hardware test.",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(cameras_launch),
                condition=IfCondition(LaunchConfiguration("enable_cameras")),
                launch_arguments={
                    "serial_no": LaunchConfiguration("depth_camera_serial_no"),
                }.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(remote_launch),
                condition=IfCondition(
                    LaunchConfiguration("start_remote_controller")
                ),
            ),
            OpaqueFunction(
                function=lambda context: [hardware_node_from_context(context)]
            ),
            Node(
                package="bxi_example_py_elf3",
                executable="bxi_example_py_elf3_demo",
                name="bxi_example_py_elf3_demo",
                output="screen",
                parameters=[
                    {"/topic_prefix": "hardware/"},
                    {"/state_machine_config": state_machine_config},
                ],
                emulate_tty=True,
                arguments=[("__log_level:=debug")],
            ),
        ]
    )
