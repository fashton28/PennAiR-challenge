"""Starts the whole pipeline: video stream + shape detection.

    ros2 launch pennair_vision pennair.launch.py video_path:=/abs/path/to/video.mp4
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("video_path", description="Absolute path of the video to stream"),
            Node(
                package="pennair_vision",
                executable="video_publisher",
                name="video_publisher",
                parameters=[{"video_path": LaunchConfiguration("video_path")}],
                output="screen",
            ),
            Node(
                package="pennair_vision",
                executable="shape_detector",
                name="shape_detector",
                output="screen",
            ),
        ]
    )
