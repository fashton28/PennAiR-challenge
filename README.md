# PennAiR-challenge
The completed PennAiR software challenge.

To-Do
[] Part 1 -- Basic shape detection
[X] Research
[] figure out a combination of openCV techniques.

# Part 2 
[X] Refactoring code to fit in a function.
[X] Applying function frame by frame
[X] Centroid (fix extra)
[X] refine comments

Working on background agnostic algorithm



# Part 5: ROS2

`ros2_ws/src/pennair_vision` holds two nodes.

| Node | Subscribes | Publishes |
|---|---|---|
| `video_publisher` | - | `camera/image_raw` (`sensor_msgs/Image`) at the video's own fps |
| `shape_detector` | `camera/image_raw` | `detections/image` (annotated `sensor_msgs/Image`) |

`pennair_vision/detection.py` is the algorithm from `main.py` without the video I/O.

## Setup on macOS (RoboStack)

ROS2 has no official macOS binaries, so it is installed natively through conda packages built by RoboStack.

```sh
brew install micromamba
micromamba shell init --shell zsh --root-prefix ~/mamba   # then open a new terminal
micromamba create -n ros2 -c conda-forge -c robostack-jazzy \
    ros-jazzy-desktop ros-jazzy-cv-bridge ros-jazzy-rqt-image-view colcon-common-extensions rosdep
micromamba env config vars set -n ros2 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
micromamba activate ros2
```

The last variable matters: the default Fast DDS transport silently drops the 6 MB 1080p frames on macOS, while CycloneDDS delivers them at the full 30 fps.

## Run

```sh
micromamba activate ros2
cd ros2_ws
colcon build --symlink-install
source install/setup.zsh
ros2 run pennair_vision video_publisher --ros-args -p video_path:=$PWD/../media/hardvideotest.mp4
ros2 run pennair_vision shape_detector            # second terminal
ros2 run rqt_image_view rqt_image_view            # third terminal, pick /detections/image
```
