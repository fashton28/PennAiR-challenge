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

Everything through one script (no environment setup needed beyond the install above):

```sh
./run.sh                       # start the pipeline: video stream + shape detection
./run.sh echo                  # 2nd terminal: readable detections table
./run.sh view                  # 3rd terminal: annotated video window
./run.sh raw                   # raw /detections messages (YAML)
./run.sh media/videotest.mp4   # run on a different video
```

`./run.sh` activates the conda env, sources the workspace, and calls
`ros2 launch pennair_vision pennair.launch.py`, which starts both nodes.

| Topic | Type | Content |
|---|---|---|
| `camera/image_raw` | `sensor_msgs/Image` | the video stream, ~30 fps |
| `detections` | `pennair_vision_msgs/ShapeArray` | per shape: centered pixel centroid, depth (in), 3D position (in), is_circle, outline polygon |
| `detections/image` | `sensor_msgs/Image` | annotated frames |

After editing node code, no rebuild is needed (`--symlink-install`); after editing
messages or adding files, rebuild with `colcon build --symlink-install` in `ros2_ws`.
