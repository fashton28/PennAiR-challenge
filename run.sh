#!/bin/zsh
# One-command runner for the PennAiR ROS2 pipeline.
#
#   ./run.sh                      start video stream + shape detection
#   ./run.sh <path/to/video.mp4>  same, with a different video
#   ./run.sh echo                 readable detections table   (2nd terminal)
#   ./run.sh view                 annotated video window      (3rd terminal)
#   ./run.sh raw                  raw /detections messages
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"

export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/mamba}"

eval "$(micromamba shell hook -s zsh)" 2>/dev/null || {
  echo "micromamba not found; install with: brew install micromamba" >&2; exit 1; }
micromamba activate ros2
source "$ROOT/ros2_ws/install/setup.zsh"

case "${1:-}" in
  echo) exec ros2 run pennair_vision detections_echo ;;
  view) exec ros2 run rqt_image_view rqt_image_view /detections/image ;;
  raw)  exec ros2 topic echo /detections ;;
  *)    VIDEO="${1:-$ROOT/media/hardvideotest.mp4}"
        [[ "$VIDEO" = /* ]] || VIDEO="$PWD/$VIDEO"
        exec ros2 launch pennair_vision pennair.launch.py "video_path:=$VIDEO" ;;
esac
