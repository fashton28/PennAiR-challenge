"""Human-readable view of /detections: one aligned line per shape.

    ros2 run pennair_vision detections_echo
"""

import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from pennair_vision_msgs.msg import ShapeArray

HEADER = f"{'shape':<8} {'x':>6} {'y':>6} {'depth':>8}  {'X':>8} {'Y':>8} {'Z':>8}  outline"
UNITS = f"{'':<8} {'px':>6} {'px':>6} {'in':>8}  {'in':>8} {'in':>8} {'in':>8}  vertices"


class DetectionsEcho(Node):
    def __init__(self):
        super().__init__("detections_echo")
        self.sub = self.create_subscription(ShapeArray, "detections", self.on_detections, 10)
        print(HEADER + "\n" + UNITS + "\n" + "-" * len(UNITS), flush=True)

    def on_detections(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        clock = time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t % 1 * 10)}"
        print(f"frame @ {clock}  ({len(msg.shapes)} shapes)", flush=True)
        for s in msg.shapes:
            kind = "circle" if s.is_circle else "polygon"
            n = len(s.outline.points)
            outline = f"{n}" if n <= 5 else f"{n} (raw contour)"
            if s.depth < 0:
                print(f"  {kind:<8} {s.x:>6} {s.y:>6} {'?':>8}  {'?':>8} {'?':>8} {'?':>8}  {outline}", flush=True)
            else:
                print(
                    f"  {kind:<8} {s.x:>6} {s.y:>6} {s.depth:>8.1f}"
                    f"  {s.position.x:>8.1f} {s.position.y:>8.1f} {s.position.z:>8.1f}  {outline}",
                    flush=True,
                )


def main(args=None):
    rclpy.init(args=args)
    node = DetectionsEcho()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
