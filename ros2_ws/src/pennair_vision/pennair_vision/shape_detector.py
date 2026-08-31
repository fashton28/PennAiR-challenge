"""Runs shape detection on camera/image_raw and publishes the results.

Publishes detections (positions, outlines) on `detections` and the
annotated frame on `detections/image`.

ros2 run pennair_vision shape_detector
"""

import cv2
import rclpy
from rclpy.executors import ExternalShutdownException
from cv_bridge import CvBridge
from rclpy.node import Node
from geometry_msgs.msg import Point, Point32, Polygon
from sensor_msgs.msg import Image

from pennair_vision import detection
from pennair_vision.detection import detect_shapes
from pennair_vision_msgs.msg import Shape, ShapeArray


class ShapeDetector(Node):
    def __init__(self):
        super().__init__("shape_detector")
        self.bridge = CvBridge()
        # queue depth 1: the detector is slower than the stream, so keep only
        # the newest frame instead of working through a stale backlog
        self.sub = self.create_subscription(
            Image, "camera/image_raw", self.on_image, 1
        )
        self.pub = self.create_publisher(ShapeArray, "detections", 10)
        self.image_pub = self.create_publisher(Image, "detections/image", 10)
        self.frames = 0
        self.get_logger().info("waiting for frames on camera/image_raw")

    def on_image(self, msg):
        # same per-frame work as the loop in main.py
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        shapes = detect_shapes(frame)
        self.frames += 1
        if self.frames % 100 == 1:
            self.get_logger().info(f"frame {self.frames}: {len(shapes)} shapes")
        cv2.drawContours(frame, shapes, -1, (0, 255, 0), 3)

        out = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out.header = msg.header
        try:
            self.pub.publish(self.to_msg(shapes, msg.header, frame.shape))
            self.image_pub.publish(out)
        except Exception:
            if rclpy.ok():  # a real failure; only shutdown mid-callback is expected
                raise

    def to_msg(self, shapes, header, frame_shape):
        """Pack contours into a ShapeArray: centered pixel centroid, shared
        depth from the circle, 3D position, and the outline polygon."""
        h, w = frame_shape[:2]
        depth = detection.last_Z
        out = ShapeArray()
        out.header = header
        for contour in shapes:
            M = cv2.moments(contour)
            shape = Shape()
            shape.x = int(M["m10"] / M["m00"]) - w // 2
            shape.y = h // 2 - int(M["m01"] / M["m00"])
            shape.is_circle = detection.is_circle(contour)
            shape.outline = Polygon(
                points=[Point32(x=float(px), y=float(py)) for px, py in contour[:, 0, :]]
            )
            if depth is not None:
                # rounded to 0.01 in; the measurement itself is only good to ~1 in
                X, Y, Z = detection.pixel_to_3d(shape.x, shape.y, depth).round(2)
                shape.depth = Z
                shape.position = Point(x=X, y=Y, z=Z)
            else:
                shape.depth = -1.0
            out.shapes.append(shape)
        return out


def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetector()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl+C
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
