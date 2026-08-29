"""Runs shape detection on camera/image_raw and publishes the annotated frame.

    ros2 run pennair_vision shape_detector
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from pennair_vision.detection import detect_shapes


class ShapeDetector(Node):
    def __init__(self):
        super().__init__("shape_detector")
        self.bridge = CvBridge()
        self.sub = self.create_subscription(Image, "camera/image_raw", self.on_image, 10)
        self.pub = self.create_publisher(Image, "detections/image", 10)

    def on_image(self, msg):
        # same per-frame work as the loop in main.py
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        shapes = detect_shapes(frame)
        cv2.drawContours(frame, shapes, -1, (0, 255, 0), 3)

        out = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out.header = msg.header
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ShapeDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
