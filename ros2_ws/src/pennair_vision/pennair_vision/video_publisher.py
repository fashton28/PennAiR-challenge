"""Streams a video file as sensor_msgs/Image at the file's native frame rate.

    ros2 run pennair_vision video_publisher --ros-args -p video_path:=/path/to/video.mp4
"""

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class VideoPublisher(Node):
    def __init__(self):
        super().__init__("video_publisher")
        self.declare_parameter("video_path", "")
        path = self.get_parameter("video_path").value
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open video: {path}")

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, "camera/image_raw", 10)

        # one frame per timer tick keeps the stream at the source's real speed
        fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.timer = self.create_timer(1.0 / fps, self.publish_frame)
        self.get_logger().info(f"streaming {path} at {fps:.1f} fps")

    def publish_frame(self):
        ok, frame = self.cap.read()
        if not ok:  # end of video: start over
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.cap.read()
            if not ok:
                return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera"
        self.pub.publish(msg)

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = VideoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
