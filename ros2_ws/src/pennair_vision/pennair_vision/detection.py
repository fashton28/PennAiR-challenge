"""Shape detection with depth from camera intrinsics.

Same algorithm as main.py at the repository root, without the video I/O,
so a ROS node can call detect_shapes() on each incoming frame.
"""

import cv2
import numpy as np

# camera intrinsics (given): x = fx*X/Z, y = fy*Y/Z with the principal point
# at (0, 0), so pixels are measured from the frame center
K = np.array([[2564.3186869, 0, 0], [0, 2569.70273111, 0], [0, 0, 1]])
K_INV = np.linalg.inv(K)
CIRCLE_RADIUS_IN = 10.0  # given

# depth of the flat surface; kept from the last frame with a clean circle
last_Z = None


def is_circle(contour):
    """Fill ratio vs the enclosing circle: ~0.9 circle, ~0.76 pentagon."""
    _, r_px = cv2.minEnclosingCircle(contour)
    return cv2.contourArea(contour) / (np.pi * r_px**2) > 0.85


def touches_border(contour, w, h):
    """Clipped by the frame edge, so its apparent size is wrong."""
    x, y, bw, bh = cv2.boundingRect(contour)
    return x == 0 or y == 0 or x + bw >= w or y + bh >= h


def circle_depth(contour):
    """Z = fx * R / r_px; r from area since the enclosing circle reads large."""
    r_px = np.sqrt(cv2.contourArea(contour) / np.pi)
    return K[0, 0] * CIRCLE_RADIUS_IN / r_px


def pixel_to_3d(x, y, Z):
    """Centered pixel at depth Z -> camera-frame (X, Y, Z) in inches."""
    return Z * K_INV @ np.array([x, y, 1.0])


def detect_shapes(frame):
    """Detect solid shapes in one BGR frame, annotate them, return contours."""
    # flatten the background texture, then measure each pixel's color
    # distance from the frame's median (background) color
    flat = cv2.medianBlur(frame, 21)
    bg_color = np.median(flat.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(flat.astype(np.float32) - bg_color, axis=2)

    # threshold from background statistics, recomputed per frame
    med = np.median(dist)
    mad = np.median(np.abs(dist - med))
    mask = (dist > med + 8 * mad).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = [c for c in contours if cv2.contourArea(c) > 8000]

    # the circle is the only shape of known size; on a flat surface facing
    # the camera every shape shares its depth
    global last_Z
    h, w = frame.shape[:2]
    circles = [c for c in shapes if is_circle(c) and not touches_border(c, w, h)]
    if circles:
        last_Z = circle_depth(circles[0])
    Z = last_Z

    for contour in shapes:
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        # pixels from the frame center, y up
        x = cX - w // 2
        y = h // 2 - cY
        label = f"({x}, {y}, {Z:.1f} in)" if Z is not None else f"({x}, {y})"
        cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
        cv2.putText(
            frame,
            label,
            (100 + cX, 100 + cY),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    return shapes
