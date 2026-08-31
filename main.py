import os
import subprocess

import cv2
import numpy as np

VIDEO_PATH = "./media/hardvideotest.mp4"
OUTPUT_PATH = "./media/output.mp4"
RAW_PATH = "./media/output_raw.mp4"  # mp4v intermediate, re-encoded by ffmpeg

# camera intrinsics (given); principal point (0, 0) => pixels measured from the frame center
K = np.array([[2564.3186869, 0, 0], [0, 2569.70273111, 0], [0, 0, 1]])
K_INV = np.linalg.inv(K)
CIRCLE_RADIUS_IN = 10.0  # given

last_Z = None  # depth of the surface, from the last frame with a clean circle


def is_circle(contour):
    # fill ratio vs enclosing circle: ~0.9 circle, ~0.76 pentagon
    _, r_px = cv2.minEnclosingCircle(contour)
    return cv2.contourArea(contour) / (np.pi * r_px**2) > 0.85


def touches_border(contour, w, h):
    x, y, bw, bh = cv2.boundingRect(contour)
    return x == 0 or y == 0 or x + bw >= w or y + bh >= h


def circle_depth(contour):
    r_px = np.sqrt(cv2.contourArea(contour) / np.pi)
    return K[0, 0] * CIRCLE_RADIUS_IN / r_px


def pixel_to_3d(x, y, Z):
    return Z * K_INV @ np.array([x, y, 1.0])


def fill_holes(mask):
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(padded, None, (0, 0), 255)
    return mask | cv2.bitwise_not(padded[1:-1, 1:-1])


def drop_small(mask, min_area):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels].astype(np.uint8) * 255


def background_distance(lab, med, mad):
    return np.sqrt((((lab - med) / mad) ** 2).sum(axis=2))


def texture(frame, k=7):
    # Laplacian energy: zero on flat or linearly shaded patches, high on gravel/grass
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.blur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)), (k, k))


def segment(frame):
    band = 7
    k_color = 5.0

    # background colour model: median/MAD of the flattened frame in Lab
    flat = cv2.medianBlur(frame, 21)
    lab_flat = cv2.cvtColor(flat, cv2.COLOR_BGR2LAB).astype(np.float32)
    pixels = lab_flat.reshape(-1, 3)
    med = np.median(pixels, axis=0)
    mad = np.maximum(np.median(np.abs(pixels - med), axis=0) * 1.4826, 2.0)  # floor: gray bg has zero chroma spread

    coarse = (background_distance(lab_flat, med, mad) > k_color).astype(np.uint8) * 255
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    # gradient bands inside a shape can match the background colour; they are smooth
    energy = texture(frame)
    smooth = energy < 0.5 * np.median(energy)
    near = cv2.dilate(coarse, np.ones((31, 31), np.uint8)) > 0
    coarse |= (smooth & near).astype(np.uint8) * 255
    coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    coarse = fill_holes(drop_small(coarse, 2000))

    # re-decide a band around each region from the raw pixels (the flattened frame smears edges)
    lab = cv2.cvtColor(cv2.medianBlur(frame, 7), cv2.COLOR_BGR2LAB).astype(np.float32)
    d_bg = background_distance(lab, med, mad)
    kernel = np.ones((2 * band + 1, 2 * band + 1), np.uint8)
    core = cv2.erode(coarse, kernel) > 0
    ring = cv2.dilate(coarse, kernel) > 0
    weight = cv2.blur(core.astype(np.float32), (21, 21))
    shape_lab = np.dstack([cv2.blur(lab_flat[:, :, i] * core, (21, 21)) for i in range(3)])
    shape_lab /= np.maximum(weight, 1e-6)[..., None]
    d_shape = background_distance(lab, shape_lab, mad)
    smooth_edge = cv2.dilate(smooth.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    joins = ring & (weight > 0.02) & (((d_shape < d_bg) & (d_bg > 2 * k_color)) | (smooth_edge & (d_shape < 1.5 * d_bg)))

    mask = (core | joins).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return fill_holes(drop_small(mask, 8000))


def intersect(line_a, line_b):
    (p0, d0), (p1, d1) = line_a, line_b
    cross = d0[0] * d1[1] - d0[1] * d1[0]
    if abs(cross) < 1e-6:
        return None
    t = ((p1[0] - p0[0]) * d1[1] - (p1[1] - p0[1]) * d1[0]) / cross
    return p0 + t * d0


def polygonize(contour):
    # fit a line to each outline edge; the shape is the polygon with the fewest sides (3-5)
    # whose long sides, extended and intersected, cover the outline without adding much area
    if is_circle(contour):
        return contour
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    points = contour.reshape(-1, 2)
    vertices = cv2.approxPolyDP(contour, 0.012 * perimeter, True).reshape(-1, 2)
    n = len(vertices)
    if n < 3:
        return contour
    index = [int(np.argmin(((points - v) ** 2).sum(axis=1))) for v in vertices]
    lines, lengths = [], []
    for i in range(n):
        a, b = index[i], index[(i + 1) % n]
        segment = points[a : b + 1] if b >= a else np.vstack([points[a:], points[: b + 1]])
        vx, vy, x0, y0 = cv2.fitLine(segment.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
        lines.append((np.array([x0, y0]), np.array([vx, vy])))
        lengths.append(np.linalg.norm(vertices[(i + 1) % n] - vertices[i]))
    by_length = sorted(range(n), key=lambda i: -lengths[i])
    for k in (3, 4, 5):
        if k > n:
            break
        sides = sorted(by_length[:k])
        corners = [intersect(lines[sides[j]], lines[sides[(j + 1) % k]]) for j in range(k)]
        if any(c is None for c in corners):
            continue
        polygon = np.array(corners).round().astype(np.int32).reshape(-1, 1, 2)
        if not cv2.isContourConvex(polygon):
            continue
        if not 0.97 * area <= cv2.contourArea(polygon) <= 1.12 * area:
            continue
        inside = [cv2.pointPolygonTest(polygon, (float(x), float(y)), True) for x, y in points[::4]]
        if np.mean(np.array(inside) >= -6) < 0.97:
            continue
        return polygon
    return contour


def detect_shapes(frame):
    mask = segment(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    shapes = [polygonize(c) for c in contours if cv2.contourArea(c) > 8000]

    # the circle is the only shape of known size; on a flat surface every shape shares its depth
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
        x = cX - w // 2  # pixels from the frame center, y up
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


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {VIDEO_PATH}")

    # >=60 fps output: duplicate frames and stamp the file dup * source fps
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    dup = max(1, int(np.ceil(60 / src_fps)))
    out_fps = src_fps * dup
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(RAW_PATH, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    done = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            shapes = detect_shapes(frame)
            cv2.drawContours(frame, shapes, -1, (0, 255, 0), 3)
            for _ in range(dup):
                writer.write(frame)
            cv2.imshow("detections", frame)

            done += 1
            if done % 30 == 0:
                print(f"processed {done}/{total} frames", flush=True)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()
        print(f"wrote {RAW_PATH} ({done} frames processed)", flush=True)

    print("encoding for QuickTime...", flush=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            RAW_PATH,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            OUTPUT_PATH,
        ],
        check=True,
    )
    os.remove(RAW_PATH)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
