import os
import subprocess

import cv2
import numpy as np

VIDEO_PATH = "./media/hardvideotest.mp4"  # TODO: replace with the real video file
OUTPUT_PATH = "./media/output.mp4"  # annotated video written here
RAW_PATH = "./media/output_raw.mp4"  # intermediate OpenCV output (mp4v)


def detect_shapes(frame):
    """Detect solid shapes in one BGR frame; returns their contours."""
    # flatten the background texture (shapes conserve their original form)
    flat = cv2.medianBlur(frame, 21)

    # background color model: the background dominates the frame, so the
    # image-wide median color is the background color, and each pixel's
    # distance from it says "how not-background is this?"
    bg_color = np.median(flat.reshape(-1, 3), axis=0)
    dist = np.linalg.norm(flat.astype(np.float32) - bg_color, axis=2)

    # threshold using background statistics (median + 8*MAD), recomputed per
    # frame so the cut adapts to lighting and background changes
    med = np.median(dist)
    mad = np.median(np.abs(dist - med))
    mask = (dist > med + 8 * mad).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    # contours + area filter drops residual background speckle + implementing centroid detection
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 8000:
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)
            cv2.putText(
                frame,
                "testing!",
                (100 + cX, 100 + cY),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

    return [c for c in contours if cv2.contourArea(c) > 8000]


def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise SystemExit(f"could not open video: {VIDEO_PATH}")

    # the writer's fps is playback metadata, not a quality dial: stamping a
    # 30 fps source as 120 just plays it 4x too fast. To guarantee >=60 fps
    # at the CORRECT speed, each frame is written `dup` times and the file is
    # stamped dup * source fps (e.g. 30.3 fps source -> 2 copies at 60.6 fps)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    dup = max(1, int(np.ceil(60 / src_fps)))
    out_fps = src_fps * dup
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # OpenCV writes a reliable mp4v intermediate; QuickTime needs H.264, but
    # this build's own avc1 writer produces corrupt streams, so ffmpeg does
    # the final H.264 encode below instead
    writer = cv2.VideoWriter(RAW_PATH, cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    done = 0
    # the finally block guarantees writer.release() runs even on Ctrl+C or a
    # crash; without it the mp4 never gets its index (moov atom) and no
    # player can open it
    try:
        while True:
            # streamed input: one frame per iteration, no peeking ahead
            ok, frame = cap.read()
            if not ok:  # end of video
                break

            shapes = detect_shapes(frame)
            cv2.drawContours(frame, shapes, -1, (0, 255, 0), 3)
            for _ in range(dup):
                writer.write(frame)
            cv2.imshow("detections", frame)

            done += 1
            if done % 30 == 0:
                print(f"processed {done}/{total} frames", flush=True)

            # waitKey(1) keeps the loop running (0 would freeze on this frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        writer.release()  # finalizes the intermediate file
        cv2.destroyAllWindows()
        print(f"wrote {RAW_PATH} ({done} frames processed)", flush=True)

    # re-encode to H.264 for macOS: -c:v libx264 is the codec QuickTime
    # expects and -pix_fmt yuv420p the pixel format it requires; -y allows
    # overwriting a previous output
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
