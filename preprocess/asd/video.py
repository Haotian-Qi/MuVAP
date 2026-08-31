"""Reading face crops off a source video.

Shared by the dataset adapters that pack from video rather than from
pre-extracted frames.
"""

import cv2
import numpy as np


def crop_face(frame, bbox, gray=True, dim=(112, 112)):
    if bbox is None:
        return np.full(dim if gray else (dim[1], dim[0], 3), 128, dtype=np.uint8)

    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

    if x1 >= x2 or y1 >= y2:
        return np.full(dim if gray else (dim[1], dim[0], 3), 128, dtype=np.uint8)

    h, w = frame.shape[:2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    size = max(x2 - x1, y2 - y1)
    half = size // 2

    x1s, y1s = cx - half, cy - half
    x2s, y2s = x1s + size, y1s + size

    x1c, y1c = max(0, x1s), max(0, y1s)
    x2c, y2c = min(w, x2s), min(h, y2s)

    if x1c >= x2c or y1c >= y2c:
        return np.full(dim if gray else (dim[1], dim[0], 3), 128, dtype=np.uint8)

    cropped = frame[y1c:y2c, x1c:x2c]

    if gray:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        border_val = 128
    else:
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        border_val = (128, 128, 128)

    pad_top, pad_left = y1c - y1s, x1c - x1s
    pad_bottom, pad_right = y2s - y2c, x2s - x2c

    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cropped = cv2.copyMakeBorder(
            cropped,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=border_val,
        )

    return cv2.resize(cropped, dim)


def load_visual(video_path, start_frame=0, target_frames=None):
    """Decode a native-FPS video interval without changing its timeline."""
    if start_frame < 0:
        raise ValueError("start_frame cannot be negative")
    if target_frames is not None and target_frames < 0:
        raise ValueError("target_frames cannot be negative")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise ValueError(f"video has an invalid frame rate: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    while target_frames is None or len(frames) < target_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames, fps
