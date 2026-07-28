import os
import tempfile
from typing import List

import cv2
import numpy as np


def resize_preserving_aspect_ratio(
    frame: np.ndarray,
    target_height: int,
) -> np.ndarray:
    """
    Resize a frame while preserving its original proportions.
    """
    if frame is None or frame.size == 0:
        raise ValueError("The video contains an invalid frame.")

    height, width = frame.shape[:2]

    if height <= 0 or width <= 0:
        raise ValueError("The video frame has invalid dimensions.")

    scale = target_height / float(height)
    target_width = max(
        int(round(width * scale)),
        1,
    )

    return cv2.resize(
        frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )


def process_video(
    video_bytes: bytes,
    target_fps: int = 3,
    target_height: int = 256,
    max_frames: int = 40,
) -> List[np.ndarray]:
    """
    Decode video bytes and return a limited number of resized frames.

    The limits are intentionally conservative so the service can run
    on a low-memory Render instance.
    """
    if not video_bytes:
        return []

    frames: List[np.ndarray] = []
    temporary_path = None
    capture = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4",
        ) as temporary_file:
            temporary_file.write(video_bytes)
            temporary_path = temporary_file.name

        capture = cv2.VideoCapture(temporary_path)

        if not capture.isOpened():
            return []

        source_fps = capture.get(cv2.CAP_PROP_FPS)

        if not source_fps or source_fps <= 0:
            source_fps = 30.0

        safe_target_fps = max(target_fps, 1)

        frame_skip = max(
            int(round(source_fps / safe_target_fps)),
            1,
        )

        frame_index = 0

        while len(frames) < max_frames:
            success, frame = capture.read()

            if not success:
                break

            if frame_index % frame_skip == 0:
                resized_frame = resize_preserving_aspect_ratio(
                    frame,
                    target_height,
                )

                frames.append(resized_frame)

            frame_index += 1

        return frames

    finally:
        if capture is not None:
            capture.release()

        if temporary_path and os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass