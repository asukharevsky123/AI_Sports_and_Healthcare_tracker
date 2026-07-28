from dataclasses import dataclass
from typing import List, Tuple

import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose


@dataclass
class SubjectFilterResult:
    frames: List[np.ndarray]
    total_frames: int
    kept_frames: int
    removed_frames: int


# Landmarks used to confirm that a real person is visible.
CORE_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
]


def _has_visible_subject(
    pose_landmarks,
    minimum_visible_landmarks: int = 4,
    minimum_visibility: float = 0.35,
    minimum_body_width: float = 0.05,
    minimum_body_height: float = 0.15,
) -> bool:
    """
    Determine whether a detected pose is likely to be a real,
    sufficiently visible person.

    MediaPipe coordinates are normalized from 0 to 1.
    """
    if pose_landmarks is None:
        return False

    landmarks = pose_landmarks.landmark

    visible_points = []

    for landmark_index in CORE_LANDMARKS:
        landmark = landmarks[landmark_index.value]

        if landmark.visibility >= minimum_visibility:
            visible_points.append(
                (
                    float(landmark.x),
                    float(landmark.y),
                )
            )

    if len(visible_points) < minimum_visible_landmarks:
        return False

    x_values = [point[0] for point in visible_points]
    y_values = [point[1] for point in visible_points]

    body_width = max(x_values) - min(x_values)
    body_height = max(y_values) - min(y_values)

    # Reject tiny or accidental detections.
    if body_width < minimum_body_width:
        return False

    if body_height < minimum_body_height:
        return False

    return True


def filter_frames_with_subject(
    frames: List[np.ndarray],
    minimum_visible_landmarks: int = 4,
    minimum_visibility: float = 0.35,
) -> SubjectFilterResult:
    """
    Remove every frame in which MediaPipe cannot confidently
    detect a visible human subject.

    Returns the filtered frames and processing statistics.
    """
    if not frames:
        return SubjectFilterResult(
            frames=[],
            total_frames=0,
            kept_frames=0,
            removed_frames=0,
        )

    filtered_frames: List[np.ndarray] = []

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        smooth_landmarks=False,
        enable_segmentation=False,
        min_detection_confidence=0.40,
        min_tracking_confidence=0.40,
    ) as detector:

        for frame in frames:
            if frame is None or frame.size == 0:
                continue

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            rgb_frame.flags.writeable = False

            results = detector.process(rgb_frame)

            subject_is_visible = _has_visible_subject(
                results.pose_landmarks,
                minimum_visible_landmarks=minimum_visible_landmarks,
                minimum_visibility=minimum_visibility,
            )

            if subject_is_visible:
                filtered_frames.append(frame)

    total_frames = len(frames)
    kept_frames = len(filtered_frames)

    return SubjectFilterResult(
        frames=filtered_frames,
        total_frames=total_frames,
        kept_frames=kept_frames,
        removed_frames=total_frames - kept_frames,
    )