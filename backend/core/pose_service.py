from typing import Dict, List

import cv2
import mediapipe as mp
import numpy as np


mp_pose = mp.solutions.pose


LANDMARK_MAP = {
    "shoulder": mp_pose.PoseLandmark.LEFT_SHOULDER,
    "elbow": mp_pose.PoseLandmark.LEFT_ELBOW,
    "wrist": mp_pose.PoseLandmark.LEFT_WRIST,
    "hip": mp_pose.PoseLandmark.LEFT_HIP,
    "knee": mp_pose.PoseLandmark.LEFT_KNEE,
    "ankle": mp_pose.PoseLandmark.LEFT_ANKLE,
}


def extract_keypoints(
    frames: List[np.ndarray],
) -> List[Dict[str, list]]:
    """
    Extract selected body landmarks from each OpenCV frame.
    """
    all_keypoints: List[Dict[str, list]] = []

    # model_complexity=0 uses MediaPipe's lightest pose model.
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.45,
        min_tracking_confidence=0.45,
    ) as pose:

        for frame in frames:
            frame_data: Dict[str, list] = {}

            if frame is None or frame.size == 0:
                all_keypoints.append(frame_data)
                continue

            # OpenCV reads BGR images.
            # MediaPipe expects RGB images.
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            rgb_frame.flags.writeable = False

            results = pose.process(rgb_frame)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                for name, landmark_index in LANDMARK_MAP.items():
                    landmark = landmarks[
                        landmark_index.value
                    ]

                    frame_data[name] = [
                        float(landmark.x),
                        float(landmark.y),
                        float(landmark.visibility),
                    ]

            all_keypoints.append(frame_data)

    return all_keypoints