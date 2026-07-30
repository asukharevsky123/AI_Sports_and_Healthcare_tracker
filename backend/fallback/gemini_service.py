import base64
import os
from typing import Any, Dict, List

import cv2
import numpy as np
import requests


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def _encode_frame(frame: np.ndarray) -> str:
    """
    Convert an OpenCV BGR frame into a base64 JPEG string.
    """
    if frame is None or frame.size == 0:
        raise ValueError("Cannot encode an empty frame.")

    success, encoded_image = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 75],
    )

    if not success:
        raise ValueError("Failed to encode video frame.")

    return base64.b64encode(
        encoded_image.tobytes()
    ).decode("utf-8")


def analyze_with_fallback(
    frames: List[np.ndarray],
    mode: str,
    reason: str = "Pose confidence was too low.",
) -> Dict[str, Any]:
    """
    Return fallback analysis when MediaPipe cannot analyze the video
    reliably.

    When GEMINI_API_KEY is available, this function attempts a Gemini
    vision request. Otherwise, it returns local recording guidance.
    """
    frame_count = len(frames)

    local_feedback = (
        "The movement could not be analyzed reliably because the "
        "subject was not visible clearly enough in enough frames. "
        "Please record a short full-body video with good lighting, "
        "a steady camera, and the entire movement visible."
    )

    if not GEMINI_API_KEY or not frames:
        return {
            "mode": mode,
            "used_fallback": True,
            "issues": [],
            "feedback": local_feedback,
            "diagnostics": {
                "frames_available": frame_count,
                "reason": reason,
                "fallback_provider": "local",
            },
        }

    try:
        # Use only a few frames to keep memory and request size low.
        maximum_frames = 4

        if frame_count <= maximum_frames:
            selected_frames = frames
        else:
            selected_indices = np.linspace(
                0,
                frame_count - 1,
                maximum_frames,
                dtype=int,
            )

            selected_frames = [
                frames[index]
                for index in selected_indices
            ]

        request_parts = [
            {
                "text": (
                    f"Analyze this {mode} movement video. "
                    "Describe visible posture or movement issues and "
                    "provide short, practical improvement suggestions. "
                    "Do not diagnose a medical condition."
                )
            }
        ]

        for frame in selected_frames:
            request_parts.append(
                {
                    "inline_data": {
                        "mime_type": "image/jpeg",
                        "data": _encode_frame(frame),
                    }
                }
            )

        endpoint = (
            "https://generativelanguage.googleapis.com/"
            "v1beta/models/gemini-1.5-flash:generateContent"
            f"?key={GEMINI_API_KEY}"
        )

        response = requests.post(
            endpoint,
            json={
                "contents": [
                    {
                        "parts": request_parts
                    }
                ]
            },
            timeout=45,
        )

        response.raise_for_status()
        response_data = response.json()

        feedback = (
            response_data
            .get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text")
        )

        if not feedback:
            feedback = local_feedback

        return {
            "mode": mode,
            "used_fallback": True,
            "issues": [],
            "feedback": feedback,
            "diagnostics": {
                "frames_available": frame_count,
                "frames_sent": len(selected_frames),
                "reason": reason,
                "fallback_provider": "gemini",
            },
        }

    except Exception as exc:
        print(
            f"Gemini fallback failed: {repr(exc)}",
            flush=True,
        )

        return {
            "mode": mode,
            "used_fallback": True,
            "issues": [],
            "feedback": local_feedback,
            "diagnostics": {
                "frames_available": frame_count,
                "reason": reason,
                "fallback_provider": "local",
                "fallback_error": str(exc),
            },
        }


# Compatibility alias for any older code still using this name.
def analyze_with_gemini(
    frames: List[np.ndarray],
    mode: str,
) -> Dict[str, Any]:
    return analyze_with_fallback(
        frames=frames,
        mode=mode,
        reason="Pose confidence was too low.",
    )