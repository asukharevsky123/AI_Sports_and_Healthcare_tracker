from typing import Any, Dict


def analyze_with_fallback(
    frame_count: int,
    mode: str,
    reason: str,
) -> Dict[str, Any]:
    """
    Return a safe fallback result when pose detection is unreliable.

    This function does not claim that an external vision model analyzed
    the video. It gives the user practical recording instructions.
    """
    return {
        "mode": mode,
        "used_fallback": True,
        "issues": [],
        "feedback": (
            "The movement could not be analyzed reliably. "
            "Record a short full-body video with the camera steady, "
            "good lighting, and the entire movement visible. "
            "A side view is usually best for squats and sprint form."
        ),
        "diagnostics": {
            "frames_processed": frame_count,
            "reason": reason,
            "fallback_provider": "local",
        },
    }