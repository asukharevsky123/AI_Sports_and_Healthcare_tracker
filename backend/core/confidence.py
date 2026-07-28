from typing import Dict, List


def is_low_confidence(
    keypoints: List[Dict[str, list]],
    minimum_average_visibility: float = 0.50,
    maximum_empty_frame_ratio: float = 0.45,
    maximum_low_visibility_ratio: float = 0.35,
) -> bool:
    """
    Determine whether pose data is reliable enough for angle analysis.
    """
    if not keypoints:
        return True

    empty_frames = 0
    visibility_values = []
    low_visibility_count = 0

    for frame in keypoints:
        if not frame:
            empty_frames += 1
            continue

        for point in frame.values():
            if not point or len(point) < 3:
                low_visibility_count += 1
                continue

            try:
                visibility = float(point[2])
            except (TypeError, ValueError):
                low_visibility_count += 1
                continue

            visibility_values.append(visibility)

            if visibility < 0.20:
                low_visibility_count += 1

    if not visibility_values:
        return True

    average_visibility = (
        sum(visibility_values)
        / len(visibility_values)
    )

    empty_frame_ratio = (
        empty_frames
        / max(len(keypoints), 1)
    )

    total_landmark_count = (
        len(visibility_values)
        + low_visibility_count
    )

    low_visibility_ratio = (
        low_visibility_count
        / max(total_landmark_count, 1)
    )

    return (
        average_visibility < minimum_average_visibility
        or empty_frame_ratio > maximum_empty_frame_ratio
        or low_visibility_ratio > maximum_low_visibility_ratio
    )