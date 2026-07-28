from typing import Sequence

import numpy as np


def _to_point(point: Sequence[float]) -> np.ndarray:
    """
    Convert a coordinate sequence into a two-dimensional NumPy point.

    Landmark values may contain:
        [x, y]
    or:
        [x, y, visibility]
    """
    array = np.asarray(point, dtype=float)

    if array.size < 2:
        raise ValueError(
            "A point must contain at least x and y coordinates."
        )

    return array[:2]


def calculate_angle(
    point_a: Sequence[float],
    point_b: Sequence[float],
    point_c: Sequence[float],
) -> float:
    """
    Calculate angle ABC in degrees.

    point_b is the vertex of the angle.
    """
    a = _to_point(point_a)
    b = _to_point(point_b)
    c = _to_point(point_c)

    vector_ba = a - b
    vector_bc = c - b

    length_ba = np.linalg.norm(vector_ba)
    length_bc = np.linalg.norm(vector_bc)

    denominator = length_ba * length_bc

    if denominator <= 1e-12:
        return 0.0

    cosine_angle = np.dot(
        vector_ba,
        vector_bc,
    ) / denominator

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0,
    )

    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    return float(angle)


def calculate_flexion_angle(
    point_a: Sequence[float],
    point_b: Sequence[float],
    point_c: Sequence[float],
) -> float:
    """
    Calculate joint flexion.

    This currently returns the interior joint angle so it remains
    compatible with the original analysis modules.

    Examples:
        Straight elbow or knee: approximately 180 degrees
        Bent elbow or knee: a smaller angle
    """
    return calculate_angle(
        point_a,
        point_b,
        point_c,
    )


def calculate_lean_angle(
    upper_point: Sequence[float],
    lower_point: Sequence[float],
) -> float:
    """
    Calculate body-segment lean from the vertical axis.

    A vertical body segment is approximately 0 degrees.
    Greater values indicate greater lean away from vertical.
    """
    upper = _to_point(upper_point)
    lower = _to_point(lower_point)

    segment = upper - lower

    segment_length = np.linalg.norm(segment)

    if segment_length <= 1e-12:
        return 0.0

    # Image y-coordinates increase downward.
    # This vertical reference points upward.
    vertical = np.array(
        [0.0, -1.0],
        dtype=float,
    )

    cosine_angle = np.dot(
        segment,
        vertical,
    ) / segment_length

    cosine_angle = np.clip(
        cosine_angle,
        -1.0,
        1.0,
    )

    angle = np.degrees(
        np.arccos(cosine_angle)
    )

    # Report the smaller deviation from the vertical direction.
    if angle > 90.0:
        angle = 180.0 - angle

    return float(abs(angle))