from typing import Sequence

import numpy as np


def calculate_angle(
    point_a: Sequence[float],
    point_b: Sequence[float],
    point_c: Sequence[float],
) -> float:
    """
    Calculate the angle ABC in degrees.
    """
    a = np.asarray(point_a, dtype=float)
    b = np.asarray(point_b, dtype=float)
    c = np.asarray(point_c, dtype=float)

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

    return float(
        np.degrees(
            np.arccos(cosine_angle)
        )
    )