"""Changing the frame of points and vectors."""

import numpy as np


def rotate_vectors(rotations, vectors) -> np.ndarray:
    """Rotate each vector by its own rotation, for example taxel forces from their
    taxel frames to the sensor frame.

    Args:
        rotations: Rotations `(n, 3, 3)`, or one `(3, 3)` applied to all.
        vectors: Vectors `(n, 3)`.

    Returns:
        Rotated vectors `(n, 3)`.
    """
    rotations = np.asarray(rotations, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    return np.einsum("...ij,...j->...i", rotations, vectors)


def transform_points(rotation, translation, points) -> np.ndarray:
    """Apply the rigid transform `p -> R p + t` to points.

    Args:
        rotation: Rotation `(3, 3)`.
        translation: Translation `(3,)`.
        points: Points `(n, 3)`.

    Returns:
        Transformed points `(n, 3)`.
    """
    points = np.asarray(points, dtype=float)
    return points @ np.asarray(rotation, dtype=float).T + np.asarray(translation)
