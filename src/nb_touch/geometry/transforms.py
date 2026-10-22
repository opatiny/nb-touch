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


def rotations_from_vectors(rotation_vectors) -> np.ndarray:
    """Return the rotation of each rotation vector (axis times angle in rad).

    Rodrigues' formula, `R = I + sin θ K + (1 - cos θ) K^2`, with `K` the skew matrix
    of the unit axis.

    Args:
        rotation_vectors: Rotation vectors `(n, 3)`.

    Returns:
        Rotations `(n, 3, 3)`. The identity for a zero vector.
    """
    vectors = np.asarray(rotation_vectors, dtype=float).reshape(-1, 3)
    angles = np.linalg.norm(vectors, axis=1)
    axes = np.divide(
        vectors, angles[:, None], out=np.zeros_like(vectors), where=angles[:, None] > 0
    )
    skew = np.zeros((len(vectors), 3, 3))
    skew[:, 0, 1], skew[:, 0, 2], skew[:, 1, 2] = -axes[:, 2], axes[:, 1], -axes[:, 0]
    skew -= skew.transpose(0, 2, 1)
    sines, cosines = np.sin(angles)[:, None, None], np.cos(angles)[:, None, None]
    return np.eye(3) + sines * skew + (1.0 - cosines) * skew @ skew


def angles_between_rotations(first, second) -> np.ndarray:
    """Return the angle of the rotation taking each `first` to its `second`, in rad.

    Args:
        first: Rotations `(n, 3, 3)`.
        second: Rotations `(n, 3, 3)`.

    Returns:
        Angles `(n,)` in `[0, π]`, from `cos θ = (trace(R1^T R2) - 1) / 2`.
    """
    relative = np.swapaxes(np.asarray(first, dtype=float), -1, -2) @ second
    cosines = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(cosines, -1.0, 1.0))
