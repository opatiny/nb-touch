"""Projection of 3×3 matrices onto rotations, see `docs/theory/so3-projection.md`."""

import numpy as np


def project_to_so3(matrices) -> np.ndarray:
    """Return the closest rotation to each 3×3 matrix.

    SVD `A = U S V^T`, then `R = U V^T`, flipping the last column of `U` where
    `det(R) = -1` so the result is a rotation, not a mirror (Umeyama, IEEE TPAMI
    13(4), 1991).

    Args:
        matrices: One matrix `(3, 3)` or a stack `(..., 3, 3)`.

    Returns:
        Rotations with the same shape: orthonormal, determinant +1.
    """
    matrices = np.asarray(matrices, dtype=float)
    u, _, vt = np.linalg.svd(matrices)
    signs = np.sign(np.linalg.det(u @ vt))
    # det is ±1 here; a 0 can only come from rounding, treat it as +1.
    signs = np.where(signs == 0, 1.0, signs)
    u = u.copy()
    u[..., :, 2] *= signs[..., None]
    return u @ vt
