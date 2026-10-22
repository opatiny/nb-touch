"""Projection of 3×3 matrices onto rotations, see `docs/theory/so3-projection.md`."""

import numpy as np
import torch

DEGENERATE = 1e-12
"""Sum of two singular values below which the projection's gradient is set to 0."""


def signed_svd(matrices: torch.Tensor):
    """Return `U, S, V^T` of `A = U S V^T`, with the last column of `U` and last
    singular value negated where `det(U V^T) = -1`, so `U V^T` is a rotation.
    """
    u, s, vt = torch.linalg.svd(matrices)
    signs = torch.sign(torch.linalg.det(u @ vt))
    # det is ±1 here; a 0 can only come from rounding, treat it as +1.
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    flip = torch.ones_like(s)
    flip[..., 2] = signs
    return u * flip[..., None, :], s * flip, vt


class ProjectToSO3(torch.autograd.Function):
    """`R = U V^T` from `signed_svd`, with the polar factor's derivative as
    gradient: `dL/dA = U [(H - H^T) / (s_i + s_j)] V^T`, `H = U^T (dL/dR) V`.

    Torch's own SVD gradient divides by `s_i^2 - s_j^2` and is NaN at a rotation,
    where all singular values are 1. This one stays finite there.
    """

    @staticmethod
    def forward(ctx, matrices):
        u, s, vt = signed_svd(matrices)
        ctx.save_for_backward(u, s, vt)
        return u @ vt

    @staticmethod
    def backward(ctx, grad):
        u, s, vt = ctx.saved_tensors
        h = u.transpose(-1, -2) @ grad @ vt.transpose(-1, -2)
        sums = s[..., :, None] + s[..., None, :]
        safe = torch.where(sums.abs() > DEGENERATE, sums, torch.ones_like(sums))
        omega = torch.where(
            sums.abs() > DEGENERATE, (h - h.transpose(-1, -2)) / safe, 0.0
        )
        return u @ omega @ vt


def project_to_so3_torch(matrices: torch.Tensor) -> torch.Tensor:
    """Return the closest rotation to each 3×3 matrix, differentiable with autograd.

    SVD `A = U S V^T`, then `R = U V^T`, flipping the last column of `U` where
    `det(R) = -1` so the result is a rotation, not a mirror (Umeyama, IEEE TPAMI
    13(4), 1991).

    Args:
        matrices: One matrix `(3, 3)` or a stack `(..., 3, 3)`.

    Returns:
        Rotations with the same shape: orthonormal, determinant +1.
    """
    return ProjectToSO3.apply(matrices)


def project_to_so3(matrices) -> np.ndarray:
    """Return the closest rotation to each 3×3 matrix, see `project_to_so3_torch`.

    Args:
        matrices: One matrix `(3, 3)` or a stack `(..., 3, 3)`.

    Returns:
        Rotations with the same shape: orthonormal, determinant +1.
    """
    matrices = torch.as_tensor(np.asarray(matrices, dtype=float))
    return project_to_so3_torch(matrices).numpy()
