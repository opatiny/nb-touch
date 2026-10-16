# Projection to SO(3)

`project_to_so3` takes any 3×3 matrix and returns the closest rotation matrix: orthonormal, with determinant +1.

## How

1. Singular value decomposition: $A = U S V^T$.
2. $R = U V^T$.
3. If $\det(R) = -1$, flip the sign of the last column of $U$ and recompute $R$, so the result is a rotation and not a mirror.

Source: Umeyama, "Least-squares estimation of transformation parameters between two point patterns", IEEE TPAMI 13(4), 1991.

## Why

The [calibration](taxel-orientation.md) solver treats each $R_i$ as 9 free numbers, so its result is close to a rotation but not exactly one. The paper projects it the same way ([arXiv 2605.28812](https://arxiv.org/abs/2605.28812)).
