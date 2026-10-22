"""The taxel-CoP mapping of Pan et al. 2026, Section 3.2: from a CoP force to taxel
forces and back. See `docs/theory/cop.md`.

All vectors are in the sensor frame: taxel forces must be rotated from their taxel
frames first (`rotate_vectors`). Forces act on the sensor, so a push points against
the outward normals; the paper's inward normals are their opposites. See
Conventions in `README.md`.
"""

import numpy as np

from nb_touch.contact.active import taxel_norms

SIGMA_M = 0.003
"""Spread σ of the Gaussian weights, in m. Our choice, about one taxel spacing on
the ORCA sensors: the paper gives no value."""

REGULARISATION = 1e-3
"""Ridge regularisation λ of the CoP force solve, no unit. Our choice: the paper
gives no value."""

COINCIDENT_M = 1e-12
"""Distance below which a taxel counts as being at the CoP, in m."""


def unit(vectors) -> np.ndarray:
    """Normalise vectors `(..., 3)`, leaving zero vectors at zero."""
    vectors = np.asarray(vectors, dtype=float)
    lengths = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, lengths, out=np.zeros_like(vectors), where=lengths > 0)


def cop_position(positions, forces) -> np.ndarray:
    """Return the CoP: taxel positions averaged with their force norms as weights.

    Args:
        positions: Taxel positions `(n, 3)`, in m.
        forces: Taxel forces `(n, 3)`, in N.

    Raises:
        ValueError: If every force is zero, which leaves no CoP.
    """
    norms = taxel_norms(forces)
    if norms.sum() == 0.0:
        raise ValueError("every force is zero: no CoP")
    return norms @ np.asarray(positions, dtype=float) / norms.sum()


def cop_normal(positions, normals, cop) -> np.ndarray:
    """Return the surface normal at the CoP, by inverse distance weighting of the
    taxel normals, weights `1 / ||p_i - p_cop||` (Shepard 1968).

    Args:
        positions: Taxel positions `(n, 3)`, in m.
        normals: Unit outward taxel normals `(n, 3)`.
        cop: CoP position `(3,)`, in m.

    Returns:
        The unit normal `(3,)`: the mean normal of the taxels at the CoP if any, NaN
        when the weighted normals cancel out.
    """
    normals = np.asarray(normals, dtype=float)
    distances = np.linalg.norm(np.asarray(positions, dtype=float) - cop, axis=1)
    at_cop = distances < COINCIDENT_M
    if at_cop.any():
        total = normals[at_cop].sum(axis=0)
    else:
        total = (1.0 / distances) @ normals
    length = np.linalg.norm(total)
    return total / length if length > 0.0 else np.full(3, np.nan)


def gaussian_weights(positions, cop, sigma: float = SIGMA_M) -> np.ndarray:
    """Return `w_i = exp(-||p_i - p_cop||^2 / (2 σ^2))` for every taxel, `(n,)`.

    Raises:
        ValueError: If `sigma` is not positive.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    offsets = np.asarray(positions, dtype=float) - cop
    return np.exp(-np.sum(offsets**2, axis=1) / (2.0 * sigma**2))


def blended_directions(positions, normals, cop, weights) -> np.ndarray:
    """Return `b_i = normalise(w_i n_i + (1 - w_i) v_i)`, with `v_i` the unit vector
    from the CoP to taxel `i`: the direction of each taxel's normal force.

    Args:
        positions: Taxel positions `(n, 3)`, in m.
        normals: Unit inward taxel normals `(n, 3)`, as in the paper.
        cop: CoP position `(3,)`, in m.
        weights: Gaussian weights `(n,)`.

    Returns:
        Unit directions `(n, 3)`. Where the blend cancels out, the taxel normal.
    """
    normals = np.asarray(normals, dtype=float)
    weights = np.asarray(weights, dtype=float)[:, None]
    away = unit(np.asarray(positions, dtype=float) - cop)
    blended = unit(weights * normals + (1.0 - weights) * away)
    cancelled = np.linalg.norm(blended, axis=1) == 0.0
    blended[cancelled] = normals[cancelled]
    return blended


def shear_projector(normal) -> np.ndarray:
    """Return `P = I - n n^T` `(3, 3)`, projecting onto the plane across `normal`."""
    normal = np.asarray(normal, dtype=float)
    return np.eye(3) - np.outer(normal, normal)


def taxel_force_matrices(
    positions, normals, cop, normal, sigma: float = SIGMA_M
) -> np.ndarray:
    """Return `M_i = w_i (b_i n_cop^T + P_shear)` for every taxel, so `f_i = M_i f_cop`.

    `n_i` and `n_cop` are the paper's inward normals, the opposites of `normals` and
    `normal`.

    Args:
        positions: Taxel positions `(n, 3)`, in m.
        normals: Unit outward taxel normals `(n, 3)`.
        cop: CoP position `(3,)`, in m.
        normal: Unit outward normal at the CoP `(3,)`.
        sigma: Spread of the Gaussian weights, in m.

    Returns:
        Matrices `(n, 3, 3)`.
    """
    inward = -np.asarray(normal, dtype=float)
    weights = gaussian_weights(positions, cop, sigma)
    directions = blended_directions(
        positions, -np.asarray(normals, dtype=float), cop, weights
    )
    normal_part = directions[:, :, None] * inward[None, None]
    return weights[:, None, None] * (normal_part + shear_projector(inward))


def cop_to_taxel_forces(cop_force, matrices) -> np.ndarray:
    """Return the taxel forces `f_i = M_i f_cop`, `(n, 3)` in N.

    Args:
        cop_force: CoP force `(3,)`, in N.
        matrices: `M_i` `(n, 3, 3)`, from `taxel_force_matrices`.
    """
    return np.asarray(matrices, dtype=float) @ np.asarray(cop_force, dtype=float)


def cop_force_operator(matrices, regularisation: float = REGULARISATION) -> np.ndarray:
    """Return the ridge solve `K = (A^T A + λ^2 I)^-1 A^T`, with `A` stacking the
    `M_i`, split per taxel: `f_cop = sum_i K_i f_i`.

    Args:
        matrices: `M_i` `(n, 3, 3)`, from `taxel_force_matrices`.
        regularisation: λ. With 0, the pseudo-inverse of `A`, giving the
            minimum-norm least-squares solution even when `A^T A` is singular.

    Returns:
        `K_i` `(n, 3, 3)`.
    """
    stacked = np.asarray(matrices, dtype=float).reshape(-1, 3)
    if regularisation == 0:
        operator = np.linalg.pinv(stacked)
    else:
        gram = stacked.T @ stacked + regularisation**2 * np.eye(3)
        operator = np.linalg.solve(gram, stacked.T)
    return operator.reshape(3, -1, 3).transpose(1, 0, 2)


def taxel_to_cop_force(
    forces, matrices, regularisation: float = REGULARISATION
) -> np.ndarray:
    """Return the CoP force by ridge least squares, `(A^T A + λ^2 I)^-1 A^T b`, with
    `A` stacking the `M_i` and `b` the taxel forces.

    Args:
        forces: Taxel forces `(n, 3)`, in N.
        matrices: `M_i` `(n, 3, 3)`, from `taxel_force_matrices`.
        regularisation: λ. With 0, the minimum-norm least-squares solution, which
            exists even when `A^T A` is singular.

    Returns:
        The CoP force `(3,)`, in N. Zero when there is no taxel.
    """
    operators = cop_force_operator(matrices, regularisation)
    return np.einsum("nij,nj->i", operators, np.asarray(forces, dtype=float))
