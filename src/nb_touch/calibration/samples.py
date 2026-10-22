"""Everything of a recording the taxel orientation calibration needs, computed once:
the parts of the CoP model that do not depend on the taxel rotations.

The CoP position uses the force norms only and the normals come from the prior
rotations, so the CoP, the Jacobian at it and the ridge solve are fixed: only the
taxel forces in the sensor frame, `R_i f_i`, change with the rotations. See
`docs/theory/taxel-orientation.md`.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from nb_touch.contact.active import active_taxels
from nb_touch.contact.contacts import ContactSettings
from nb_touch.contact.cop_model import (
    cop_force_operator,
    cop_normal,
    cop_position,
    taxel_force_matrices,
)
from nb_touch.io.recordings import Recording

Jacobian = Callable[[np.ndarray, str | None, np.ndarray], np.ndarray]
"""`jacobian(q, link, point) -> (3, J)`: position Jacobian of `point` (sensor frame,
in m) on `link` at joint angles `q` `(J,)`, expressed in the sensor frame."""


@dataclass(frozen=True)
class CalibrationSamples:
    """The samples used by the calibration, E active taxels over S samples.

    Attributes:
        sample_index: `(E,)` sample of each active taxel, in `0..S-1`.
        taxel_index: `(E,)` taxel of each active taxel.
        operators: `(E, 3, 3)` ridge solve `K_i`, so the CoP force of a sample is
            the sum of `K_i R_i f_i` over its active taxels.
        forces: `(E, 3)` taxel forces `f_i`, in their taxel frames, in N.
        jacobians: `(S, 3, J)` Jacobians at the CoPs, in the sensor frame.
        torques: `(S, J)` recorded joint torques, in N·m.
    """

    sample_index: np.ndarray
    taxel_index: np.ndarray
    operators: np.ndarray
    forces: np.ndarray
    jacobians: np.ndarray
    torques: np.ndarray

    @property
    def num_samples(self) -> int:
        """Number of samples S used."""
        return len(self.torques)


def checked_jacobian(jacobian, num_joints: int) -> np.ndarray:
    """Return `jacobian` as an array, raising a `ValueError` unless it is `(3, J)`."""
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.shape != (3, num_joints):
        raise ValueError(f"expected a (3, {num_joints}) Jacobian, got {jacobian.shape}")
    return jacobian


def sample_cop(positions, normals, forces, settings: ContactSettings):
    """Return the active taxels, CoP and ridge solve of one sample.

    Args:
        positions: Taxel positions `(T, 3)`, in m.
        normals: Unit outward prior normals `(T, 3)`.
        forces: Taxel forces `(T, 3)`, in their taxel frames, in N.
        settings: `threshold`, `min_taxels`, `sigma` and `regularisation` are used.

    Returns:
        `(taxels, cop, operators)`, or `None` if fewer than `min_taxels` taxels are
        active or the normal at the CoP is undefined.
    """
    settings = settings.from_geometry(positions)
    taxels = np.flatnonzero(active_taxels(forces, settings.threshold))
    if len(taxels) < max(settings.min_taxels, 1):
        return None
    points, zone_normals = positions[taxels], normals[taxels]
    cop = cop_position(points, forces[taxels])
    normal = cop_normal(points, zone_normals, cop)
    if np.isnan(normal).any():
        return None
    matrices = taxel_force_matrices(points, zone_normals, cop, normal, settings.sigma)
    return taxels, cop, cop_force_operator(matrices, settings.regularisation)


def calibration_samples(
    recording: Recording,
    positions,
    normals,
    jacobian: Jacobian,
    link: str | None = None,
    settings: ContactSettings = ContactSettings(),
) -> CalibrationSamples:
    """Prepare the samples of a recording for the calibration.

    Args:
        recording: Taxel forces, joint angles and joint torques.
        positions: Taxel positions `(T, 3)`, in m.
        normals: Unit outward prior normals `(T, 3)`.
        jacobian: See `Jacobian`.
        link: Link the sensor is mounted on, passed to `jacobian`.
        settings: Active taxels and CoP model settings.

    Raises:
        ValueError: If no sample has enough active taxels.
    """
    positions = np.asarray(positions, dtype=float)
    normals = np.asarray(normals, dtype=float)
    settings = settings.from_geometry(positions)
    entries, jacobians, torques = [], [], []
    for forces, q, tau in zip(
        recording.taxel_forces_n,
        recording.joint_pos_rad,
        recording.joint_torque_nm,
        strict=True,
    ):
        found = sample_cop(positions, normals, forces, settings)
        if found is None:
            continue
        taxels, cop, operators = found
        entries.append((len(torques), taxels, operators, forces[taxels]))
        jacobians.append(checked_jacobian(jacobian(q, link, cop), len(q)))
        torques.append(tau)
    if not torques:
        raise ValueError("no sample has enough active taxels")
    return CalibrationSamples(
        sample_index=np.concatenate([np.full(len(e[1]), e[0]) for e in entries]),
        taxel_index=np.concatenate([e[1] for e in entries]),
        operators=np.concatenate([e[2] for e in entries]),
        forces=np.concatenate([e[3] for e in entries]),
        jacobians=np.array(jacobians),
        torques=np.array(torques),
    )
