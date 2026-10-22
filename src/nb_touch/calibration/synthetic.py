"""Synthetic finger and recordings with known taxel rotations, for tests and demos.

The finger is held still, as in the paper's data collection (a stiff PD controller
holds the joints while an object is pressed on the fingertip), so its Jacobian does
not depend on the joint angles.
"""

import numpy as np

from nb_touch.calibration.samples import sample_cop
from nb_touch.contact.contacts import ContactSettings
from nb_touch.contact.cop_model import (
    cop_normal,
    cop_to_taxel_forces,
    taxel_force_matrices,
)
from nb_touch.geometry.frames import normals_from_rotations
from nb_touch.geometry.transforms import rotate_vectors, rotations_from_vectors
from nb_touch.io.recordings import Recording
from nb_touch.io.sensor_geometry import SensorGeometry

FINGER_AXES = np.array(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
)
"""Unit joint axes of the synthetic 4-joint finger, in the sensor frame: abduction,
then three flexion joints, like a human finger."""

FINGER_ORIGINS = np.array(
    [
        [0.0, -0.1, -0.005],
        [0.0, -0.1, -0.005],
        [0.0, -0.055, -0.005],
        [0.0, -0.02, -0.005],
    ]
)
"""Points on the joint axes, in the sensor frame, in m: the finger runs along -y
from the sensor."""

RATE_HZ = 20.0
"""Sampling rate of the synthetic recordings, as in the paper (Appendix A)."""


def revolute_jacobian(axes, origins, point) -> np.ndarray:
    """Return the position Jacobian `(3, J)` of `point` for revolute joints: column
    `j` is `a_j x (p - o_j)`.

    Args:
        axes: Unit joint axes `(J, 3)`.
        origins: A point on each axis `(J, 3)`, in m.
        point: The point `(3,)`, in m, in the same frame.
    """
    offsets = np.asarray(point, dtype=float) - np.asarray(origins, dtype=float)
    return np.cross(np.asarray(axes, dtype=float), offsets).T


def finger_jacobian(q, link, point) -> np.ndarray:
    """Jacobian `(3, 4)` of the synthetic finger at `point`, in the sensor frame.

    `q` and `link` are ignored: the finger is held still.
    """
    return revolute_jacobian(FINGER_AXES, FINGER_ORIGINS, point)


def perturbed_rotations(rotations, max_angle: float, rng) -> np.ndarray:
    """Return each rotation `(n, 3, 3)` turned about a random axis of its own frame,
    by a random angle up to `max_angle`, in rad.
    """
    count = len(rotations)
    axes = rng.normal(size=(count, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = rng.uniform(0.0, max_angle, size=(count, 1))
    return np.asarray(rotations) @ rotations_from_vectors(axes * angles)


def random_cop_force(normal, normal_range, max_slip: float, rng) -> np.ndarray:
    """Return a push against the outward `normal` of norm in `normal_range`, in N,
    plus a shear across it in a random direction, up to `max_slip` times the push."""
    push = rng.uniform(*normal_range)
    shear = rng.normal(size=3)
    shear -= (shear @ normal) * normal
    shear *= rng.uniform(0.0, max_slip) * push / np.linalg.norm(shear)
    return -push * normal + shear


def model_torques(positions, normals, forces, settings: ContactSettings):
    """Return the joint torques `-J^T f_cop` `(4,)` of the synthetic finger, with the
    CoP and CoP force estimated from the taxel forces, as the calibration does.

    Args:
        positions: Taxel positions `(T, 3)`, in m.
        normals: Unit outward taxel normals `(T, 3)`.
        forces: Taxel forces `(T, 3)`, in the sensor frame, in N.
        settings: Active taxels and CoP model settings.

    Returns:
        The torques, zero if the sample has too few active taxels.
    """
    found = sample_cop(positions, normals, forces, settings)
    if found is None:
        return np.zeros(len(FINGER_AXES))
    taxels, cop, operators = found
    cop_force = np.einsum("nij,nj->i", operators, forces[taxels])
    return -finger_jacobian(None, None, cop).T @ cop_force


def synthetic_recording(
    geometry: SensorGeometry,
    rotations,
    num_samples: int,
    rng,
    normal_range=(0.5, 3.0),
    max_slip: float = 0.5,
    force_noise: float = 0.0,
    torque_noise: float = 0.0,
    settings: ContactSettings = ContactSettings(),
) -> Recording:
    """Simulate random contacts on the fingertip of the synthetic finger.

    Each sample presses at a random taxel: the CoP model turns the CoP force into
    taxel forces with the geometry's normals, and `rotations` take them to the
    taxel frames. The joint torques, `-J^T f_cop`, come from the CoP and CoP force
    estimated back from the taxel forces (`model_torques`), so with the true
    rotations the calibration's model explains them exactly, without the bias of
    the estimate (`docs/theory/cop.md`, Known bias).

    Args:
        geometry: Sensor geometry with rotations, giving the normals.
        rotations: True taxel rotations `(T, 3, 3)`, taxel frame to sensor frame.
        num_samples: Number of samples N.
        rng: A `numpy.random.Generator`.
        normal_range: Smallest and largest push, in N.
        max_slip: Largest shear over push.
        force_noise: Standard deviation of the noise on each force component, N.
        torque_noise: Standard deviation of the noise on each torque, in N·m.
        settings: Active taxels and CoP model settings, the same as the
            calibration's.
    """
    positions = geometry.positions
    normals = normals_from_rotations(geometry.rotations)
    forces, torques = [], []
    for taxel in rng.integers(len(positions), size=num_samples):
        cop = positions[taxel]
        normal = cop_normal(positions, normals, cop)
        cop_force = random_cop_force(normal, normal_range, max_slip, rng)
        matrices = taxel_force_matrices(positions, normals, cop, normal, settings.sigma)
        sensor_forces = cop_to_taxel_forces(cop_force, matrices)
        forces.append(rotate_vectors(np.transpose(rotations, (0, 2, 1)), sensor_forces))
        torques.append(model_torques(positions, normals, sensor_forces, settings))
    forces, torques = np.array(forces), np.array(torques)
    return Recording(
        timestamps_s=np.arange(num_samples) / RATE_HZ,
        joint_pos_rad=np.zeros_like(torques),
        joint_torque_nm=torques + rng.normal(0.0, torque_noise, torques.shape),
        taxel_forces_n=forces + rng.normal(0.0, force_noise, forces.shape),
    )
