import numpy as np
import pytest

from nb_touch.calibration.synthetic import (
    FINGER_AXES,
    finger_jacobian,
    model_torques,
    perturbed_orientations,
    random_cop_force,
    revolute_jacobian,
    synthetic_recording,
)
from nb_touch.contact.contacts import ContactSettings
from nb_touch.geometry.frames import normals_from_orientations
from nb_touch.geometry.synthetic import plane_sensor
from nb_touch.geometry.transforms import angles_between_rotations, rotate_vectors

UP = np.array([0.0, 0.0, 1.0])


def test_revolute_jacobian_columns():
    jacobian = revolute_jacobian(
        [[0.0, 0, 1], [1.0, 0, 0]], np.zeros((2, 3)), [1, 0, 0]
    )
    np.testing.assert_allclose(jacobian, [[0.0, 0], [1.0, 0], [0.0, 0]])


def test_point_on_axis_has_zero_column():
    jacobian = revolute_jacobian([[0.0, 0, 1]], [[0.0, 0, 0]], [0.0, 0, 2.0])
    np.testing.assert_allclose(jacobian, 0.0)


def test_finger_jacobian_sees_every_force_direction():
    jacobian = finger_jacobian(None, None, np.zeros(3))
    assert jacobian.shape == (3, len(FINGER_AXES))
    assert np.linalg.matrix_rank(jacobian) == 3


def test_perturbed_orientations_stay_within_angle():
    orientations = plane_sensor().orientations
    turned = perturbed_orientations(orientations, 0.3, np.random.default_rng(0))
    angles = angles_between_rotations(orientations, turned)
    assert (angles <= 0.3 + 1e-9).all() and (angles > 0).any()
    np.testing.assert_allclose(np.linalg.det(turned), 1.0)


def test_zero_perturbation_keeps_orientations():
    orientations = plane_sensor().orientations
    turned = perturbed_orientations(orientations, 0.0, np.random.default_rng(0))
    np.testing.assert_allclose(turned, orientations)


def test_random_cop_force_push_and_slip():
    rng = np.random.default_rng(1)
    for _ in range(20):
        force = random_cop_force(UP, (1.0, 2.0), 0.5, rng)
        assert -2.0 <= force[2] <= -1.0
        assert np.linalg.norm(force[:2]) <= -0.5 * force[2] + 1e-12


def test_recording_shapes():
    sensor = plane_sensor()
    recording = synthetic_recording(
        sensor, sensor.orientations, 10, np.random.default_rng(2)
    )
    assert recording.num_samples == 10
    assert recording.num_joints == len(FINGER_AXES)
    assert recording.num_taxels == sensor.num_taxels


def test_recording_forces_are_in_taxel_frames():
    sensor = plane_sensor()
    orientations = perturbed_orientations(
        sensor.orientations, 0.5, np.random.default_rng(3)
    )
    recording = synthetic_recording(sensor, orientations, 5, np.random.default_rng(4))
    sensor_forces = rotate_vectors(orientations, recording.taxel_forces_n[0])
    torques = model_torques(
        sensor.positions,
        normals_from_orientations(sensor.orientations),
        sensor_forces,
        ContactSettings(),
    )
    np.testing.assert_allclose(torques, recording.joint_torque_nm[0])


def test_model_torques_zero_without_contact():
    sensor = plane_sensor()
    normals = normals_from_orientations(sensor.orientations)
    forces = np.zeros((sensor.num_taxels, 3))
    np.testing.assert_allclose(
        model_torques(sensor.positions, normals, forces, ContactSettings()), 0.0
    )


@pytest.mark.parametrize("noise", ["force_noise", "torque_noise"])
def test_noise_changes_recording(noise):
    sensor = plane_sensor()
    clean = synthetic_recording(
        sensor, sensor.orientations, 5, np.random.default_rng(5)
    )
    noisy = synthetic_recording(
        sensor, sensor.orientations, 5, np.random.default_rng(5), **{noise: 0.01}
    )
    assert not np.allclose(clean.taxel_forces_n, noisy.taxel_forces_n) or not (
        np.allclose(clean.joint_torque_nm, noisy.joint_torque_nm)
    )
