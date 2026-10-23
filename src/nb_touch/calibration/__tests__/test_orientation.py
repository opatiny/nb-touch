import numpy as np
import pytest
import torch

from nb_touch.calibration.orientation import (
    CalibrationSettings,
    calibrate_orientations,
    predict_torques,
    samples_to_tensors,
)
from nb_touch.calibration.samples import calibration_samples, sample_cop
from nb_touch.calibration.synthetic import (
    finger_jacobian,
    perturbed_orientations,
    synthetic_recording,
)
from nb_touch.contact.cop_model import (
    cop_normal,
    taxel_force_matrices,
    taxel_to_cop_force,
)
from nb_touch.geometry.frames import normals_from_orientations
from nb_touch.geometry.synthetic import cylinder_sensor, plane_sensor, sphere_sensor
from nb_touch.geometry.transforms import angles_between_rotations, rotate_vectors
from nb_touch.io.recordings import Recording
from nb_touch.io.sensor_geometry import SensorGeometry

PRIOR_ERROR = np.radians(20.0)
"""Largest angle between the prior and true orientations."""

SETTINGS = CalibrationSettings(steps=1000)
"""More steps than the paper's 100, to recover the orientations closely."""


def calibrated(sensor, num_samples=600, seed=0, **noise):
    """Calibrate on a synthetic recording, return the result and true orientations."""
    rng = np.random.default_rng(seed)
    true = perturbed_orientations(sensor.orientations, PRIOR_ERROR, rng)
    recording = synthetic_recording(sensor, true, num_samples, rng, **noise)
    return (
        calibrate_orientations(recording, sensor, finger_jacobian, None, SETTINGS),
        true,
    )


def errors_deg(result, true):
    return np.degrees(angles_between_rotations(result.geometry.orientations, true))


def test_predict_torques_matches_numpy_cop_force():
    sensor = plane_sensor()
    rng = np.random.default_rng(1)
    orientations = perturbed_orientations(sensor.orientations, 0.3, rng)
    recording = synthetic_recording(sensor, sensor.orientations, 1, rng)
    normals = normals_from_orientations(sensor.orientations)
    samples = calibration_samples(recording, sensor.positions, normals, finger_jacobian)
    predicted = predict_torques(
        torch.as_tensor(orientations), samples_to_tensors(samples)
    )

    taxels, cop, _ = sample_cop(
        sensor.positions, normals, recording.taxel_forces_n[0], SETTINGS.contact
    )
    normal = cop_normal(sensor.positions[taxels], normals[taxels], cop)
    sigma = SETTINGS.contact.from_geometry(sensor.positions).sigma
    matrices = taxel_force_matrices(
        sensor.positions[taxels], normals[taxels], cop, normal, sigma
    )
    forces = rotate_vectors(orientations[taxels], recording.taxel_forces_n[0][taxels])
    expected = -finger_jacobian(None, None, cop).T @ taxel_to_cop_force(
        forces, matrices
    )
    np.testing.assert_allclose(predicted[0].numpy(), expected)


@pytest.mark.parametrize("sensor", [plane_sensor(), cylinder_sensor(), sphere_sensor()])
def test_known_orientations_are_recovered(sensor):
    result, true = calibrated(sensor)
    errors = errors_deg(result, true)
    assert errors.max() < 3.0, f"max error {errors.max():.2f}°"
    assert result.losses[-1] < 1e-3 * result.losses[0]


def test_noise_still_recovers_orientations():
    result, true = calibrated(plane_sensor(), force_noise=0.01, torque_noise=1e-4)
    errors = errors_deg(result, true)
    assert np.median(errors) < 3.0, f"median error {np.median(errors):.2f}°"


def test_result_orientations_are_proper():
    result, _ = calibrated(sphere_sensor())
    orientations = result.geometry.orientations
    identity = np.broadcast_to(np.eye(3), orientations.shape)
    np.testing.assert_allclose(
        orientations @ orientations.transpose(0, 2, 1), identity, atol=1e-9
    )
    np.testing.assert_allclose(np.linalg.det(orientations), 1.0)


def test_losses_one_per_step_and_final():
    result, _ = calibrated(plane_sensor(), num_samples=50)
    assert result.losses.shape == (SETTINGS.steps + 1,)
    assert result.num_samples == 50


def test_never_active_taxel_keeps_prior():
    sensor = plane_sensor()
    rng = np.random.default_rng(2)
    true = perturbed_orientations(sensor.orientations, PRIOR_ERROR, rng)
    recording = synthetic_recording(sensor, true, 200, rng)
    forces = recording.taxel_forces_n.copy()
    forces[:, 0] = 0.0
    recording = Recording(
        recording.timestamps_s,
        recording.joint_pos_rad,
        recording.joint_torque_nm,
        forces,
    )
    result = calibrate_orientations(recording, sensor, finger_jacobian, None, SETTINGS)
    np.testing.assert_allclose(
        result.geometry.orientations[0], sensor.orientations[0], atol=1e-12
    )


def test_poorly_excited_joint_still_gives_rotations():
    # Without the abduction joint, forces along x leave no torque.
    def no_abduction(q, link, point):
        jacobian = finger_jacobian(q, link, point)
        jacobian[:, 0] = 0.0
        return jacobian

    sensor = plane_sensor()
    rng = np.random.default_rng(3)
    true = perturbed_orientations(sensor.orientations, PRIOR_ERROR, rng)
    recording = synthetic_recording(sensor, true, 300, rng)
    result = calibrate_orientations(recording, sensor, no_abduction, None, SETTINGS)
    assert np.isfinite(result.losses).all()
    assert result.losses[-1] < result.losses[0]
    np.testing.assert_allclose(np.linalg.det(result.geometry.orientations), 1.0)


def test_reflected_prior_gives_rotations():
    sensor = plane_sensor()
    rng = np.random.default_rng(4)
    recording = synthetic_recording(sensor, sensor.orientations, 100, rng)
    # Mirrored x axes, normals kept: det = -1.
    reflected = sensor.orientations @ np.diag([-1.0, 1.0, 1.0])
    prior = SensorGeometry(sensor.name, sensor.positions, reflected)
    result = calibrate_orientations(recording, prior, finger_jacobian, None, SETTINGS)
    np.testing.assert_allclose(np.linalg.det(result.geometry.orientations), 1.0)


def test_geometry_without_orientations_uses_estimated_prior():
    sensor = plane_sensor()
    rng = np.random.default_rng(5)
    true = perturbed_orientations(sensor.orientations, PRIOR_ERROR, rng)
    recording = synthetic_recording(sensor, true, 600, rng)
    bare = SensorGeometry(sensor.name, sensor.positions)
    result = calibrate_orientations(recording, bare, finger_jacobian, None, SETTINGS)
    assert result.geometry.orientations.shape == sensor.orientations.shape
    assert errors_deg(result, true).max() < 3.0


def test_taxel_count_mismatch_raises():
    sensor = plane_sensor()
    recording = synthetic_recording(
        sensor, sensor.orientations, 2, np.random.default_rng(6)
    )
    with pytest.raises(ValueError, match="taxels"):
        calibrate_orientations(recording, plane_sensor(rows=3, cols=3), finger_jacobian)
