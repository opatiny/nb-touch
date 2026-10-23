import numpy as np
import pytest

from nb_touch.calibration.samples import (
    calibration_samples,
    checked_jacobian,
    sample_cop,
)
from nb_touch.calibration.synthetic import finger_jacobian, synthetic_recording
from nb_touch.contact.contacts import ContactSettings
from nb_touch.geometry.frames import normals_from_orientations
from nb_touch.geometry.synthetic import plane_sensor
from nb_touch.io.recordings import Recording

SENSOR = plane_sensor()
NORMALS = normals_from_orientations(SENSOR.orientations)


def recording_with_forces(forces) -> Recording:
    """A recording of the given taxel forces `(N, T, 3)`, 4 joints at rest."""
    count = len(forces)
    return Recording(
        timestamps_s=np.arange(count),
        joint_pos_rad=np.zeros((count, 4)),
        joint_torque_nm=np.zeros((count, 4)),
        taxel_forces_n=forces,
    )


def test_checked_jacobian_shape():
    assert checked_jacobian(np.zeros((3, 4)), 4).shape == (3, 4)
    with pytest.raises(ValueError, match=r"\(3, 4\)"):
        checked_jacobian(np.zeros((4, 3)), 4)


def test_sample_without_contact_is_skipped():
    forces = np.zeros((SENSOR.num_taxels, 3))
    assert sample_cop(SENSOR.positions, NORMALS, forces, ContactSettings()) is None


def test_sample_below_min_taxels_is_skipped():
    forces = np.zeros((SENSOR.num_taxels, 3))
    forces[12] = [0.0, 0.0, 1.0]
    settings = ContactSettings(min_taxels=2)
    assert sample_cop(SENSOR.positions, NORMALS, forces, settings) is None


def test_single_active_taxel_gives_its_force():
    forces = np.zeros((SENSOR.num_taxels, 3))
    forces[12] = [0.1, 0.0, 1.0]
    taxels, cop, operators = sample_cop(
        SENSOR.positions, NORMALS, forces, ContactSettings(min_taxels=1)
    )
    np.testing.assert_array_equal(taxels, [12])
    np.testing.assert_allclose(cop, SENSOR.positions[12])
    np.testing.assert_allclose(operators[0] @ forces[12], forces[12], rtol=1e-5)


def test_samples_skip_inactive_and_keep_indices_aligned():
    rng = np.random.default_rng(0)
    recording = synthetic_recording(SENSOR, SENSOR.orientations, 3, rng)
    forces = recording.taxel_forces_n.copy()
    forces[1] = 0.0
    samples = calibration_samples(
        recording_with_forces(forces), SENSOR.positions, NORMALS, finger_jacobian
    )
    assert samples.num_samples == 2
    assert set(samples.sample_index) == {0, 1}
    assert len(samples.taxel_index) == len(samples.operators) == len(samples.forces)
    recorded_sample = np.array([0, 2])[samples.sample_index]
    np.testing.assert_allclose(
        samples.forces, forces[recorded_sample, samples.taxel_index]
    )


def test_samples_reproduce_recorded_torques():
    rng = np.random.default_rng(1)
    recording = synthetic_recording(SENSOR, SENSOR.orientations, 4, rng)
    samples = calibration_samples(recording, SENSOR.positions, NORMALS, finger_jacobian)
    parts = np.einsum("eij,ej->ei", samples.operators, samples.forces)
    cop_forces = np.zeros((samples.num_samples, 3))
    np.add.at(cop_forces, samples.sample_index, parts)
    torques = -np.einsum("sij,si->sj", samples.jacobians, cop_forces)
    np.testing.assert_allclose(torques, recording.joint_torque_nm)


def test_jacobian_gets_angles_and_link():
    recording = synthetic_recording(
        SENSOR, SENSOR.orientations, 2, np.random.default_rng(2)
    )
    calls = []

    def mock_jacobian(q, link, point):
        calls.append((q, link))
        return finger_jacobian(q, link, point)

    calibration_samples(recording, SENSOR.positions, NORMALS, mock_jacobian, "tip")
    assert [link for _, link in calls] == ["tip", "tip"]
    np.testing.assert_allclose(calls[0][0], recording.joint_pos_rad[0])


def test_no_active_sample_raises():
    recording = recording_with_forces(np.zeros((2, SENSOR.num_taxels, 3)))
    with pytest.raises(ValueError, match="no sample"):
        calibration_samples(recording, SENSOR.positions, NORMALS, finger_jacobian)
