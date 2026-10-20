import numpy as np
import pytest

from nb_touch.geometry.frames import (
    normals_from_rotations,
    rotation_from_z,
    rotations_from_normals,
    taxel_rotations,
    with_estimated_rotations,
)
from nb_touch.geometry.synthetic import plane_sensor, sphere_sensor
from nb_touch.io.sensor_geometry import SensorGeometry


@pytest.mark.parametrize(
    "normal",
    [
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.6, 0.8],
        [0.6, 0.0, -0.8],
    ],
)
def test_rotation_from_z_is_rotation_with_z_on_normal(normal):
    rotation = rotation_from_z(normal)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(rotation[:, 2], normal, atol=1e-12)


def test_rotation_from_z_of_z_is_identity():
    np.testing.assert_allclose(rotation_from_z([0.0, 0.0, 1.0]), np.eye(3))


def test_normals_round_trip_through_rotations():
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.6, 0.8]])
    np.testing.assert_allclose(
        normals_from_rotations(rotations_from_normals(normals)), normals, atol=1e-12
    )


def test_no_normals_give_empty_rotations():
    assert rotations_from_normals(np.zeros((0, 3))).shape == (0, 3, 3)


def test_taxel_rotations_on_sphere_close_to_true_normals():
    sensor = sphere_sensor()
    estimated = normals_from_rotations(taxel_rotations(sensor.positions))
    true = normals_from_rotations(sensor.rotations)
    assert np.all(np.sum(estimated * true, axis=1) > 0.9)


def test_with_estimated_rotations_fills_missing():
    geometry = SensorGeometry("s", plane_sensor().positions)
    completed = with_estimated_rotations(geometry)
    np.testing.assert_allclose(completed.rotations, plane_sensor().rotations, atol=1e-9)


def test_with_estimated_rotations_keeps_given():
    geometry = sphere_sensor()
    assert with_estimated_rotations(geometry) is geometry
