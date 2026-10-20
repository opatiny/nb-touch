import numpy as np
import pytest

from nb_touch.geometry.frames import normals_from_rotations
from nb_touch.geometry.synthetic import cylinder_sensor, plane_sensor, sphere_sensor


def test_plane_grid_shape_and_centre():
    sensor = plane_sensor(rows=2, cols=3, spacing=0.01)
    assert sensor.num_taxels == 6
    np.testing.assert_allclose(sensor.positions.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(sensor.positions[:, 2], 0.0)


def test_cylinder_taxels_on_surface_with_radial_normals():
    sensor = cylinder_sensor(radius=0.01)
    radial = sensor.positions[:, [0, 2]]
    np.testing.assert_allclose(np.linalg.norm(radial, axis=1), 0.01)
    normals = normals_from_rotations(sensor.rotations)
    np.testing.assert_allclose(normals[:, [0, 2]], radial / 0.01, atol=1e-12)
    np.testing.assert_allclose(normals[:, 1], 0.0, atol=1e-12)


def test_cylinder_single_column_is_on_top():
    sensor = cylinder_sensor(cols=1)
    np.testing.assert_allclose(sensor.positions[:, 0], 0.0)


def test_sphere_taxels_on_cap_with_radial_normals():
    sensor = sphere_sensor(radius=0.02, count=30, cap=np.pi / 4)
    assert sensor.num_taxels == 30
    np.testing.assert_allclose(np.linalg.norm(sensor.positions, axis=1), 0.02)
    normals = normals_from_rotations(sensor.rotations)
    np.testing.assert_allclose(normals, sensor.positions / 0.02, atol=1e-12)
    assert np.all(normals[:, 2] >= np.cos(np.pi / 4) - 1e-12)


@pytest.mark.parametrize("make", [plane_sensor, cylinder_sensor, sphere_sensor])
def test_rotations_are_rotations(make):
    rotations = make().rotations
    np.testing.assert_allclose(
        rotations @ rotations.transpose(0, 2, 1),
        np.broadcast_to(np.eye(3), rotations.shape),
        atol=1e-12,
    )
