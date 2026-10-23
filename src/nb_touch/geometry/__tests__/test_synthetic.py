import numpy as np
import pytest

from nb_touch.geometry.frames import normals_from_orientations
from nb_touch.geometry.synthetic import (
    SHAPES,
    cylinder_sensor,
    near_square_grid,
    plane_sensor,
    sphere_sensor,
    synthetic_sensor,
)


def test_plane_grid_shape_and_centre():
    sensor = plane_sensor(rows=2, cols=3, spacing=0.01)
    assert sensor.num_taxels == 6
    np.testing.assert_allclose(sensor.positions.mean(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(sensor.positions[:, 2], 0.0)


def test_cylinder_taxels_on_surface_with_radial_normals():
    sensor = cylinder_sensor(radius=0.01)
    radial = sensor.positions[:, [0, 2]]
    np.testing.assert_allclose(np.linalg.norm(radial, axis=1), 0.01)
    normals = normals_from_orientations(sensor.orientations)
    np.testing.assert_allclose(normals[:, [0, 2]], radial / 0.01, atol=1e-12)
    np.testing.assert_allclose(normals[:, 1], 0.0, atol=1e-12)


def test_cylinder_single_column_is_on_top():
    sensor = cylinder_sensor(cols=1)
    np.testing.assert_allclose(sensor.positions[:, 0], 0.0)


def test_sphere_taxels_on_cap_with_radial_normals():
    sensor = sphere_sensor(radius=0.02, count=30, cap=np.pi / 4)
    assert sensor.num_taxels == 30
    np.testing.assert_allclose(np.linalg.norm(sensor.positions, axis=1), 0.02)
    normals = normals_from_orientations(sensor.orientations)
    np.testing.assert_allclose(normals, sensor.positions / 0.02, atol=1e-12)
    assert np.all(normals[:, 2] >= np.cos(np.pi / 4) - 1e-12)


@pytest.mark.parametrize("make", [plane_sensor, cylinder_sensor, sphere_sensor])
def test_orientations_are_rotations(make):
    orientations = make().orientations
    np.testing.assert_allclose(
        orientations @ orientations.transpose(0, 2, 1),
        np.broadcast_to(np.eye(3), orientations.shape),
        atol=1e-12,
    )


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("count", [1, 2, 7, 25, 87])
def test_synthetic_sensor_has_exact_count(shape, count):
    sensor = synthetic_sensor(shape, count)
    assert sensor.num_taxels == count
    assert sensor.orientations.shape == (count, 3, 3)


def test_synthetic_cylinder_keeps_spacing_around_axis():
    sensor = synthetic_sensor("cylinder", 25, spacing=0.003, radius=0.01)
    # First row: taxels 0 and 1 are neighbours around the axis.
    arc_length = 0.01 * np.arccos(
        np.dot(sensor.positions[0, [0, 2]], sensor.positions[1, [0, 2]]) / 0.01**2
    )
    assert arc_length == pytest.approx(0.003)


def test_near_square_grid():
    assert near_square_grid(25) == (5, 5)
    assert near_square_grid(26) == (5, 6)
    assert near_square_grid(1) == (1, 1)


@pytest.mark.parametrize(("shape", "count"), [("cube", 10), ("plane", 0)])
def test_synthetic_sensor_rejects_bad_input(shape, count):
    with pytest.raises(ValueError):
        synthetic_sensor(shape, count)
