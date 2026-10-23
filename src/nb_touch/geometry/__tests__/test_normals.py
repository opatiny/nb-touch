import numpy as np
import pytest

from nb_touch.geometry.frames import normals_from_orientations
from nb_touch.geometry.normals import (
    nearest_neighbours,
    pairwise_distances,
    plane_normals,
    taxel_normals,
)
from nb_touch.geometry.synthetic import cylinder_sensor, plane_sensor, sphere_sensor


def test_pairwise_distances():
    distances = pairwise_distances([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    np.testing.assert_allclose(distances, [[0.0, 5.0], [5.0, 0.0]])


def test_nearest_neighbours_itself_first_and_clamped():
    points = [[0.0, 0, 0], [1.0, 0, 0], [3.0, 0, 0]]
    np.testing.assert_array_equal(
        nearest_neighbours(points, 10), [[0, 1, 2], [1, 0, 2], [2, 1, 0]]
    )


def test_plane_normals_are_unit_and_along_z_on_a_plane():
    normals = plane_normals(plane_sensor().positions)
    np.testing.assert_allclose(np.abs(normals[:, 2]), 1.0)


@pytest.mark.parametrize(
    ("sensor", "tolerance"),
    [(plane_sensor(), 1e-9), (cylinder_sensor(), 0.1), (sphere_sensor(), 0.1)],
)
def test_taxel_normals_match_known_normals(sensor, tolerance):
    expected = normals_from_orientations(sensor.orientations)
    normals = taxel_normals(sensor.positions)
    # Outward everywhere, and close to the true normal.
    assert np.all(np.sum(normals * expected, axis=1) > 1.0 - tolerance)


def test_normals_point_outward_even_when_fit_flips_them():
    sensor = sphere_sensor()
    normals = taxel_normals(sensor.positions)
    assert np.all(np.sum(normals * sensor.positions, axis=1) > 0)


def test_fewer_than_three_taxels_raise():
    with pytest.raises(ValueError, match="at least 3"):
        taxel_normals([[0.0, 0, 0], [1.0, 0, 0]])
