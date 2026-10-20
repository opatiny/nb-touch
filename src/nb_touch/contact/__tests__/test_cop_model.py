import numpy as np
import pytest

from nb_touch.contact.cop_model import (
    blended_directions,
    cop_normal,
    cop_position,
    cop_to_taxel_forces,
    gaussian_weights,
    shear_projector,
    taxel_force_matrices,
    taxel_to_cop_force,
    unit,
)
from nb_touch.geometry.frames import normals_from_rotations
from nb_touch.geometry.synthetic import cylinder_sensor, plane_sensor, sphere_sensor

PLANE = plane_sensor(rows=5, cols=5, spacing=0.003)
PLANE_NORMALS = normals_from_rotations(PLANE.rotations)
UP = np.array([0.0, 0.0, 1.0])


def test_unit_keeps_zero_vectors():
    np.testing.assert_allclose(
        unit([[3.0, 0, 4.0], [0, 0, 0]]), [[0.6, 0, 0.8], [0, 0, 0]]
    )


def test_cop_weighted_by_force_norms():
    positions = [[0.0, 0, 0], [1.0, 0, 0]]
    forces = [[0.0, 0, 1.0], [0.0, 3.0, 0]]
    np.testing.assert_allclose(cop_position(positions, forces), [0.75, 0, 0])


def test_cop_of_single_taxel_is_its_position():
    np.testing.assert_allclose(cop_position([[1.0, 2, 3]], [[0, 0, 0.5]]), [1, 2, 3])


def test_cop_of_zero_forces_raises():
    with pytest.raises(ValueError, match="zero"):
        cop_position([[0.0, 0, 0]], [[0.0, 0, 0]])


def test_cop_normal_on_plane_is_up():
    np.testing.assert_allclose(
        cop_normal(PLANE.positions, PLANE_NORMALS, [1e-3, 0, 0]), UP
    )


def test_cop_normal_at_taxel_is_its_normal():
    sensor = sphere_sensor()
    normals = normals_from_rotations(sensor.rotations)
    np.testing.assert_allclose(
        cop_normal(sensor.positions, normals, sensor.positions[5]), normals[5]
    )


def test_cop_normal_between_taxels_is_blend():
    positions = [[-1.0, 0, 0], [1.0, 0, 0]]
    normals = [[-0.6, 0, 0.8], [0.6, 0, 0.8]]
    np.testing.assert_allclose(cop_normal(positions, normals, [0.0, 0, 0]), UP)


def test_cop_normal_cancelling_is_nan():
    positions = [[-1.0, 0, 0], [1.0, 0, 0]]
    normals = [[0.0, 0, 1.0], [0.0, 0, -1.0]]
    assert np.isnan(cop_normal(positions, normals, [0.0, 0, 0])).all()


def test_gaussian_weight_is_one_at_cop_and_decays():
    weights = gaussian_weights([[0.0, 0, 0], [0.003, 0, 0]], [0.0, 0, 0], 0.003)
    np.testing.assert_allclose(weights, [1.0, np.exp(-0.5)])


@pytest.mark.parametrize("sigma", [0.0, -1.0])
def test_gaussian_needs_positive_sigma(sigma):
    with pytest.raises(ValueError, match="positive"):
        gaussian_weights([[0.0, 0, 0]], [0.0, 0, 0], sigma)


def test_blended_direction_is_normal_at_cop_and_outward_far_away():
    positions = np.array([[0.0, 0, 0], [1.0, 0, 0]])
    directions = blended_directions(positions, [UP, UP], [0.0, 0, 0], [1.0, 0.0])
    np.testing.assert_allclose(directions, [UP, [1.0, 0, 0]])


def test_blended_direction_falls_back_to_normal_when_cancelled():
    directions = blended_directions([[0.0, 0, -1.0]], [UP], [0.0, 0, 0], [0.5])
    np.testing.assert_allclose(directions, [UP])


def test_shear_projector():
    projector = shear_projector(UP)
    np.testing.assert_allclose(projector @ UP, 0.0)
    np.testing.assert_allclose(projector @ projector, projector)
    np.testing.assert_allclose(projector @ [1.0, 2.0, 3.0], [1.0, 2.0, 0.0])


def test_matrix_at_cop_is_identity_on_plane():
    # At the CoP w = 1 and b = n, so M = n n^T + I - n n^T = I.
    matrices = taxel_force_matrices(
        PLANE.positions, PLANE_NORMALS, PLANE.positions[12], UP
    )
    np.testing.assert_allclose(matrices[12], np.eye(3), atol=1e-12)


@pytest.mark.parametrize("sensor", [PLANE, cylinder_sensor(), sphere_sensor()])
@pytest.mark.parametrize("sigma", [0.002, 0.003, 1.0])
def test_round_trip_recovers_cop_force(sensor, sigma):
    normals = normals_from_rotations(sensor.rotations)
    cop = sensor.positions.mean(axis=0)
    normal = cop_normal(sensor.positions, normals, cop)
    matrices = taxel_force_matrices(sensor.positions, normals, cop, normal, sigma)
    cop_force = np.array([0.3, -0.2, 2.0])
    forces = cop_to_taxel_forces(cop_force, matrices)
    np.testing.assert_allclose(taxel_to_cop_force(forces, matrices, 0.0), cop_force)


def test_cop_outside_array_still_recovered_without_regularisation():
    cop = np.array([0.02, 0.0, 0.0])
    matrices = taxel_force_matrices(PLANE.positions, PLANE_NORMALS, cop, UP, 0.01)
    forces = cop_to_taxel_forces([0.0, 0.0, 1.0], matrices)
    np.testing.assert_allclose(
        taxel_to_cop_force(forces, matrices, 0.0), [0, 0, 1.0], atol=1e-12
    )


def test_regularisation_shrinks_force():
    cop = PLANE.positions[12]
    matrices = taxel_force_matrices(PLANE.positions, PLANE_NORMALS, cop, UP)
    forces = cop_to_taxel_forces([0.0, 0.0, 1.0], matrices)
    small = taxel_to_cop_force(forces, matrices, 1e-3)
    large = taxel_to_cop_force(forces, matrices, 1e3)
    np.testing.assert_allclose(small, [0, 0, 1.0], rtol=1e-5, atol=1e-12)
    assert np.linalg.norm(large) < 1e-3


def test_tiny_sigma_away_from_taxels_gives_zero_force():
    cop = np.array([0.0015, 0.0015, 0.0])
    matrices = taxel_force_matrices(PLANE.positions, PLANE_NORMALS, cop, UP, 1e-5)
    forces = np.tile([0.0, 0.0, 1.0], (PLANE.num_taxels, 1))
    for regularisation in (0.0, 1e-3):
        np.testing.assert_allclose(
            taxel_to_cop_force(forces, matrices, regularisation), 0.0
        )


@pytest.mark.parametrize("regularisation", [0.0, 1e-3])
def test_no_taxel_gives_zero_force(regularisation):
    force = taxel_to_cop_force(np.zeros((0, 3)), np.zeros((0, 3, 3)), regularisation)
    np.testing.assert_allclose(force, 0.0)
