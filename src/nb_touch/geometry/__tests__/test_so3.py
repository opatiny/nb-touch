import numpy as np
import pytest

from nb_touch.geometry.frames import rotation_from_z
from nb_touch.geometry.so3 import project_to_so3


def is_rotation(matrix):
    return np.allclose(matrix @ matrix.T, np.eye(3)) and np.isclose(
        np.linalg.det(matrix), 1.0
    )


def test_rotation_is_unchanged():
    rotation = rotation_from_z(np.array([0.6, 0.0, 0.8]))
    np.testing.assert_allclose(project_to_so3(rotation), rotation, atol=1e-12)


def test_noisy_rotation_projects_back_close():
    rotation = rotation_from_z(np.array([0.0, 0.6, 0.8]))
    noisy = rotation + 1e-3 * np.random.default_rng(0).normal(size=(3, 3))
    projected = project_to_so3(noisy)
    assert is_rotation(projected)
    np.testing.assert_allclose(projected, rotation, atol=5e-3)


def test_scaled_rotation_loses_scale():
    rotation = rotation_from_z(np.array([0.6, 0.0, 0.8]))
    np.testing.assert_allclose(project_to_so3(3.0 * rotation), rotation, atol=1e-12)


def test_reflection_gives_rotation():
    assert is_rotation(project_to_so3(np.diag([1.0, 1.0, -1.0])))


@pytest.mark.parametrize("matrix", [np.zeros((3, 3)), np.ones((3, 3))])
def test_degenerate_matrices_give_rotation(matrix):
    assert is_rotation(project_to_so3(matrix))


def test_stack_of_random_matrices():
    matrices = np.random.default_rng(1).normal(size=(10, 3, 3))
    projected = project_to_so3(matrices)
    assert projected.shape == (10, 3, 3)
    assert all(is_rotation(matrix) for matrix in projected)


def test_empty_stack():
    assert project_to_so3(np.zeros((0, 3, 3))).shape == (0, 3, 3)
