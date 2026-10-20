import numpy as np

from nb_touch.geometry.frames import rotation_from_z
from nb_touch.geometry.transforms import rotate_vectors, transform_points

QUARTER_TURN_Z = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def test_each_vector_rotated_by_its_own_rotation():
    rotations = np.stack([np.eye(3), QUARTER_TURN_Z])
    vectors = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        rotate_vectors(rotations, vectors), [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )


def test_one_rotation_applied_to_all_vectors():
    vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    np.testing.assert_allclose(
        rotate_vectors(QUARTER_TURN_Z, vectors), [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )


def test_local_z_goes_to_normal():
    normal = np.array([0.0, 0.6, 0.8])
    rotated = rotate_vectors(rotation_from_z(normal)[None], np.array([[0, 0, 1.0]]))
    np.testing.assert_allclose(rotated[0], normal)


def test_no_vectors():
    assert rotate_vectors(np.zeros((0, 3, 3)), np.zeros((0, 3))).shape == (0, 3)


def test_transform_points_rotates_then_translates():
    points = np.array([[1.0, 0.0, 0.0]])
    moved = transform_points(QUARTER_TURN_Z, [0.0, 0.0, 2.0], points)
    np.testing.assert_allclose(moved, [[0.0, 1.0, 2.0]])
