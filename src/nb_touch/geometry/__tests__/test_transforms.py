import numpy as np

from nb_touch.geometry.frames import rotation_from_z
from nb_touch.geometry.transforms import (
    angles_between_rotations,
    rotate_vectors,
    rotations_from_vectors,
    transform_points,
)

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


def test_rotation_vector_quarter_turn_about_z():
    np.testing.assert_allclose(
        rotations_from_vectors([[0.0, 0.0, np.pi / 2]])[0], QUARTER_TURN_Z, atol=1e-12
    )


def test_zero_rotation_vector_is_identity():
    np.testing.assert_allclose(rotations_from_vectors([[0.0, 0.0, 0.0]])[0], np.eye(3))


def test_rotation_vector_keeps_its_axis_and_is_a_rotation():
    vector = np.array([0.3, -0.5, 0.2])
    rotation = rotations_from_vectors([vector])[0]
    np.testing.assert_allclose(rotation @ vector, vector, atol=1e-12)
    np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(rotation), 1.0)


def test_rotation_vector_angle():
    rotation = rotations_from_vectors([[0.0, 0.7, 0.0]])[0]
    assert np.isclose(np.arccos((np.trace(rotation) - 1) / 2), 0.7)


def test_empty_rotation_vectors():
    assert rotations_from_vectors(np.zeros((0, 3))).shape == (0, 3, 3)


def test_angles_between_rotations():
    turns = rotations_from_vectors([[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0, 0, np.pi]])
    first = np.stack([QUARTER_TURN_Z] * 3)
    np.testing.assert_allclose(
        angles_between_rotations(first, first @ turns), [0.0, 0.4, np.pi], atol=1e-7
    )
