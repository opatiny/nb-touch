import numpy as np
import pytest

from nb_touch.contact.contacts import ContactSettings, find_contacts
from nb_touch.contact.cop_model import cop_to_taxel_forces, taxel_force_matrices
from nb_touch.contact.detector import ContactDetector
from nb_touch.contact.filters import Debounce
from nb_touch.geometry.frames import normals_from_rotations
from nb_touch.geometry.neighbours import taxel_neighbours
from nb_touch.geometry.synthetic import plane_sensor, sphere_sensor
from nb_touch.geometry.transforms import rotate_vectors
from nb_touch.io.sensor_geometry import SensorGeometry

SPHERE = sphere_sensor(radius=0.01, count=80)
NORMALS = normals_from_rotations(SPHERE.rotations)


def sphere_push(taxel, cop_force):
    """Return sensor-frame taxel forces for a CoP force at a taxel of the sphere."""
    cop = SPHERE.positions[taxel]
    matrices = taxel_force_matrices(SPHERE.positions, NORMALS, cop, NORMALS[taxel])
    return cop_to_taxel_forces(cop_force, matrices)


def test_taxel_frame_forces_match_sensor_frame_result():
    forces = sphere_push(0, -2.0 * NORMALS[0])
    local = rotate_vectors(SPHERE.rotations.transpose(0, 2, 1), forces)
    detector = ContactDetector(SPHERE)
    from_local = detector.update(local, in_taxel_frames=True)
    from_sensor = ContactDetector(SPHERE).update(forces, in_taxel_frames=False)
    expected = find_contacts(
        SPHERE.positions, NORMALS, forces, taxel_neighbours(SPHERE.positions)
    )
    assert len(from_local) == len(from_sensor) == len(expected) == 1
    np.testing.assert_allclose(from_local[0].cop_force, expected[0].cop_force)
    np.testing.assert_allclose(from_sensor[0].cop_force, expected[0].cop_force)


def test_missing_rotations_are_estimated():
    plane = plane_sensor()
    detector = ContactDetector(SensorGeometry("p", plane.positions))
    np.testing.assert_allclose(
        detector.normals, normals_from_rotations(plane.rotations), atol=1e-9
    )


def test_filter_delays_contact():
    forces = sphere_push(0, -2.0 * NORMALS[0])
    settings = ContactSettings()
    detector = ContactDetector(SPHERE, settings, Debounce(settings.threshold, 2))
    assert detector.update(forces, in_taxel_frames=False) == []
    assert len(detector.update(forces, in_taxel_frames=False)) == 1


def test_force_frame_must_be_given():
    detector = ContactDetector(SPHERE)
    with pytest.raises(TypeError, match="in_taxel_frames"):
        detector.update(np.zeros((SPHERE.num_taxels, 3)))
    with pytest.raises(TypeError):
        detector.update(np.zeros((SPHERE.num_taxels, 3)), True)
