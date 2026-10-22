import numpy as np
import pytest

from nb_touch.contact.contacts import (
    ContactSettings,
    contact_from_taxels,
    find_contacts,
    split_force,
)
from nb_touch.contact.cop_model import (
    SIGMA_M,
    SIGMA_SPACING_FACTOR,
    cop_to_taxel_forces,
    spacing_sigma,
    taxel_force_matrices,
)
from nb_touch.geometry.frames import normals_from_rotations
from nb_touch.geometry.neighbours import taxel_neighbours
from nb_touch.geometry.synthetic import plane_sensor, sphere_sensor

PLANE = plane_sensor(rows=9, cols=9, spacing=0.003)
NORMALS = normals_from_rotations(PLANE.rotations)
NEIGHBOURS = taxel_neighbours(PLANE.positions)
UP = np.array([0.0, 0.0, 1.0])
CENTRE = 40  # middle of the 9 x 9 grid


def push(taxel, cop_force, sigma=SIGMA_SPACING_FACTOR * 0.003):
    """Return the plane's taxel forces for a CoP force at a taxel."""
    matrices = taxel_force_matrices(
        PLANE.positions, NORMALS, PLANE.positions[taxel], UP, sigma
    )
    return cop_to_taxel_forces(cop_force, matrices)


def test_split_force():
    assert split_force([3.0, 0.0, -4.0], UP) == pytest.approx((4.0, 3.0, 0.75))


def test_split_force_pull_has_nan_ratio():
    normal_force, _, ratio = split_force([0.0, 0.0, 1.0], UP)
    assert normal_force == -1.0
    assert np.isnan(ratio)


def test_split_force_nan_normal():
    assert all(np.isnan(split_force([0.0, 0, 1.0], np.full(3, np.nan))))


def test_centred_normal_push_recovered():
    contacts = find_contacts(
        PLANE.positions, NORMALS, push(CENTRE, [0, 0, -2.0]), NEIGHBOURS
    )
    assert len(contacts) == 1
    contact = contacts[0]
    np.testing.assert_allclose(contact.cop, PLANE.positions[CENTRE], atol=1e-12)
    np.testing.assert_allclose(contact.normal, UP)
    np.testing.assert_allclose(contact.cop_force, [0, 0, -2.0], rtol=1e-5, atol=1e-12)
    assert contact.normal_force == pytest.approx(2.0, rel=1e-5)
    assert contact.shear_force == pytest.approx(0.0, abs=1e-9)
    assert CENTRE in contact.taxels


def test_sheared_push_moves_cop_towards_shear():
    # Shear makes the taxel force norms asymmetric, so the force-weighted CoP moves:
    # a property of the paper's CoP estimate, not of the solve.
    contact = find_contacts(
        PLANE.positions, NORMALS, push(CENTRE, [0.5, 0, -2.0]), NEIGHBOURS
    )[0]
    shift = contact.cop - PLANE.positions[CENTRE]
    assert 0 < shift[0] < 0.001


@pytest.mark.xfail(
    strict=True,
    reason="CoP estimate biased by shear: slip ratio 0.29 instead of 0.25 (0.278 "
    "with all taxels, exact with the true CoP)",
)
def test_sheared_push_slip_ratio_within_10_percent():
    contact = find_contacts(
        PLANE.positions, NORMALS, push(CENTRE, [0.5, 0, -2.0]), NEIGHBOURS
    )[0]
    assert contact.slip_ratio == pytest.approx(0.25, rel=0.1)


def test_two_pushes_give_two_contacts_strongest_first():
    # Taxels 20 and 60: rows and columns 2 and 6 of the 9 x 9 grid.
    forces = push(20, [0, 0, -1.0], 0.003) + push(60, [0, 0, -3.0], 0.003)
    contacts = find_contacts(PLANE.positions, NORMALS, forces, NEIGHBOURS)
    assert len(contacts) == 2
    np.testing.assert_allclose(contacts[0].cop, PLANE.positions[60], atol=1e-4)
    np.testing.assert_allclose(contacts[1].cop, PLANE.positions[20], atol=1e-4)


def test_no_force_no_contact():
    assert find_contacts(PLANE.positions, NORMALS, np.zeros((81, 3)), NEIGHBOURS) == []


def test_single_taxel_contact():
    forces = np.zeros((81, 3))
    forces[CENTRE] = [0.0, 0.0, 1.0]
    settings = ContactSettings(min_taxels=1)
    contacts = find_contacts(PLANE.positions, NORMALS, forces, NEIGHBOURS, settings)
    assert len(contacts) == 1
    assert contacts[0].count == 1
    np.testing.assert_allclose(contacts[0].cop_force, [0, 0, 1.0], rtol=1e-5)
    assert contacts[0].max_force == contacts[0].mean_force == 1.0


def test_single_taxel_dropped_by_default():
    forces = np.zeros((81, 3))
    forces[CENTRE] = [0.0, 0.0, 1.0]
    assert find_contacts(PLANE.positions, NORMALS, forces, NEIGHBOURS) == []


def test_zero_threshold_ignores_all_zero_zone():
    settings = ContactSettings(threshold=-1.0, min_taxels=1)
    contacts = find_contacts(
        PLANE.positions, NORMALS, np.zeros((81, 3)), NEIGHBOURS, settings
    )
    assert contacts == []


def test_total_force_is_sum_of_zone_forces():
    forces = push(CENTRE, [0, 0, -2.0])
    contact = find_contacts(PLANE.positions, NORMALS, forces, NEIGHBOURS)[0]
    np.testing.assert_allclose(contact.total_force, forces[contact.taxels].sum(axis=0))


def test_contact_on_sphere_has_outward_normal():
    sensor = sphere_sensor(radius=0.01, count=60)
    normals = normals_from_rotations(sensor.rotations)
    taxels = np.arange(sensor.num_taxels)
    top = int(np.argmax(sensor.positions[:, 2]))
    cop = sensor.positions[top]
    sigma = spacing_sigma(sensor.positions)
    matrices = taxel_force_matrices(sensor.positions, normals, cop, normals[top], sigma)
    forces = cop_to_taxel_forces(-0.5 * normals[top], matrices)
    contact = contact_from_taxels(taxels, sensor.positions, normals, forces)
    assert contact.normal[2] > 0.99
    assert contact.normal_force > 0


def test_default_sigma_is_factor_times_median_taxel_spacing():
    sensor = plane_sensor(rows=3, cols=4, spacing=0.004)
    settings = ContactSettings().from_geometry(sensor.positions)
    assert settings.sigma == pytest.approx(SIGMA_SPACING_FACTOR * 0.004)


def test_sigma_factor_scales_the_spacing():
    sensor = plane_sensor(rows=3, cols=4, spacing=0.004)
    settings = ContactSettings(sigma_factor=2.0).from_geometry(sensor.positions)
    assert settings.sigma == pytest.approx(0.008)


def test_sigma_factor_ignored_when_sigma_set():
    settings = ContactSettings(sigma=0.01, sigma_factor=2.0)
    assert settings.from_geometry(PLANE.positions).sigma == 0.01


def test_set_sigma_is_kept():
    settings = ContactSettings(sigma=0.01)
    assert settings.from_geometry(PLANE.positions) is settings


def test_single_taxel_sensor_falls_back_to_default_sigma():
    assert ContactSettings().from_geometry(np.zeros((1, 3))).sigma == SIGMA_M


def test_find_contacts_uses_taxel_spacing_as_sigma():
    sensor = plane_sensor(rows=9, cols=9, spacing=0.005)
    normals = normals_from_rotations(sensor.rotations)
    cop_force = np.array([0.0, 0.0, -2.0])
    matrices = taxel_force_matrices(
        sensor.positions,
        normals,
        sensor.positions[CENTRE],
        UP,
        SIGMA_SPACING_FACTOR * 0.005,
    )
    forces = cop_to_taxel_forces(cop_force, matrices)
    (contact,) = find_contacts(
        sensor.positions,
        normals,
        forces,
        taxel_neighbours(sensor.positions),
        ContactSettings(threshold=0.0, regularisation=0.0),
    )
    assert contact.cop_force == pytest.approx(cop_force, abs=1e-6)
