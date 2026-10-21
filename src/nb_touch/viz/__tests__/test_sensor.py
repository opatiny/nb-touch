import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from nb_touch.contact.contacts import ContactSettings, find_contacts  # noqa: E402
from nb_touch.geometry.frames import normals_from_rotations  # noqa: E402
from nb_touch.geometry.neighbours import taxel_neighbours  # noqa: E402
from nb_touch.geometry.synthetic import plane_sensor  # noqa: E402
from nb_touch.viz.sensor import (  # noqa: E402
    ACTIVE_RGBA,
    TAXEL_RGBA,
    draw_contacts,
    draw_cop,
    draw_taxel_forces,
    draw_taxels,
)

PLANE = plane_sensor(rows=3, cols=3)


def test_taxels_drawn_with_active_highlighted(make_scene):
    scene = make_scene(20)
    active = np.zeros(9, dtype=bool)
    active[4] = True
    assert draw_taxels(scene, PLANE.positions, active) == 9
    colours = [tuple(scene.geoms[i].rgba) for i in range(9)]
    assert colours.count(pytest.approx(ACTIVE_RGBA)) == 1
    assert colours.count(pytest.approx(TAXEL_RGBA)) == 8


def test_taxels_without_mask_all_plain(make_scene):
    scene = make_scene(20)
    draw_taxels(scene, PLANE.positions)
    assert all(scene.geoms[i].rgba == pytest.approx(TAXEL_RGBA) for i in range(9))


def test_small_taxel_forces_skipped(make_scene):
    scene = make_scene(40)
    forces = np.zeros((9, 3))
    forces[0] = [0.0, 0.0, 1.0]
    assert draw_taxel_forces(scene, PLANE.positions, forces, 0.01, 0.1) == 1


def test_cop_is_sphere_and_arrow(make_scene):
    scene = make_scene(10)
    assert draw_cop(scene, np.zeros(3), [0.0, 0.0, 2.0], 0.01) == 3
    assert scene.geoms[0].type == mujoco.mjtGeom.mjGEOM_SPHERE


def test_zero_cop_force_is_only_a_sphere(make_scene):
    scene = make_scene(10)
    assert draw_cop(scene, np.zeros(3), np.zeros(3), 0.01) == 1


def test_contacts_drawn(make_scene):
    forces = np.zeros((9, 3))
    forces[[0, 1, 3]] = [0.0, 0.0, 1.0]
    normals = normals_from_rotations(PLANE.rotations)
    contacts = find_contacts(
        PLANE.positions,
        normals,
        forces,
        taxel_neighbours(PLANE.positions),
        ContactSettings(min_taxels=1),
    )
    scene = make_scene(10)
    assert draw_contacts(scene, contacts, 0.01) == 3
    assert draw_contacts(scene, [], 0.01) == 0
