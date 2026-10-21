import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from nb_touch.viz.markers import draw_spheres  # noqa: E402


def test_draw_spheres_one_geom_per_point(make_scene):
    scene = make_scene(maxgeom=10)
    centres = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
    assert draw_spheres(scene, centres, 0.001, (1, 0, 0, 1)) == 2
    assert scene.ngeom == 2
    geom = scene.geoms[1]
    assert geom.type == mujoco.mjtGeom.mjGEOM_SPHERE
    assert geom.pos == pytest.approx([0.1, 0.2, 0.3])
    assert geom.size[0] == pytest.approx(0.001)
    assert geom.rgba == pytest.approx([1, 0, 0, 1])


def test_draw_spheres_no_points(make_scene):
    scene = make_scene(maxgeom=10)
    assert draw_spheres(scene, np.empty((0, 3)), 0.001, (1, 0, 0, 1)) == 0
    assert scene.ngeom == 0


def test_draw_spheres_stops_when_the_scene_is_full(make_scene):
    scene = make_scene(maxgeom=2)
    scene.ngeom = 1
    assert draw_spheres(scene, np.zeros((3, 3)), 0.001, (1, 0, 0, 1)) == 1
    assert scene.ngeom == 2
