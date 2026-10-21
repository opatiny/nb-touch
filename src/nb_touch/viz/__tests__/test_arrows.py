import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from nb_touch.viz.arrows import draw_arrows, force_arrows, split_arrow  # noqa: E402


def test_force_arrows_scale_the_force():
    ((start, end),) = force_arrows([1, 1, 1], [0, 0, 2], scale=0.01)
    assert start == pytest.approx([1, 1, 1])
    assert end == pytest.approx([1, 1, 1.02])


def test_force_arrows_drop_small_forces():
    starts = [[0, 0, 0], [1, 0, 0], [2, 0, 0]]
    forces = [[0.01, 0, 0], [0, 0.5, 0], [0, 0, -0.05]]
    arrows = force_arrows(starts, forces, scale=1.0, min_force=0.05)
    assert [list(start) for start, _ in arrows] == [[1, 0, 0], [2, 0, 0]]


def test_force_arrows_with_no_force():
    assert force_arrows(np.zeros((0, 3)), np.zeros((0, 3)), scale=1.0) == []
    assert force_arrows([0, 0, 0], [0, 0, 0], scale=1.0, min_force=0.01) == []


def test_split_arrow_keeps_a_fixed_head():
    start, neck, end = split_arrow([0, 0, 0], [0, 0, 0.1], head_length=0.01)
    assert neck == pytest.approx([0, 0, 0.09])
    assert end == pytest.approx([0, 0, 0.1])


def test_split_arrow_shorter_than_its_head_is_all_head():
    start, neck, _ = split_arrow([0, 0, 0], [0.002, 0, 0], head_length=0.01)
    assert neck == pytest.approx(start)


def test_split_arrow_of_zero_length():
    assert split_arrow([1, 1, 1], [1, 1, 1], head_length=0.01) is None


def test_draw_arrows_draws_a_shaft_and_a_head(make_scene):
    scene = make_scene(10)
    drawn = draw_arrows(
        scene,
        [(np.zeros(3), np.array([0, 0, 0.1]))],
        (1, 0, 0, 1),
        thickness=0.004,
        head_length=0.01,
    )
    assert drawn == 1 and scene.ngeom == 2
    shaft, head = scene.geoms[0], scene.geoms[1]
    assert shaft.type == mujoco.mjtGeom.mjGEOM_CYLINDER
    assert head.type == mujoco.mjtGeom.mjGEOM_ARROW
    assert head.pos == pytest.approx([0, 0, 0.09])
    assert head.size[2] == pytest.approx(0.01)
    # thickness is a diameter; MuJoCo sizes connectors by radius.
    assert shaft.size[0] == pytest.approx(0.002)
    assert head.size[0] == pytest.approx(0.002)
    assert head.rgba == pytest.approx([1, 0, 0, 1])


def test_draw_arrows_heads_do_not_grow_with_the_force(make_scene):
    scene = make_scene(10)
    arrows = [
        (np.zeros(3), np.array([0.05, 0, 0])),
        (np.zeros(3), np.array([0.5, 0, 0])),
    ]
    draw_arrows(scene, arrows, (1, 1, 1, 1), head_length=0.004)
    heads = [
        scene.geoms[i]
        for i in range(scene.ngeom)
        if scene.geoms[i].type == mujoco.mjtGeom.mjGEOM_ARROW
    ]
    assert [h.size[2] for h in heads] == pytest.approx([0.004, 0.004])


def test_draw_arrows_short_arrow_is_only_a_head(make_scene):
    scene = make_scene(10)
    draw_arrows(
        scene, [(np.zeros(3), np.array([0.001, 0, 0]))], (1, 1, 1, 1), head_length=0.004
    )
    assert scene.ngeom == 1
    assert scene.geoms[0].size[2] == pytest.approx(0.001)


def test_draw_arrows_skips_zero_length_arrows(make_scene):
    scene = make_scene(10)
    assert draw_arrows(scene, [(np.ones(3), np.ones(3))], (1, 1, 1, 1)) == 0
    assert scene.ngeom == 0


def test_draw_arrows_appends_after_existing_geoms(make_scene):
    scene = make_scene(10)
    arrow = (np.zeros(3), np.ones(3))
    draw_arrows(scene, [arrow], (1, 0, 0, 1))
    draw_arrows(scene, [arrow, arrow], (0, 1, 0, 1))
    assert scene.ngeom == 6
    assert scene.geoms[5].rgba == pytest.approx([0, 1, 0, 1])


def test_draw_arrows_stops_at_maxgeom(make_scene):
    scene = make_scene(5)
    arrow = (np.zeros(3), np.ones(3))
    # Two geoms per arrow: the third arrow would only half fit, so it is dropped whole.
    assert draw_arrows(scene, [arrow] * 5, (1, 1, 1, 1)) == 2
    assert scene.ngeom == 4
