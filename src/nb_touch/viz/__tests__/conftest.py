import pytest

mujoco = pytest.importorskip("mujoco")

MODEL = "<mujoco><worldbody><light pos='0 0 1'/></worldbody></mujoco>"


@pytest.fixture
def make_scene():
    """Return a function making an empty `MjvScene` holding at most `maxgeom` geoms.

    Adapted from `make_scene` in `dextHand`, `src/dexthand/testing.py`.
    """
    model = mujoco.MjModel.from_xml_string(MODEL)
    return lambda maxgeom: mujoco.MjvScene(model, maxgeom=maxgeom)
