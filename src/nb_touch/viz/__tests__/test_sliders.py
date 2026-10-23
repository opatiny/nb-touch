import pytest

mujoco = pytest.importorskip("mujoco")

from nb_touch.viz.sliders import (
    add_sliders,
    compile_with_sliders,
    set_sliders,
    slider_values,
)

MODEL = "<mujoco><worldbody><light pos='0 0 1'/></worldbody></mujoco>"
RANGES = {"sigma_mm": (0.5, 20.0), "lambda": (0.0, 1.0)}


def test_add_sliders_makes_one_actuator_per_slider():
    model = mujoco.MjModel.from_xml_string(add_sliders(MODEL, RANGES))
    assert model.nu == 2
    assert model.actuator("sigma_mm").ctrlrange == pytest.approx([0.5, 20.0])
    assert model.actuator("lambda").ctrlrange == pytest.approx([0.0, 1.0])


def test_add_sliders_keeps_the_model():
    model = mujoco.MjModel.from_xml_string(add_sliders(MODEL, RANGES))
    assert model.nlight == 1


def test_add_sliders_with_no_slider():
    model = mujoco.MjModel.from_xml_string(add_sliders(MODEL, {}))
    assert model.nu == 0


def test_compile_with_sliders_sets_the_values():
    model, data = compile_with_sliders(MODEL, RANGES, {"sigma_mm": 3, "lambda": 0.1})
    assert slider_values(model, data) == pytest.approx({"sigma_mm": 3.0, "lambda": 0.1})


def test_set_sliders_changes_only_the_named_ones():
    model, data = compile_with_sliders(MODEL, RANGES, {"sigma_mm": 3, "lambda": 0.1})
    set_sliders(data, {"lambda": 0.5})
    assert slider_values(model, data) == pytest.approx({"sigma_mm": 3.0, "lambda": 0.5})


def test_set_sliders_rejects_an_unknown_name():
    _, data = compile_with_sliders(MODEL, RANGES, {})
    with pytest.raises(KeyError):
        set_sliders(data, {"unknown": 1.0})
