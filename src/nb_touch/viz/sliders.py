"""Sliders in the MuJoCo viewer, to change script settings while it runs.

The passive viewer has no API for custom widgets, but its Control panel shows one
slider per actuator. Each slider here is a motor on a hidden dummy joint: its control
value is the setting. The model is never stepped, so nothing moves.
"""

import mujoco

DUMMY_NAME = "slider_dummy"
"""Name of the hidden body and joint the slider motors act on."""


def add_sliders(xml: str, ranges: dict[str, tuple[float, float]]) -> str:
    """Return the MJCF `xml` with one viewer slider per entry of `ranges`.

    Args:
        xml: MJCF model, ending with `</mujoco>`.
        ranges: Slider name (shown in the Control panel) to its (min, max) range.

    Returns:
        The model with a hidden dummy body and one motor per slider.
    """
    motors = "".join(
        f'<motor name="{name}" joint="{DUMMY_NAME}" ctrlrange="{low} {high}"/>'
        for name, (low, high) in ranges.items()
    )
    sliders = (
        f'<worldbody><body name="{DUMMY_NAME}">'
        f'<joint name="{DUMMY_NAME}" type="slide"/>'
        '<inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>'
        f"</body></worldbody><actuator>{motors}</actuator>"
    )
    head, tail = xml.rsplit("</mujoco>", 1)
    return f"{head}{sliders}</mujoco>{tail}"


def set_sliders(data, values: dict[str, float]) -> None:
    """Move the sliders named in `values` to their value, in place in `data`."""
    for name, value in values.items():
        data.actuator(name).ctrl[0] = value


def slider_values(model, data) -> dict[str, float]:
    """Return the current value of every slider, by name."""
    return {
        model.actuator(index).name: float(data.ctrl[index]) for index in range(model.nu)
    }


def compile_with_sliders(xml: str, ranges, values) -> tuple:
    """Return the `(model, data)` of `xml` with sliders, set to their `values`."""
    model = mujoco.MjModel.from_xml_string(add_sliders(xml, ranges))
    data = mujoco.MjData(model)
    set_sliders(data, values)
    return model, data
