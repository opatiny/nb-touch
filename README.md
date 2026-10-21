# Beyond binary touch sensing

This library is useful for: processing data of a robot equipped with taxel array sensors, as well as to detect the taxels orientation based on experimental measurements.

Hardware-agnostic python library implementing the main functionalities of the article Pan, Jiahe, et al. "Beyond Binary: Sim-to-Real Dexterous Manipulation with Physics-Grounded Contact Representation." arXiv preprint arXiv:2605.28812 (2026).

The features include:

- Conversion from Center of Pressure (CoP) to individual taxel forces
- Conversion from taxel forces to CoP force
- Tools for automatic taxel orientation detection: this requires torque sensors for all the links to the touch sensor.

## Activate .venv when developping

If necessary: deactivate previously activated venv with `deactivate`.

Activate the local venv with `source .venv/bin/activate`.

## Running tests

In an activated venv terminal: `pytest`

Otherwise: `uv run pytest`

## Recording data on the ORCA hand

To record the data for the taxel orientation calibration on the ORCA hand, use `tools/touch/record_taxel_calibration.py` in [`dextHand`](https://github.com/opatiny/robManip) (doc: `docs/tools/record-taxel-calibration.md`). It writes the [recording format](docs/formats/recordings.md).

## Examples

- `examples/cop_in_mujoco.py`: the taxel-CoP mapping on a synthetic sensor, shown in MuJoCo. Needs `uv sync --extra viz`; run with `uv run mjpython examples/cop_in_mujoco.py` on macOS. See [viz.md](docs/usage/viz.md).
