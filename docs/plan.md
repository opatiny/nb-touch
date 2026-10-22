# Implementation plan

Plan for implementing `nb-touch`, a hardware-agnostic Python library based on Pan et al. 2026, "Beyond Binary: Sim-to-Real Dexterous Manipulation with Physics-Grounded Contact Representation" ([arXiv 2605.28812](https://arxiv.org/abs/2605.28812)). Nothing is implemented yet.

## Principles

- Small functions with meaningful names, one per equation.
- Every function tested, with edge cases, in `__tests__/` folders next to the code.
- Synthetic sensors (taxels on a plane, a cylinder, a sphere) for tests and demos.
- Hardware-agnostic: the user provides taxel geometry, joint torques and kinematics.

## Package layout

- `src/nb_touch/io/`: read and write the [sensor geometry](formats/sensor-geometry.md) and the [recordings](theory/taxel-orientation.md#recording-format).
- `src/nb_touch/geometry/`: [`project_to_so3`](theory/so3-projection.md), point and vector transforms, normal estimation, synthetic sensors.
- `src/nb_touch/contact/`: taxel forces to N contacts, each with CoP, CoP force and stats ([cop.md](theory/cop.md)).
- `src/nb_touch/calibration/`: taxel orientation calibration ([taxel-orientation.md](theory/taxel-orientation.md)).
- `src/nb_touch/viz/`: MuJoCo display of taxel forces and CoP forces, optional extra.
- `examples/`: demos, for example a random CoP force and the resulting taxel forces shown in MuJoCo.

## Steps

1. Setup: `uv`, Python >=3.12, `numpy`, `pyyaml`, `torch`, `pytest`, GitHub test workflow ([dev.md](dev.md)).
2. File formats and their readers and writers.
3. Geometry utils and synthetic sensors.
4. Contacts: segmentation into zones, CoP and stats per zone.
5. Taxel orientation calibration.
6. Visualisation and examples, last: it may turn out to belong in `dextHand` instead.
7. Publishing to PyPI.

## Code reused from `dextHand`

To be moved into `nb-touch` with credit, then removed from `dextHand`, which will depend on `nb-touch`:

- Contact zones: `src/dexthand/touch/contact.py`, `contact_zone.py`, `contact_filters.py`. `tools/touch/detect_touch_contact.py` becomes a thin display wrapper.
- Normal estimation: `src/dexthand/touch/taxels.py`.
- MuJoCo arrows: `src/dexthand/sim/arrows.py`.

## Decisions

- Use a virtual environment (`uv`).
- Several contacts per sensor, each with its own CoP.
- The user provides joint torques, including any current-to-torque mapping.
- Visualisation comes last.
- Recordings: npz plus a YAML sidecar ([taxel-orientation.md](theory/taxel-orientation.md#recording-format)).
- Calibration solver: same as the paper, torch with Adam ([taxel-orientation.md](theory/taxel-orientation.md)).
- Tests in `nb-touch` use synthetic data only. Validation on real data is done in `dextHand`, on the ORCA hand.
