# Documentation

Index of the `nb-touch` docs.

## Project

- [plan.md](plan.md): implementation plan, package layout and open questions.

## Theory

- [cop.md](theory/cop.md): from taxel forces to one or more contacts, each with its CoP and CoP force.
- [taxel-orientation.md](theory/taxel-orientation.md): calibrating the taxel orientations from joint torques, and the solver.
- [so3-projection.md](theory/so3-projection.md): `project_to_so3`, the closest rotation to a 3×3 matrix.
- [joint-torques.md](theory/joint-torques.md): which torques the calibration needs and where they come from.

## File formats

- [sensor-geometry.md](formats/sensor-geometry.md): taxel positions and orientations in the sensor frame.
- [recordings.md](formats/recordings.md): data recorded for the taxel orientation calibration, and why npz rather than JSON.

## Development

- [setup.md](development/setup.md): virtual environment and tests.
- [ci-and-publishing.md](development/ci-and-publishing.md): GitHub checks, automatic releases and publishing to PyPI.
