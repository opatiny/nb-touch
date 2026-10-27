# Visualisation in MuJoCo

Drawing a sensor's taxels, taxel forces and contacts in MuJoCo, in `src/nb_touch/viz/`.

## Functions

They draw into a `MjvScene`, such as a passive viewer's `user_scn`, and change nothing in the model. Positions and forces are in the scene's world frame.

- `draw_taxels(scene, positions, active=None, radius=0.0004)`: one sphere per taxel, grey, active taxels yellow.
- `draw_taxel_forces(scene, positions, forces, scale, min_force=0.0)`: one blue arrow per taxel force.
- `draw_cop(scene, cop, cop_force, scale, rgba=COP_RGBA)`: a sphere at the CoP and an arrow for its force, red by default.
- `draw_contacts(scene, contacts, scale)`: `draw_cop` for every `Contact`.

## Example: CoP in MuJoCo

`examples/cop_in_mujoco.py` shows the [taxel-CoP mapping](cop/cop.md) on a synthetic sensor (see ...).

- A true CoP force (green) moves around the sensor, its shear turning.
- The model's forward direction turns it into taxel forces (blue), plus noise.
- The contact detection estimates the CoP and CoP force back from the taxel forces alone (red). Taxels in a contact are yellow.

Run it on macOS with `uv run mjpython examples/cop_in_mujoco.py`, elsewhere with `uv run python examples/cop_in_mujoco.py`. Settings are constants at the top of the script.

MacOS needs `mjpython` for MuJoCo's passive viewer ([MuJoCo Python docs](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)).
