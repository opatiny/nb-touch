# Visualisation

Drawing a sensor's taxels, taxel forces and contacts in MuJoCo, in `src/nb_touch/viz/`. Needs the `viz` extra: `uv sync --extra viz`.

## Functions

They draw into a `MjvScene`, such as a passive viewer's `user_scn`, and change nothing in the model. Positions and forces are in the scene's world frame.

- `draw_taxels(scene, positions, active=None, radius=0.0004)`: one sphere per taxel, grey, active taxels yellow.
- `draw_taxel_forces(scene, positions, forces, scale, min_force=0.0)`: one blue arrow per taxel force.
- `draw_cop(scene, cop, cop_force, scale, rgba=COP_RGBA)`: a sphere at the CoP and an arrow for its force, red by default.
- `draw_contacts(scene, contacts, scale)`: `draw_cop` for every `Contact`.

| Parameter   | What it does                                        | Default  |
| ----------- | --------------------------------------------------- | -------- |
| `scene`     | `MjvScene` to draw into.                            | required |
| `positions` | Taxel positions (n, 3), in m.                       | required |
| `active`    | Mask (n,) of the taxels to highlight.               | none     |
| `radius`    | Taxel sphere radius, in m.                          | `0.0004` |
| `forces`    | Taxel forces (n, 3), in N.                          | required |
| `scale`     | Arrow length per newton, in m.                      | required |
| `min_force` | Taxel forces below this get no arrow, in N.         | `0.0`    |
| `cop`       | CoP position (3,), in m.                            | required |
| `cop_force` | CoP force (3,), in N.                               | required |
| `rgba`      | Color.                                              | red      |
| `contacts`  | Contacts from `ContactDetector` or `find_contacts`. | required |

Arrow and sphere helpers (`arrows.py`, `markers.py`) taken from `dextHand`, `src/dexthand/sim/arrows.py` and `markers.py`.

## Example: CoP in MuJoCo

`examples/cop_in_mujoco.py` shows the [taxel-CoP mapping](../theory/cop.md) on a synthetic sensor:

- A true CoP force (green) moves around the sensor, its shear turning.
- The model's forward direction turns it into taxel forces (blue), plus noise.
- The contact detection estimates the CoP and CoP force back from the taxel forces alone (red). Taxels in a contact are yellow.
- Once per second, the terminal prints the CoP error and both forces.

Run it on macOS with `uv run mjpython examples/cop_in_mujoco.py`, elsewhere with `uv run python examples/cop_in_mujoco.py`. Settings are constants at the top of the script:

| Constant              | What it does                                                         | Default                         |
| --------------------- | -------------------------------------------------------------------- | ------------------------------- |
| `SHAPE`               | Synthetic sensor: `plane`, `cylinder` or `sphere`.                   | `sphere`                        |
| `NUM_TAXELS`          | Number of taxels.                                                    | `80`                            |
| `NORMAL_FORCE_N`      | Normal part of the true CoP force, in N.                             | `2.0`                           |
| `SHEAR_FORCE_N`       | Shear part of the true CoP force, in N.                              | `0.5`                           |
| `NOISE_N`             | Standard deviation of the noise on each taxel force component, in N. | `0.02`                          |
| `PATH_RADIUS_M`       | Radius of the true CoP's circle, seen from above, in m.              | `0.003`                         |
| `PERIOD_S`            | Time for one turn of the circle, in s.                               | `8.0`                           |
| `SETTINGS`            | `ContactSettings` of the detection, its σ also makes the forces.     | `ContactSettings(min_taxels=1)` |
| `COP_SCALE_M_PER_N`   | CoP force arrow length per newton, in m.                             | `0.006`                         |
| `TAXEL_SCALE_M_PER_N` | Taxel force arrow length per newton, in m.                           | `0.02`                          |
| `RATE_HZ`             | Viewer refresh rate, in Hz.                                          | `30.0`                          |

macOS needs `mjpython` for MuJoCo's passive viewer ([MuJoCo Python docs](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)).
