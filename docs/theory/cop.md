# Contacts and CoP

From a set of taxels with their xyz forces to N contacts on a sensor, each with its centre of pressure (CoP), CoP force and stats.

Signs of forces and normals: see [Conventions](../../README.md#conventions).

## Pipeline

1. **Active taxels**: taxels whose force norm is above a threshold, optionally filtered over time (hysteresis or debounce) so contacts do not flicker.
2. **Zones**: active taxels grouped into connected groups of neighbours. Each zone is one contact.
3. **Per zone**:
   - CoP position: taxel positions averaged with the force norms as weights, expressed in sensor frame.
   - CoP force: the paper's ridge least-squares solve ([arXiv 2605.28812](https://arxiv.org/abs/2605.28812), Section 3.2), restricted to the zone's taxels. x,y,z components expressed in sensor frame.
   - Stats: total force, normal force, shear force, slip ratio (shear / normal), max and mean taxel force, taxel count, contact normal.

> Limitation: two contacts on adjacent taxels merge into one zone. Splitting them needs a local-maxima step.

## Usage

Implemented in `src/nb_touch/contact/`, imported from `nb_touch.contact`.

This project uses the following reference frames: the **sensor** frame and the frames for each individual **taxels**. In order to combine the taxel forces, they first need to be transformed into a common frame, in our case the sensor frame.

In order to detect the contact zones on a finger and get the CoP position and force for each of these zones, use `ContactDetector(geometry, settings=ContactSettings(), active_filter=None)`. it will compute the taxel's normals and neighbours internally.

Moreover, if the orientation of some taxel frames are missing, the contact detectors will fill them in based on the geometry!!

Each missing rotation is the smallest one taking the sensor's z axis to the taxel's normal (Rodrigues' formula):

- z: the taxel's outward normal, from a plane fitted to the taxel and its nearest neighbours.
- x and y: the sensor's x and y, tilted as little as possible to stay perpendicular to that normal. On a flat sensor, they are the sensor's x and y.
- The positions of the taxels only allow to estimate the normal (z axis), not the rotation about it: the real x and y of a taxel should come from the [orientation calibration](taxel-orientation.md) or a known taxel layout.

Source: `src/nb_touch/geometry/frames.py`; see also [Finding normal at taxel](../sensor-geometry.md#finding-normal-at-taxel).

`ContactSettings` (symbols as in the paper, Section 3.2):

| Field            | What it does                                                                                                    | Default |
| ---------------- | --------------------------------------------------------------------------------------------------------------- | ------- |
| `threshold`      | Taxel force norm above which a taxel is active, in N.                                                           | `0.2`   |
| `min_taxels`     | Minimum number of active taxels in a contact.                                                                   | `3`     |
| `sigma`          | Spread σ of the Gaussian weights, in m; `None` for `sigma_factor` times the median taxel spacing of the sensor. | `None`  |
| `sigma_factor`   | σ over the median taxel spacing, used when `sigma` is `None`. A sensor of a single taxel gets σ = 3 mm.         | `1.5`   |
| `regularisation` | Ridge λ of the CoP force solve; 0 gives the minimum-norm least squares.                                         | `0.001` |

## The lambda and sigma parameters

## Sigma

`sigma`or $\sigma$ (in meters): spread of the Gaussian weights $w_i$, which defines the distance over which a taxel still shares the CoP force.

- Small $\sigma$: only taxels at the CoP count; the others get $w_i \approx 0$ and their $\hat{b}_i$ turn to the radial direction $\hat{v}_i$.
- Large $\sigma$: every taxel gets $w_i \approx 1$, so each is modelled as carrying the whole CoP force, and the solve returns about the mean taxel force instead of their sum.
- Used both in the forward model (from the CoP force to the taxel forces) and in the inverse solve.

## Regularisation

`regularisation` or $\lambda$ (no unit): parameter for the ridge regularisation ([Hoerl and Kennard 1970](https://doi.org/10.1080/00401706.1970.10488634)) of the inverse solve.

- In plain words: when the taxel forces do not pin down the CoP force well, plain least squares can return a huge answer driven by noise. Ridge accepts a small bias towards 0 in exchange for a much more stable answer.
- Adds $\lambda^2$ to the diagonal of $A^T A$, with $A$ the matrix of the forward model, from the CoP force to the stacked taxel forces, so the solve stays defined when $A^T A$ is singular or close to it (few taxels, all $w_i \approx 0$).
- Pulls the CoP force $f_{cop}$ towards 0: negligible while $\lambda^2 \ll \sum_i w_i^2$, the size of $A^T A$.
- $\lambda = 0$: plain least squares (minimum-norm solution when singular).
- Used only in the inverse solve, not in the forward model.

## Default values

The paper gives no value `sigma` and `regularisation`. By default, we set σ to be 1.5 times the median taxel spacing on the sensor. This ensures 8 point connectivity in a rectangular grid. The default value for `regularisation` is 1e-3. To learn more about that parameter, refer to [Estimating a good λ from the data](lambda-from-data.md).

## Known bias

Shear makes the taxel force norms asymmetric, so the force-weighted CoP moves towards the shear and the solved force is off. Push of [0.5, 0, -2] N at the centre of a 9×9 plane, 3 mm spacing, σ = 3 mm (computed by us):

| CoP                        | CoP shift | Slip ratio (true 0.25) |
| -------------------------- | --------- | ---------------------- |
| True                       | 0         | 0.25                   |
| Estimated, all taxels      | 0.36 mm   | 0.278                  |
| Estimated, threshold 0.2 N | 0.48 mm   | 0.291                  |
