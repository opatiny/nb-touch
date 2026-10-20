# Contacts and CoP

From a set of taxels with their xyz forces to N contacts on a sensor, each with its centre of pressure (CoP), CoP force and stats.

## Pipeline

1. **Active taxels**: taxels whose force norm is above a threshold, optionally filtered over time (hysteresis or debounce) so contacts do not flicker.
2. **Zones**: active taxels grouped into connected groups of neighbours. Each zone is one contact.
3. **Per zone**:
   - CoP position: taxel positions averaged with the force norms as weights, expressed in sensor frame.
   - CoP force: the paper's solve (below), restricted to the zone's taxels. x,y,z components expressed in sensor frame.
   - Stats: total force, normal force, shear force, slip ratio (shear / normal), max and mean taxel force, taxel count, contact normal.

> Limitation: two contacts on adjacent taxels merge into one zone. Splitting them needs a local-maxima step.

## CoP model (Pan et al. 2026)

Per sensor, the contact is a 3D force and a 3D position **in the sensor frame**. Only the normal component is used, as simulated shear is unreliable.

- Active taxels: $\mathcal{A} = \{i : \|f_i\| > \epsilon\}$.
- Forward model, CoP force to taxel forces: $f_i = M_i f_{cop}$, with $M_i = w_i (\hat{b}_i \hat{n}_{cop}^T + P_{shear})$.
  - Weights: $w_i = \exp(-\|p_i - p_{cop}\|^2 / (2\sigma^2))$.
  - Normal force direction at each taxel: $\hat{b}_i = \text{normalise}(w_i \hat{n}_i + (1 - w_i) \hat{v}_i)$, with $\hat{n}_i$ the taxel normal and $\hat{v}_i$ the unit vector from the CoP to taxel $i$.
  - Shear projector: $P_{shear} = I - \hat{n}_{cop} \hat{n}_{cop}^T$.
- CoP normal: taxel normals averaged with weights $1 / \|p_i - p_{cop}\|$ (inverse distance weighting, Shepard 1968).
- Inverse, taxel forces to CoP force (ridge least squares): $f_{cop} = (A^T A + \lambda^2 I)^{-1} A^T b$, with $A$ stacking the $M_i$ and $b$ stacking the $f_i$.

Source: [arXiv 2605.28812](https://arxiv.org/abs/2605.28812), from an automatic summary of the HTML version: check against the paper before implementing.

### What are lambda and sigma?

- $\sigma$ (in meters): spread of the Gaussian weights $w_i$, so the distance over which a taxel still shares the CoP force.
  - Small $\sigma$: only taxels at the CoP count; the others get $w_i \approx 0$ and their $\hat{b}_i$ turn to the radial direction $\hat{v}_i$.
  - Large $\sigma$: every taxel gets $w_i \approx 1$, so each is modelled as carrying the whole CoP force, and the solve returns about the mean taxel force instead of their sum.
- $\lambda$ (no unit): ridge regularisation (Hoerl and Kennard 1970) of the inverse solve.
  - Adds $\lambda^2$ to the diagonal of $A^T A$, so the solve stays defined when $A^T A$ is singular or close to it (few taxels, all $w_i \approx 0$).
  - Pulls $f_{cop}$ towards 0: negligible while $\lambda^2 \ll \sum_i w_i^2$, the size of $A^T A$.
  - $\lambda = 0$: plain least squares (minimum-norm solution when singular).

Source: `src/nb_touch/contact/cop_model.py`.

## Usage

Implemented in `src/nb_touch/contact/`, imported from `nb_touch.contact`. Forces are in the sensor frame, except where `in_taxel_frames` says otherwise.

- `ContactDetector(geometry, settings=ContactSettings(), active_filter=None)`: holds a sensor's normals and neighbours, computed once. Missing rotations are estimated from the positions.
- `ContactDetector.update(forces, in_taxel_frames=True)`: the contacts in one frame, largest CoP force first. With `in_taxel_frames`, forces are rotated to the sensor frame first.
- `find_contacts(positions, normals, forces, neighbours, settings=ContactSettings())`: the same for one frame, without the detector or a filter.
- `make_filter(name, threshold, off_threshold=0.1, frames=5)`: `none`, `hysteresis` or `debounce`, to pass as `active_filter`.

| Parameter         | What it does                                                     | Default                           |
| ----------------- | ---------------------------------------------------------------- | --------------------------------- |
| `geometry`        | `SensorGeometry` of the sensor.                                  | required                          |
| `settings`        | `ContactSettings`, below.                                        | `ContactSettings()`               |
| `active_filter`   | Object with `update(norms)` returning the taxels in contact.     | threshold at `settings.threshold` |
| `forces`          | Taxel forces (n, 3), in N.                                       | required                          |
| `in_taxel_frames` | Forces given in each taxel's frame, as in the recording format.  | `True`                            |
| `off_threshold`   | With hysteresis, force norm below which a taxel turns off, in N. | `0.1`                             |
| `frames`          | Number of frames required over threshold to be an active taxel.  | `5`                               |

`ContactSettings`:

| Field            | What it does                                                            | Default |
| ---------------- | ----------------------------------------------------------------------- | ------- |
| `threshold`      | Taxel force norm above which a taxel is active, in N.                   | `0.2`   |
| `min_taxels`     | Fewest taxels a contact needs.                                          | `3`     |
| `sigma`          | Spread σ of the Gaussian weights, in m.                                 | `0.003` |
| `regularisation` | Ridge λ of the CoP force solve; 0 gives the minimum-norm least squares. | `0.001` |

- `threshold` and `min_taxels`: values used on the ORCA hand in `dextHand`.
- `sigma` and `regularisation`: our choice, as the paper gives no value. σ is about one taxel spacing on the ORCA sensors: 3.11 mm (finger), 3.32 mm (thumb), 3.04 mm (pinky), median nearest-neighbour distance computed by us from the `orca_core` sensor configs.

Each `Contact` has: `taxels`, `cop`, `normal` (at the CoP), `cop_force` (from the solve), `total_force` (sum of the taxel forces), `normal_force` and `shear_force` (split of `cop_force` along `normal`), `slip_ratio`, `max_force`, `mean_force`, `count`.

The model's steps are also exposed, for example to simulate taxel forces from a CoP force: `cop_position`, `cop_normal`, `taxel_force_matrices` (the $M_i$), `cop_to_taxel_forces` and `taxel_to_cop_force`.

## Known bias

Shear makes the taxel force norms asymmetric, so the force-weighted CoP moves towards the shear and the solved force is off. Push of [0.5, 0, 2] N at the centre of a 9×9 plane, 3 mm spacing, σ = 3 mm (computed by us):

| CoP                        | CoP shift | Slip ratio (true 0.25) |
| -------------------------- | --------- | ---------------------- |
| True                       | 0         | 0.25                   |
| Estimated, all taxels      | 0.36 mm   | 0.278                  |
| Estimated, threshold 0.2 N | 0.48 mm   | 0.291                  |
