# Contacts and CoP

From a set of taxels with their xyz forces to N contacts on a sensor, each with its centre of pressure (CoP), CoP force and stats.

## Pipeline

1. **Active taxels**: taxels whose force norm is above a threshold, optionally filtered over time (hysteresis or debounce) so contacts do not flicker.
2. **Zones**: active taxels grouped into connected groups of neighbours. Each zone is one contact.
3. **Per zone**:
   - CoP position: taxel positions averaged with the force norms as weights.
   - CoP force: the paper's solve (below), restricted to the zone's taxels.
   - Stats: total force, normal force, shear force, slip ratio (shear / normal), max and mean taxel force, taxel count, contact normal.

Source: steps 1, 2 and the stats come from `dextHand` (`src/dexthand/touch/contact.py`, `contact_zone.py`, `contact_filters.py`), used by `tools/touch/detect_touch_contact.py`, which `nb-touch` will replace.

Limitation: two contacts on adjacent taxels merge into one zone. Splitting them needs a local-maxima step.

## CoP model (Pan et al. 2026)

Per sensor, the contact is a 3D force and a 3D position in the sensor frame. Only the normal component is used, as simulated shear is unreliable.

- Active taxels: $\mathcal{A} = \{i : \|f_i\| > \epsilon\}$.
- Forward model, CoP force to taxel forces: $f_i = M_i f_{cop}$, with $M_i = w_i (\hat{b}_i \hat{n}_{cop}^T + P_{shear})$.
- Weights: $w_i = \exp(-\|p_i - p_{cop}\|^2 / (2\sigma^2))$.
- Inverse, taxel forces to CoP force (ridge least squares): $f_{cop} = (A^T A + \lambda^2 I)^{-1} A^T b$, with $A$ stacking the $M_i$ and $b$ stacking the $f_i$.

Source: [arXiv 2605.28812](https://arxiv.org/abs/2605.28812), from an automatic summary of the HTML version: check against the paper before implementing.

## Edge cases to test

- No active taxel, a single taxel, all-zero forces.
- $\lambda = 0$, very small or very large $\sigma$.
- CoP outside the taxel array.
- Taxels on a plane, a cylinder and a sphere.
