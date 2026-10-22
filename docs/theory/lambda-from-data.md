# Estimating a good λ from the data

Ways to choose the ridge regularisation $\lambda$ of the CoP force solve from sensor data. Model and symbols: see [Contacts and CoP](cop.md).

## Inputs

- $\sigma_n$ (N): noise standard deviation of one taxel force component. Two ways to estimate it:
  - from a recording with no contact: standard deviation of each taxel force component;
  - per contact, from the fit residuals: $\hat\sigma_n^2 = \|b - A \hat f\|^2 / (3m - 3)$.
- $|\hat f|$ (N): size of the CoP force, from the plain least-squares estimate $\hat f$ ($\lambda = 0$) of the contact.

With:

- $b$: the zone's taxel forces, stacked, $(3m,)$, in N.
- $A$: the forward-model matrix of the zone, $(3m, 3)$, no unit.
- $m$: number of taxels in the zone, so $3m$ equations for 3 unknowns.

## Methods

**Hoerl–Kennard–Baldwin**:[Hoerl, Kennard and Baldwin 1975](https://doi.org/10.1080/03610927508827232) provide a simple equation for a suitable lambda value:
$\lambda^2 = 3 \hat\sigma_n^2 / |\hat f|^2$

- The 3 is the number of unknowns.
- Cheap enough to compute per contact and per frame, giving an adaptive $\lambda$.

However, the **ground truth** for the optimal lambda could be found experimentally. Use known loads (for example a force sensor during calibration) and take the $\lambda$ minimising the error on $f_{cop}$.

## When the value of lambda matters?

Not worth it with the default spread of the Gaussian weights, $\sigma$ = 1.5 times the median taxel spacing ([Default values](cop.md#default-values)), so 4.5 mm on a 3 mm grid. $\sigma$ is not the taxel noise $\sigma_n$: it sets the weight $w_i = \exp(-\|p_i - p_{cop}\|^2 / (2\sigma^2))$ of taxel $i$, with $p_i$ its position and $p_{cop}$ the CoP.

- The eigenvalues $e$ of $A^T A$ are at least about 2 (9×9 plane, 3 mm spacing, σ = 4.5 mm, zones of 3 to 81 taxels, computed by us with `src/nb_touch/contact/cop_model.py`).
- Along each eigenvector, the solve shrinks $f_{cop}$ by $\lambda^2 / (e + \lambda^2)$: below 1% for any $\lambda \le 0.1$.
- A data-driven $\lambda$ only pays off when all $w_i \approx 0$: with $\sigma$ much smaller than the taxel spacing, or zones far from every taxel.
