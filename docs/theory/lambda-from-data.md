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

## When does the value of lambda matter?

Only when $\sigma$ is small compared with the taxel spacing $s$. With $e$ the smallest eigenvalue of $A^T A$:

- Ridge shrinks $f_{cop}$ by a factor $\lambda^2 / (e + \lambda^2)$ along the matching eigenvector.
- Noise on $b$ is amplified by $\sqrt{e} / (e + \lambda^2)$ along it, so $1 / \sqrt{e}$ for $\lambda = 0$.
- So $\lambda$ only matters when $\lambda^2$ is not negligible against $e$.

Smallest $e$ on a 9×9 plane, $s$ = 3 mm, CoP between taxels, zones of 3 to 81 taxels (computed by us with `src/nb_touch/contact/cop_model.py`):

| $\sigma$          | Smallest $e$       |
| ----------------- | ------------------ |
| 1.5 $s$ (default) | 1.67               |
| $s$               | 0.83               |
| 0.5 $s$           | 0.05               |
| 0.33 $s$          | $1.3\cdot10^{-3}$  |
| 0.17 $s$          | $4.6\cdot10^{-12}$ |

With the default $\sigma$ and $\lambda = 10^{-3}$, the shrinkage is below $10^{-6}$: the value of $\lambda$ does not matter. It starts to matter for $\sigma \lesssim s/3$, where every $w_i \approx 0$.
