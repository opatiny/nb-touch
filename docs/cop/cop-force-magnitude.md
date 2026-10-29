# Magnitude of CoP force is counter-intuitive

Why the CoP force of a contact is close to one taxel's force instead of the sum of the taxel forces.

## Problem

- The paper defines $f_{cop}$ as the total contact force on the link: "representing the total contact force acting on the robot link", "the resultant contact wrench reduced to a single force vector applied at a centroidal contact point" ([arXiv 2605.28812](https://arxiv.org/abs/2605.28812), Section 3.1 and footnote 1).
- Its inverse model (Section 3.2), applied to taxel forces, returns about the mean taxel force, not their sum.

## Experimental values

We apply a uniform push of 1 N per taxel, flat sensor, 3 mm spacing, default settings. The table below compares the CoP force and the sum of the taxels in the sensor frame.

| Taxels    | $\|f_{cop}\|$ | Taxel sum |
| --------- | ------------- | --------- |
| 1         | 1.00 N        | 1 N       |
| 3 (row)   | 1.12 N        | 3 N       |
| 10 (row)  | 1.16 N        | 10 N      |
| 30 (row)  | 1.16 N        | 30 N      |
| 9 (3 × 3) | 1.22 N        | 9 N       |
