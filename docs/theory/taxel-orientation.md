# Taxel orientation calibration

Estimating the rotation $R_i$ of each taxel in the sensor frame from recorded taxel forces, joint angles and joint torques, without a force sensor.

## Method in the paper

- $R_i$ is a free 3×3 matrix, projected to a rotation with [`project_to_so3`](so3-projection.md).
- Data: 2400 samples (2 min at 20 Hz) of random fingertip contacts, with taxel forces, joint torques $\tau \in \mathbb{R}^4$ and joint angles $q \in \mathbb{R}^4$ of one finger.
- Static equilibrium: $\tau = -J^T f + g(q) \approx -J^T f$, gravity being neglected.
- Prediction: $\hat\tau = -\hat{J}_{cop}^T \hat{f}_{cop}$, with the Jacobian at the estimated CoP and the estimated [CoP force](cop.md).
- Loss: MSE between $\hat\tau$ and the recorded joint torques. Adam, learning rate 0.1, 100 steps.

Source: [arXiv 2605.28812](https://arxiv.org/html/2605.28812v1), calibration section.

$\hat{f}_{cop}$ depends on the $R_i$ through the ridge solve, so this loss is not linear in $R_i$ and needs an iterative optimiser.

## Solver

Same as the paper, with torch:

- Parameters: one free 3×3 matrix per taxel, initialised from the prior (normals estimated from the geometry).
- Each step: project to rotations, compute the CoP force with the ridge solve, predict $\hat\tau$, compute the MSE loss.
- torch computes the gradient of the loss automatically (autograd), and Adam updates the matrices: learning rate 0.1, 100 steps.
- The final matrices are projected with `project_to_so3`.

## Recording format

Data recorded for the calibration: one `.npz` file for the arrays, plus a small YAML file for the metadata.

### Arrays (`.npz`)

| Array             | Shape     | Unit                     |
| ----------------- | --------- | ------------------------ |
| `timestamps_s`    | (N,)      | s                        |
| `joint_pos_rad`   | (N, J)    | rad                      |
| `joint_torque_nm` | (N, J)    | N·m                      |
| `taxel_forces_n`  | (N, T, 3) | N, in each taxel's frame |

N samples, J joints, T taxels.

### Metadata (`.yaml`)

Sensor model, joint names, link the sensor is mounted on, sampling rate, date.

### Why npz?

npz is numpy's own format: a zip of binary arrays, written with `np.savez_compressed` and read with `np.load` ([numpy docs](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html)).

|                                          | npz                | JSON                            | CSV                                            |
| ---------------------------------------- | ------------------ | ------------------------------- | ---------------------------------------------- |
| Size of the paper's 2400 × 87 × 3 forces | about 5 MB or less | about 11 MB                     | about 11 MB                                    |
| Shapes and dtypes                        | kept               | lost (nested lists)             | lost, 2D only: forces flattened to 261 columns |
| Precision, NaN                           | exact              | rounded, no NaN in the standard | rounded unless written at full precision       |
| Human-readable                           | no                 | yes                             | yes, also in Excel                             |
| Files per recording                      | one                | one                             | one per array, rows to keep aligned            |

Sizes computed by us: 626,400 float64 values, and about 18 characters per number in JSON or CSV.

YAML or JSON remain the choice for small files, such as [sensor geometry](../formats/sensor-geometry.md) and metadata.

## Kinematics

The library takes the Jacobian through an interface, `jacobian(q, link, point) -> (3, J)`, so it stays hardware-agnostic. A MuJoCo implementation (`mj_jac`) can come with the `viz` extra.

## Edge cases to test

- Synthetic 4-DoF chain with known rotations: they must be recovered.
- Noise on forces and torques.
- Taxels never active: they keep the prior.
- Poorly excited joints, reflections (det = -1).
