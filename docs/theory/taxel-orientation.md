# Taxel orientation calibration

<!--- todo: merge recordings.ms with this doc -->

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

## Kinematics

The library takes the Jacobian through an interface, `jacobian(q, link, point) -> (3, J)`, so it stays hardware-agnostic. A MuJoCo implementation (`mj_jac`) can come with the `viz` extra.

## Edge cases to test

- Synthetic 4-DoF chain with known rotations: they must be recovered.
- Noise on forces and torques.
- Taxels never active: they keep the prior.
- Poorly excited joints, reflections (det = -1).
