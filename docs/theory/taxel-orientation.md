# Taxel orientation calibration

<!-- todo: declutter this file -->

Estimating the rotation $R_i$ of each taxel relative to the sensor frame. The calibration is solely based on a recording of taxel forces, joint angles and joint torques when an object is moved on the sensing surface.

## Method in the paper

- $R_i$ is a free 3×3 matrix, projected to a rotation with [`project_to_so3`](so3-projection.md).
- Data: 2400 samples (2 min at 20 Hz) of random fingertip contacts, with taxel forces, joint torques $\tau \in \mathbb{R}^4$ and joint angles $q \in \mathbb{R}^4$ of one finger.
- Static equilibrium: $\tau = -J^T f + g(q) \approx -J^T f$, gravity being neglected, with $f$ the force acting on the finger ([Conventions](../../README.md#conventions)).
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

## Reading and writing a recording

Implemented in `src/nb_touch/io/recordings.py`, imported from `nb_touch.io`.

A recording holds N samples of J joints and T taxels: the sample times, joint angles, joint torques and taxel forces (each in its taxel's frame), plus metadata. Format: [Recording format](#recording-format).

Save a recording of 3 samples (20 Hz), 2 joints and 2 taxels:

```python
from nb_touch.io import Recording, save_recording

recording = Recording(
    timestamps_s=[0.0, 0.05, 0.1],  # (N,), in s
    joint_pos_rad=[[0.1, 0.2], [0.1, 0.3], [0.2, 0.3]],  # (N, J), in rad
    joint_torque_nm=[[0.01, 0.02], [0.01, 0.03], [0.02, 0.03]],  # (N, J), in N·m
    taxel_forces_n=[  # (N, T, 3), in N
        [[0, 0, 0.5], [0, 0, 0.2]],
        [[0, 0, 0.6], [0, 0, 0.3]],
        [[0, 0, 0.7], [0, 0, 0.4]],
    ],
    metadata={"joint_names": ["mcp", "pip"]},
)
save_recording(recording, "results/recording")
```

Saved files:

- `results/recording.npz`: the 4 arrays, `timestamps_s` (3,), `joint_pos_rad` (3, 2), `joint_torque_nm` (3, 2) and `taxel_forces_n` (3, 2, 3).
- `results/recording.yaml`: the metadata.

```yaml
joint_names: [mcp, pip]
```

Load it back:

```python
from nb_touch.io import load_recording

recording = load_recording("results/recording")
recording.num_samples, recording.num_joints, recording.num_taxels  # (3, 2, 2)
recording.taxel_forces_n[0]  # forces of both taxels in the first sample
```

`load_recording` raises a `FileNotFoundError` if one of the two files is missing, and a `ValueError` if an array is missing or the shapes disagree.

## Kinematics

The library takes the Jacobian through an interface, `jacobian(q, link, point) -> (3, J)`, so it stays hardware-agnostic. A MuJoCo implementation (`mj_jac`) can come with the `viz` extra.

## Edge cases to test

- Synthetic 4-DoF chain with known rotations: they must be recovered.
- Noise on forces and torques.
- Taxels never active: they keep the prior.
- Poorly excited joints, reflections (det = -1).
