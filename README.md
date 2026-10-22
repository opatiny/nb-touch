# Beyond binary touch sensing

This library is useful for: processing data of a robot equipped with taxel array sensors, as well as to detect the taxels orientation based on experimental measurements.

Hardware-agnostic python library implementing the main functionalities of the article: Pan, Jiahe, et al. "Beyond Binary: Sim-to-Real Dexterous Manipulation with Physics-Grounded Contact Representation." arXiv preprint arXiv:2605.28812 (2026).
Url: https://arxiv.org/abs/2605.28812

Main features:

- Conversion from Center of Pressure (CoP) force to individual taxel forces.
- Conversion from taxel forces to CoP force.
- Tools for automatic taxel orientation detection: this requires torque sensors for all the links active when touching the sensor.

## From taxels to resultant force (and conversly)

The resultant force is computed from the taxel forces using the CoP model presented in the paper.

Both examples use a flat 3×3 grid of taxels, 3 mm apart, numbered row by row: taxel 4 is the centre one, at the origin. Positions are in m, forces in N, both in the sensor frame.

From taxels to CoP, with a 1 N push on the centre taxel only:

```python
import numpy as np

from nb_touch.contact import ContactDetector, ContactSettings
from nb_touch.geometry.synthetic import plane_sensor

np.set_printoptions(precision=3, suppress=True)
sensor = plane_sensor(rows=3, cols=3, spacing=0.003)
detector = ContactDetector(sensor, ContactSettings(min_taxels=1))

forces = np.zeros((9, 3))  # one force per taxel
forces[4] = [0.0, 0.0, 1.0]
contacts = detector.update(forces, in_taxel_frames=False)
print(contacts[0].cop)  # [0. 0. 0.]
print(contacts[0].cop_force)  # [0. 0. 1.]
```

`min_taxels=1` lets a single taxel make a contact (default: 3).

From CoP to taxels, with a 1 N push at the centre:

```python
from nb_touch.contact import cop_normal, cop_to_taxel_forces, taxel_force_matrices
from nb_touch.geometry import normals_from_rotations

normals = normals_from_rotations(sensor.rotations)
cop = np.zeros(3)
normal = cop_normal(sensor.positions, normals, cop)
matrices = taxel_force_matrices(sensor.positions, normals, cop, normal)
print(cop_to_taxel_forces([0.0, 0.0, 1.0], matrices))
```

Expected print, one row per taxel:

```text
[[-0.225 -0.225  0.185]
 [ 0.    -0.33   0.509]
 [ 0.225 -0.225  0.185]
 [-0.33   0.     0.509]
 [ 0.     0.     1.   ]
 [ 0.33   0.     0.509]
 [-0.225  0.225  0.185]
 [ 0.     0.33   0.509]
 [ 0.225  0.225  0.185]]
```

The centre taxel gets the whole push. The others get less the further they are, tilted away from the CoP.

For more details, see [docs/theory/cop.md](docs/theory/cop.md).

## CoP examples

An example script for the taxel to CoP converstion can be found in `examples/cop_in_mujoco.py`. It provides the taxel-CoP mapping on a synthetic sensor, shown in MuJoCo. Run it in an activated terminal with `python examples/cop-in-mujoco.py`.

It shows:

- **gray spheres:** all the sensor taxels
- **yellow spheres**: taxels in a contact zone
- **blue arrows**: resultant forces for each taxel
- **green arrow and dot**: true CoP force and position
- **red arrow and dot**: estimated CoP force and position, recomputed from the taxel forces

Why the green and red arrows don't match perfectly? See [Known bias](docs/theory/cop.md#known-bias).

### Planar sensor

### Cylindrical sensor

### Spherical sensor

## Taxel orientation detection

### File formats

### Reading and writing a recording

Implemented in `src/nb_touch/io/recordings.py`, imported from `nb_touch.io`.

A recording holds N samples of J joints and T taxels: the sample times, joint angles, joint torques and taxel forces (each in its taxel's frame), plus metadata. Format: [Recording format](docs/theory/taxel-orientation.md#recording-format).

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

### Use recording to compute the taxel orientations

<!--- todo: example -->

## Recording calibration data on a real hand

This library was tested on a real robotic hand: the Orca hand touch. The code and results are accessible here:

<!--- todo: add link once code is public -->
