# Beyond binary touch sensing

This library is useful for: processing data of a robot equipped with taxel array sensors, as well as to detect the taxels orientation based on experimental measurements.

Hardware-agnostic python library implementing the main functionalities of the article: Pan, Jiahe, et al. "Beyond Binary: Sim-to-Real Dexterous Manipulation with Physics-Grounded Contact Representation." arXiv preprint arXiv:2605.28812 (2026).
Url: https://arxiv.org/abs/2605.28812

Main features:

- Conversion from Center of Pressure (CoP) force to individual taxel forces.
- Conversion from taxel forces to CoP force.
- Tools for automatic taxel orientation detection: this requires torque sensors for all the links active when touching the sensor.

## Conventions

- Normals at the taxels point outward, out of the sensing surface. The sensor frame and every taxel frame are right-handed, with z along the outward normal.
- Forces are the forces acting on the sensor, as the taxels feel them: a push points against the outward normal.

Be careful:

- The paper's normals $n_i$ point inward: they are the opposites of ours.
- Hardware readings must be converted to these conventions before use.

Source: [arXiv 2605.28812](https://arxiv.org/html/2605.28812v1), Sections 3.2 (inward normals, $f_{cop}$ acting on the robot link) and 3.3 ($\tau = -J^T f$).

## From taxels to resultant force (and conversly)

The resultant force is computed from the taxel forces using the CoP model presented in the paper.

Both examples use a flat 3×3 grid of taxels, 3 mm apart, numbered row by row: taxel 4 is the centre one, at the origin. Positions are in m, forces in N, both in the sensor frame.

### From taxels to CoP

With a 1 N push on the centre taxel only:

```python
import numpy as np

from nb_touch.contact import ContactDetector, ContactSettings
from nb_touch.geometry.synthetic import plane_sensor

np.set_printoptions(precision=3, suppress=True)
sensor = plane_sensor(rows=3, cols=3, spacing=0.003)
detector = ContactDetector(sensor, ContactSettings(min_taxels=1))

forces = np.zeros((9, 3))  # one force per taxel
forces[4] = [0.0, 0.0, -1.0]
contacts = detector.update(forces, in_taxel_frames=False)
print(contacts[0].cop)  # [0. 0. 0.]
print(contacts[0].cop_force)  # [ 0.  0. -1.]
```

`min_taxels=1` lets a single taxel make a contact (default: 3).

### From CoP to taxels

With a 1 N push at the centre:

```python
from nb_touch.contact import cop_normal, cop_to_taxel_forces, taxel_force_matrices
from nb_touch.geometry import normals_from_rotations

normals = normals_from_rotations(sensor.rotations)
cop = np.zeros(3)
normal = cop_normal(sensor.positions, normals, cop)
matrices = taxel_force_matrices(sensor.positions, normals, cop, normal)
print(cop_to_taxel_forces([0.0, 0.0, -1.0], matrices))
```

Expected print, one row per taxel:

```text
[[-0.225 -0.225 -0.185]
 [ 0.    -0.33  -0.509]
 [ 0.225 -0.225 -0.185]
 [-0.33   0.    -0.509]
 [ 0.     0.    -1.   ]
 [ 0.33   0.    -0.509]
 [-0.225  0.225 -0.185]
 [ 0.     0.33  -0.509]
 [ 0.225  0.225 -0.185]]
```

The centre taxel gets the whole push. The others get less the further they are, tilted away from the CoP.

For more details, see [docs/theory/cop.md](docs/theory/cop.md).

## CoP examples

An example script for the taxel to CoP conversion can be found in `examples/cop_in_mujoco.py`. It provides the taxel-CoP mapping on a synthetic sensor, shown in MuJoCo. Run it in an activated terminal with `python examples/cop-in-mujoco.py`.

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

### Use recording to compute the taxel orientations

<!--- todo: example -->

## Recording calibration data on a real hand

This library was tested on a real robotic hand: the Orca hand touch. The code and results are accessible here:

<!--- todo: add link once code is public -->
