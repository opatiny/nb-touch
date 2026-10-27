# Beyond binary touch sensing

[![PyPI](https://img.shields.io/pypi/v/nb-touch)](https://pypi.org/project/nb-touch/)

This library is useful for: processing data of a robot equipped with taxel array sensors, as well as to detect the taxels orientation based on experimental measurements.

Hardware-agnostic python library implementing the main functionalities of the article: Pan, Jiahe, et al. "Beyond Binary: Sim-to-Real Dexterous Manipulation with Physics-Grounded Contact Representation." arXiv preprint arXiv:2605.28812 (2026).
Url: https://arxiv.org/abs/2605.28812

Main features:

- Conversion from Center of Pressure (CoP) force to individual taxel forces.
- Conversion from taxel forces to CoP force.
- Tools for automatic taxel orientation detection: this requires torque sensors for all the links active when touching the sensor.

## Installation

- `pip install nb-touch`: CoP model, contact detection, geometry and I/O.
- `pip install "nb-touch[calibration]"`: adds the taxel orientation calibration (needs `torch`).
- `pip install "nb-touch[viz]"`: adds the MuJoCo drawing helpers used by the example (needs `mujoco`).

## Conventions

- Normals at the taxels point outward, out of the sensing surface. The sensor frame and every taxel frame are right-handed, with z along the outward normal.
- Forces are the forces acting on the sensor, as the taxels feel them: a push points against the outward normal.

Be careful:

- The paper's normals $n_i$ point inward: they are the opposites of ours.
- Hardware readings must be converted to these conventions before use.

Source: [arXiv 2605.28812](https://arxiv.org/html/2605.28812v1), Sections 3.2 (inward normals, $f_{cop}$ acting on the robot link) and 3.3 ($\tau = -J^T f$).

## From taxels to resultant force (and conversely)

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
from nb_touch.contact import (
    cop_normal,
    cop_to_taxel_forces,
    spacing_sigma,
    taxel_force_matrices,
)
from nb_touch.geometry import normals_from_orientations

normals = normals_from_orientations(sensor.orientations)
cop = np.zeros(3)
normal = cop_normal(sensor.positions, normals, cop)
sigma = spacing_sigma(sensor.positions)  # 1.5 x the median taxel spacing
matrices = taxel_force_matrices(sensor.positions, normals, cop, normal, sigma)
print(cop_to_taxel_forces([0.0, 0.0, -1.0], matrices))
```

Expected print, one row per taxel:

```text
[[-0.221 -0.221 -0.56 ]
 [ 0.    -0.193 -0.777]
 [ 0.221 -0.221 -0.56 ]
 [-0.193  0.    -0.777]
 [ 0.     0.    -1.   ]
 [ 0.193  0.    -0.777]
 [-0.221  0.221 -0.56 ]
 [ 0.     0.193 -0.777]
 [ 0.221  0.221 -0.56 ]]
```

The centre taxel gets the whole push. The others get less the further they are, tilted away from the CoP.

For more details, see [docs/cop/cop.md](https://github.com/opatiny/nb-touch/blob/main/docs/cop/cop.md).

## CoP examples

An example script for the taxel to CoP conversion can be found in [`examples/cop_in_mujoco.py`](https://github.com/opatiny/nb-touch/blob/main/examples/cop_in_mujoco.py). It provides the taxel-CoP mapping on a synthetic sensor, shown in MuJoCo. It needs the viz extra. Run it in an activated terminal with `python examples/cop_in_mujoco.py` (on macOS: `mjpython examples/cop_in_mujoco.py`).

It shows:

- **gray spheres:** all the sensor taxels
- **yellow spheres**: taxels in a contact zone
- **blue arrows**: resultant forces for each taxel
- **green arrow and dot**: true CoP force and position
- **red arrow and dot**: estimated CoP force and position, recomputed from the taxel forces

Why the green and red arrows don't match perfectly? See [Known bias](https://github.com/opatiny/nb-touch/blob/main/docs/cop/cop.md#known-bias).

### Planar sensor

https://github.com/user-attachments/assets/189b1fb1-31b1-47bc-b6a7-87e07696a47e

### Cylindrical sensor

https://github.com/user-attachments/assets/d66bacdf-331c-4f35-a88c-500e3954136c

### Spherical sensor

https://github.com/user-attachments/assets/cd5f517f-bf19-4a19-b95b-86a3a4ac1ba7

## Taxel orientation detection

### Generate a recording file

A recording holds, for each sample, the taxel forces (each in its taxel's frame), the joint angles and the joint torques of the finger carrying the sensor, while an object is pressed at random places on the sensor. The paper uses 2400 samples: 2 min at 20 Hz ([arXiv 2605.28812](https://arxiv.org/html/2605.28812v1), Appendix A).

File format: [Recording format](https://github.com/opatiny/nb-touch/blob/main/docs/taxelCalib/taxel-orientation.md#recording-format).

Synthetic recording of 2400 samples on the 3×3 grid above, mounted on a 4-joint finger, with true taxel orientations (the orientations on the real hardware) up to 20° away from the geometry's:

```python
from nb_touch.calibration.synthetic import perturbed_orientations, synthetic_recording
from nb_touch.io import save_recording

rng = np.random.default_rng(0)
true_orientations = perturbed_orientations(sensor.orientations, np.radians(20), rng)
recording = synthetic_recording(sensor, true_orientations, num_samples=2400, rng=rng)
save_recording(recording, "results/recording")  # recording.npz and recording.yaml
```

How the joint torques and taxel forces are simulated: [docs/taxelCalib/synthetic-recording.md](https://github.com/opatiny/nb-touch/blob/main/docs/taxelCalib/synthetic-recording.md).

With real hardware, build a `Recording` from the measured arrays instead (see [Reading and writing a recording](https://github.com/opatiny/nb-touch/blob/main/docs/taxelCalib/taxel-orientation.md#reading-and-writing-a-recording)).

### Use recording to compute the taxel orientations

The calibration needs the recording, the sensor geometry (orientations are optional, they are only the starting point) and a function `jacobian(q, link, point)` returning the `(3, J)` position Jacobian of a point of the sensor, in the sensor frame. Here, the Jacobian of the synthetic finger:

```python
from nb_touch.calibration import calibrate_orientations
from nb_touch.calibration.synthetic import finger_jacobian
from nb_touch.geometry.transforms import angles_between_rotations
from nb_touch.io import load_recording

recording = load_recording("results/recording")
result = calibrate_orientations(recording, sensor, finger_jacobian)
before = angles_between_rotations(sensor.orientations, true_orientations)
after = angles_between_rotations(result.geometry.orientations, true_orientations)
print(np.degrees(before).max().round(1), np.degrees(after).max().round(1))  # 19.6 6.4
```

`result.geometry` is the sensor geometry with the calibrated orientations, `result.losses` the torque MSE at each step. The paper's 100 Adam steps are the default; with `settings=CalibrationSettings(steps=1000)`, the largest is expected to drop to 1.0°.

For more details, see [docs/taxelCalib/taxel-orientation.md](https://github.com/opatiny/nb-touch/blob/main/docs/taxelCalib/taxel-orientation.md).

## Recording calibration data on a real hand

This library was tested on a real robotic hand: the Orca hand touch. A small report of the experiment is in [docs/orcahand-experiment](https://github.com/opatiny/nb-touch/blob/main/docs/orcahand-experiment/README.md).
