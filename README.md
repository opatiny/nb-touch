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

From taxels to CoP:

<!--- todo: code example -->

From CoP to taxels:

<!--- todo: code example -->

For more details, see <!--- todo: add link to md -->

## CoP examples

An example script for the taxel to CoP converstion can be found in `examples/cop_in_mujoco.py`. It provides the taxel-CoP mapping on a synthetic sensor, shown in MuJoCo. Run it in an activated terminal with `python examples/cop-in-mujoco.py`.

It shows:

- **gray spheres:** all the sensor taxels
- **yellow spheres**: taxels in a contact zone
- **blue arrows**: resultant forces for each taxel
- **green arrow and dot**: true CoP force and position
- **red arrow and dot**: estimated CoP force and position, recomputed from the taxel forces

Why the green and red arrows don't match perfectly? See <!--- todo: add link to md -->

### Planar sensor

### Cylindrical sensor

### Spherical sensor

## Taxel orientation detection

### File formats

### Reading and writing a recording

Implemented in `src/nb_touch/io/recordings.py`, imported from `nb_touch.io`.

- `save_recording(recording, path)`: writes `<path>.npz` and `<path>.yaml` and returns both paths.
- `load_recording(path)`: reads both files. Raises a `FileNotFoundError` if one is missing, and a `ValueError` if an array is missing or shapes disagree.

Save recording:

<!--- todo: code example replacing above lines -->

Load recording:

<!--- todo: code example -->

### Use recording to compute the taxel orientations

<!--- todo: example -->

## Recording calibration data on a real hand

This library was tested on a real robotic hand: the Orca hand touch. The code and results are accessible here:

<!--- todo: add link once code is public -->
