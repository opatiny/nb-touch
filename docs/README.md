# Documentation

Index of the `nb-touch` docs. Usage and conventions: see the [main README](../README.md).

## Sensor and contacts

- [Sensor geometry](cop/sensor-geometry.md): file format of taxel positions and orientations, and the utilities to process them.
- [Contacts and CoP](cop/cop.md): from taxel xyz forces to contacts, each with its centre of pressure (CoP) and force.
- [Estimating λ from the data](cop/lambda-from-data.md): choosing the ridge regularisation of the CoP force solve.

## Taxel orientation calibration

- [Taxel orientation calibration](taxelCalib/taxel-orientation.md): estimating the rotation of each taxel relative to the sensor frame from a recording.
- [Projection to SO(3)](taxelCalib/so3-projection.md): closest rotation matrix to any 3×3 matrix.
- [Synthetic recording](taxelCalib/synthetic-recording.md): simulating a recording of random fingertip contacts to test the calibration.
- [Orca hand experiment](orcahand-experiment/README.md): testing the calibration on a real Orca hand touch.

## Other

- [Visualisation in MuJoCo](mujoco-viz.md): drawing taxels, taxel forces and contacts in MuJoCo.
- [Development](dev.md): development setup.
