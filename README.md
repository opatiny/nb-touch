# Beyond binary touch sensing

This library is useful for: processing data of a robot equipped with taxel array sensors, as well as to calibrate these sensors.

Hardware-agnostic python library implementing the main functionalities of the article Pan, Jiahe, et al. "Beyond Binary: Sim-to-Real Dexterous Manipulation with Physics-Grounded Contact Representation." arXiv preprint arXiv:2605.28812 (2026).

The features include:

- Conversion from Center of Pressure (CoP) to individual taxel forces
- Conversion from taxel forces to CoP force
- Tools for automatic taxel orientation detection: this requires torque sensors for all the links to the touch sensor.
