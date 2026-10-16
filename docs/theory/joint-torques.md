# Joint torques

The [taxel orientation calibration](taxel-orientation.md) needs joint torques. `nb-touch` takes them as input: any mapping from motor current to joint torque is up to the user.

## In the paper

- The paper records "applied joint torques" while "the joint actuators actively apply torques to balance the external force", but does not say how they are measured ([arXiv 2605.28812](https://arxiv.org/html/2605.28812v1)).
- On the Allegro hand used there, each joint has its own geared motor, so actuator and joint torques are the same.

## On the ORCA hand

- Motor torque from current: `dextHand`, `docs/tools/measure-motor-torques.md`, code in `src/dexthand/hand/torques.py`.
- Joint torque from motor torque, by virtual work with the calibrated joint-to-motor ratio: same doc, section "Joint torque from motor torque".

Main difficulties:

- Tendon and gearbox friction can be as large as the contact torques.
- Current readings are noisy and non-linear at low load.
- The finger's own weight must be subtracted.
- Gain errors, offsets and friction all bias the calibration: the rotations are projected at every step, so they cannot absorb a wrong torque scale.
