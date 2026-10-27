# Synthetic recording

Simulating a recording of random fingertip contacts with known taxel orientations, to test the [taxel orientation calibration](taxel-orientation.md). Implemented in `nb-touch`, `src/nb_touch/calibration/synthetic.py`.

## Finger

- 4 revolute joints, like a human finger: abduction about $z$, then 3 flexion joints about $x$. The finger runs along $-y$ from the sensor, with joint axes through $y = -0.1, -0.1, -0.055, -0.02$ m and $z = -0.005$ m, in the sensor frame (chosen by us).
- The finger is held still, as in the paper's data collection: a stiff PD controller holds the joints while an object is pressed on the fingertip. So the Jacobian does not depend on the joint angles, recorded as 0.
- Jacobian column $j$: $J_j = a_j \times (c - o_j)$, with $a_j$ the unit axis of joint $j$, $o_j$ a point on it and $c$ the contact point.
- Sampling rate: 20 Hz, as in the paper.

Source: [arXiv 2605.28812](https://arxiv.org/html/2605.28812v1), Appendix A.

## True orientations

The true orientation of a taxel is its orientation on the real hardware, which the calibration has to find. Here it is simulated: each taxel orientation of the geometry is turned about a random axis of its own frame, by a random angle uniform in $[0, \theta_{max}]$, with $\theta_{max}$ the largest error given.

## Samples

For each sample:

- CoP $c$: the position of a random taxel.
- CoP force: $f_{cop} = -p\,n + s$, with:
  - $n$: unit outward normal at $c$
  - $p$: push, uniform in $[0.5, 3]$ N (default)
  - $s$: shear, perpendicular to $n$, in a random direction, of norm uniform in $[0, 0.5\,p]$ (default)
- Taxel forces: $f_{cop}$ spread over the taxels with the [CoP model](../cop/cop.md), in the sensor frame, then taken to each taxel frame with the true orientation $R_i$ (rotation from the taxel frame to the sensor frame): $f_i = R_i^T f_i^{sensor}$.
- Joint torques: $\tau = -J(\hat c)^T \hat f_{cop}$, gravity neglected as in the paper (Section 3.3), with $\hat c$ and $\hat f_{cop}$ the CoP and CoP force estimated back from the taxel forces, as the calibration does. With the true orientations, the calibration's model then explains the torques exactly, without the [known bias](../cop/cop.md#known-bias) of the estimate.
- Noise (optional): Gaussian, with a standard deviation in N on each force component and in N·m on each torque, 0 by default.

Push range, shear and noise values are chosen by us.

## Example

See [Generate a recording file](../../README.md#generate-a-recording-file) in the README.
