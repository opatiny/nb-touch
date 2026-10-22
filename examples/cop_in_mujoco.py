# Shows the taxel-CoP mapping on a synthetic sensor in the MuJoCo viewer.
#
# A true CoP force (green) moves around the sensor and its shear turns. The model's
# forward direction turns it into taxel forces (blue arrows), with optional noise.
# The contact detection then estimates the CoP and CoP force back from the taxel
# forces alone (red). Taxels in a contact are yellow.
#
# Needs the viz extra (uv sync --extra viz). Edit the constants, then run, on macOS:
#   uv run mjpython examples/cop_in_mujoco.py
# and on Linux or Windows:
#   uv run python examples/cop_in_mujoco.py

import time

import mujoco
import mujoco.viewer
import numpy as np

from nb_touch.contact import ContactDetector, ContactSettings, cop_normal
from nb_touch.contact.cop_model import cop_to_taxel_forces, taxel_force_matrices
from nb_touch.geometry import normals_from_rotations
from nb_touch.geometry.synthetic import synthetic_sensor
from nb_touch.viz.markers import draw_spheres
from nb_touch.viz.sensor import (
    ForceView,
    draw_contacts,
    draw_cop,
    draw_taxel_forces,
    draw_taxels,
)

# simulation options
SHAPE = "plane"
"""Synthetic sensor: "plane", "cylinder" or "sphere"."""

NUM_TAXELS = 25
"""Number of taxels."""

NORMAL_FORCE_N = 1
"""Push of the true CoP force, against the outward normal, in N."""

SHEAR_FORCE_N = 0
"""Shear part of the true CoP force, in N. Its direction turns over time."""

NOISE_N = 0.0
"""Standard deviation of the noise added to each taxel force component, in N."""

PATH_RADIUS_M = 0.00
"""Radius of the circle the true CoP follows, seen from above, in m."""

PERIOD_S = 8.0
"""Time for the true CoP to go once around its circle, in s."""

MOVING = False
"""False to keep the true CoP and CoP force still, at their position at time 0."""

# contact settings
SETTINGS = ContactSettings(min_taxels=1, sigma=0.01, regularisation=0.05)
"""Contact detection settings, see `ContactSettings` in
`src/nb_touch/contact/contacts.py`. The same σ is used to make the taxel forces."""

# display options
COP_SCALE_M_PER_N = 0.006
"""CoP force arrow length per newton, in m."""

TAXEL_SCALE_M_PER_N = 0.02
"""Taxel force arrow length per newton, in m."""

FORCE_VIEW = ForceView.SENSOR_REACTION_FORCE
"""Arrows for the forces acting on the sensor, or for their opposite
(`ForceView.SENSOR_REACTION_FORCE`)."""

TRUE_RGBA = (0.2, 0.8, 0.2, 1.0)
"""Green: the true CoP and CoP force."""

RATE_HZ = 30.0
"""Viewer refresh rate, in Hz."""

MODEL = """
<mujoco>
  <visual><headlight ambient="0.5 0.5 0.5"/></visual>
  <worldbody><light pos="0 0 0.5"/></worldbody>
</mujoco>
"""


def true_cop(positions, normals, time_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the true CoP and its normal at `time_s`.

    The CoP follows a circle seen from above, lifted onto the sensor as the taxel
    positions averaged with Gaussian weights on their distance from above.
    """
    angle = 2 * np.pi * time_s / PERIOD_S
    target = PATH_RADIUS_M * np.array([np.cos(angle), np.sin(angle)])
    offsets = positions[:, :2] - target
    weights = np.exp(-np.sum(offsets**2, axis=1) / (2 * SETTINGS.sigma**2))
    cop = weights @ positions / weights.sum()
    return cop, cop_normal(positions, normals, cop)


def true_cop_force(normal, time_s: float) -> np.ndarray:
    """Return the true CoP force: a push against the outward `normal` plus a
    turning shear."""
    across = np.cross(normal, [1.0, 0.0, 0.0])
    if np.linalg.norm(across) < 1e-6:
        across = np.cross(normal, [0.0, 1.0, 0.0])
    across /= np.linalg.norm(across)
    other = np.cross(normal, across)
    angle = 2 * np.pi * time_s / (PERIOD_S / 3)
    shear = np.cos(angle) * across + np.sin(angle) * other
    return -NORMAL_FORCE_N * normal + SHEAR_FORCE_N * shear


def taxel_forces(positions, normals, cop, normal, cop_force, rng) -> np.ndarray:
    """Return the taxel forces of the model's forward direction, plus noise."""
    matrices = taxel_force_matrices(positions, normals, cop, normal, SETTINGS.sigma)
    forces = cop_to_taxel_forces(cop_force, matrices)
    return forces + rng.normal(scale=NOISE_N, size=forces.shape)


def draw_frame(scene, positions, forces, contacts, cop, cop_force) -> None:
    """Redraw the taxels, taxel forces, true CoP and estimated contacts."""
    scene.ngeom = 0
    active = np.zeros(len(positions), dtype=bool)
    for contact in contacts:
        active[contact.taxels] = True
    draw_taxels(scene, positions, active)
    draw_taxel_forces(
        scene, positions, forces, TAXEL_SCALE_M_PER_N, NOISE_N * 3, FORCE_VIEW
    )
    draw_cop(scene, cop, cop_force, COP_SCALE_M_PER_N, TRUE_RGBA, FORCE_VIEW)
    draw_contacts(scene, contacts, COP_SCALE_M_PER_N, FORCE_VIEW)
    # Sensor origin, for reference.
    draw_spheres(scene, [np.zeros(3)], 0.0003, (1.0, 1.0, 1.0, 1.0))


def report(cop, cop_force, contacts) -> str:
    """Return one line comparing the true CoP with the strongest estimated one."""
    if not contacts:
        return "no contact found"
    contact = contacts[0]
    error_mm = 1000 * np.linalg.norm(contact.cop - cop)
    return (
        f"{len(contacts)} contact(s), CoP error {error_mm:.2f} mm, "
        f"true force {np.round(cop_force, 2)} N, "
        f"estimated {np.round(contact.cop_force, 2)} N"
    )


def main() -> None:
    """Open the viewer and redraw the sensor until it is closed."""
    sensor = synthetic_sensor(SHAPE, NUM_TAXELS)
    positions = sensor.positions
    normals = normals_from_rotations(sensor.rotations)
    detector = ContactDetector(sensor, SETTINGS)
    rng = np.random.default_rng(0)
    model = mujoco.MjModel.from_xml_string(MODEL)
    data = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = positions.mean(axis=0)
            viewer.cam.distance = 0.06
            viewer.cam.elevation = -35
        start = time.monotonic()
        last_report = -1.0
        while viewer.is_running():
            now = time.monotonic() - start
            time_s = now if MOVING else 0.0
            cop, normal = true_cop(positions, normals, time_s)
            cop_force = true_cop_force(normal, time_s)
            forces = taxel_forces(positions, normals, cop, normal, cop_force, rng)
            contacts = detector.update(forces, in_taxel_frames=False)
            with viewer.lock():
                draw_frame(viewer.user_scn, positions, forces, contacts, cop, cop_force)
            viewer.sync()
            if now - last_report >= 1.0:
                print(report(cop, cop_force, contacts))
                last_report = now
            time.sleep(1.0 / RATE_HZ)


if __name__ == "__main__":
    main()
