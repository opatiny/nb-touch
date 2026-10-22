"""Draw a sensor's taxels, taxel forces and contacts in a MuJoCo scene, such as a
passive viewer's `user_scn`. Positions and forces in the scene's world frame."""

from enum import StrEnum

import numpy as np

from nb_touch.contact.contacts import Contact
from nb_touch.viz.arrows import draw_arrows, force_arrows
from nb_touch.viz.markers import draw_spheres

TAXEL_RGBA = (0.6, 0.6, 0.6, 1.0)
"""Grey: taxels."""

ACTIVE_RGBA = (1.0, 0.8, 0.0, 1.0)
"""Yellow: taxels in a contact."""

TAXEL_FORCE_RGBA = (0.2, 0.5, 1.0, 1.0)
"""Blue: taxel forces."""

COP_RGBA = (1.0, 0.2, 0.2, 1.0)
"""Red: CoPs and CoP forces estimated from the taxel forces."""


class ForceView(StrEnum):
    """Which force the arrows show. Either way, the arrow starts at its point."""

    FORCE_ON_SENSOR = "force_on_sensor"
    """The force acting on the sensor: a push is drawn inside the surface."""

    SENSOR_REACTION_FORCE = "sensor_reaction_force"
    """Its opposite, the force the sensor applies back: a push is drawn outside."""


def sensor_force_arrows(
    points,
    forces,
    scale: float,
    min_force: float = 0.0,
    view: ForceView = ForceView.FORCE_ON_SENSOR,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Turn forces acting on the sensor into arrow segments, see `ForceView`.

    Args:
        points: Application points `(3,)` or `(n, 3)`, in m.
        forces: Forces acting on the sensor, same shape, in N.
        scale: Arrow length per newton, in m.
        min_force: Smaller forces get no arrow, in N.
        view: Which force to draw.

    Returns:
        One `(start, end)` pair per arrow drawn.
    """
    if view == ForceView.SENSOR_REACTION_FORCE:
        return force_arrows(points, -np.asarray(forces, dtype=float), scale, min_force)
    return force_arrows(points, forces, scale, min_force)


def draw_taxels(scene, positions, active=None, radius: float = 0.0004) -> int:
    """Draw every taxel as a sphere, the active ones in another color.

    Args:
        scene: `MjvScene` to draw into, in place.
        positions: Taxel positions `(n, 3)`, in m.
        active: Mask `(n,)` of the taxels to highlight, or None.
        radius: Sphere radius, in m.

    Returns:
        The number of spheres drawn.
    """
    positions = np.asarray(positions, dtype=float)
    if active is None:
        active = np.zeros(len(positions), dtype=bool)
    active = np.asarray(active, dtype=bool)
    drawn = draw_spheres(scene, positions[~active], radius, TAXEL_RGBA)
    return drawn + draw_spheres(scene, positions[active], radius, ACTIVE_RGBA)


def draw_taxel_forces(
    scene,
    positions,
    forces,
    scale: float,
    min_force: float = 0.0,
    view: ForceView = ForceView.FORCE_ON_SENSOR,
) -> int:
    """Draw one arrow per taxel force, see `ForceView`.

    Args:
        scene: `MjvScene` to draw into, in place.
        positions: Taxel positions `(n, 3)`, in m.
        forces: Taxel forces `(n, 3)` in the same frame, in N.
        scale: Arrow length per newton, in m.
        min_force: Smaller forces get no arrow, in N.
        view: Which force to draw.

    Returns:
        The number of arrows drawn.
    """
    arrows = sensor_force_arrows(positions, forces, scale, min_force, view)
    return draw_arrows(
        scene, arrows, TAXEL_FORCE_RGBA, thickness=0.0003, head_length=0.002
    )


def draw_cop(
    scene,
    cop,
    cop_force,
    scale: float,
    rgba=COP_RGBA,
    view: ForceView = ForceView.FORCE_ON_SENSOR,
) -> int:
    """Draw a CoP as a sphere and its force as an arrow, see `ForceView`.

    Args:
        scene: `MjvScene` to draw into, in place.
        cop: CoP position `(3,)`, in m.
        cop_force: CoP force `(3,)`, in N.
        scale: Arrow length per newton, in m.
        rgba: Color.
        view: Which force to draw.

    Returns:
        The number of geoms drawn.
    """
    before = scene.ngeom
    draw_spheres(scene, [cop], 0.0007, rgba)
    arrows = sensor_force_arrows(cop, cop_force, scale, view=view)
    draw_arrows(scene, arrows, rgba, 0.0008, 0.01)
    return scene.ngeom - before


def draw_contacts(
    scene,
    contacts: list[Contact],
    scale: float,
    view: ForceView = ForceView.FORCE_ON_SENSOR,
) -> int:
    """Draw every contact's CoP and CoP force, see `draw_cop`.

    Returns:
        The number of geoms drawn.
    """
    return sum(
        draw_cop(scene, contact.cop, contact.cop_force, scale, view=view)
        for contact in contacts
    )
