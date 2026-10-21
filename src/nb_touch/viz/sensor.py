"""Draw a sensor's taxels, taxel forces and contacts in a MuJoCo scene, such as a
passive viewer's `user_scn`. Positions and forces in the scene's world frame."""

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
    scene, positions, forces, scale: float, min_force: float = 0.0
) -> int:
    """Draw one arrow per taxel force, starting at the taxel.

    Args:
        scene: `MjvScene` to draw into, in place.
        positions: Taxel positions `(n, 3)`, in m.
        forces: Taxel forces `(n, 3)` in the same frame, in N.
        scale: Arrow length per newton, in m.
        min_force: Smaller forces get no arrow, in N.

    Returns:
        The number of arrows drawn.
    """
    arrows = force_arrows(positions, forces, scale, min_force)
    return draw_arrows(
        scene, arrows, TAXEL_FORCE_RGBA, thickness=0.0003, head_length=0.0008
    )


def draw_cop(scene, cop, cop_force, scale: float, rgba=COP_RGBA) -> int:
    """Draw a CoP as a sphere and its force as an arrow starting at it.

    Args:
        scene: `MjvScene` to draw into, in place.
        cop: CoP position `(3,)`, in m.
        cop_force: CoP force `(3,)`, in N.
        scale: Arrow length per newton, in m.
        rgba: Color.

    Returns:
        The number of geoms drawn.
    """
    before = scene.ngeom
    draw_spheres(scene, [cop], 0.0007, rgba)
    draw_arrows(scene, force_arrows(cop, cop_force, scale), rgba, 0.0008, 0.0025)
    return scene.ngeom - before


def draw_contacts(scene, contacts: list[Contact], scale: float) -> int:
    """Draw every contact's CoP and CoP force, see `draw_cop`.

    Returns:
        The number of geoms drawn.
    """
    return sum(
        draw_cop(scene, contact.cop, contact.cop_force, scale) for contact in contacts
    )
