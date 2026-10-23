"""Draw force arrows in a MuJoCo scene, such as a passive viewer's `user_scn`.

They are redrawn every frame and change nothing in the model. Taken from `dextHand`,
`src/dexthand/sim/arrows.py`.
"""

import mujoco
import numpy as np

ARROW_SIZE_PER_LENGTH = 2.0
"""`size[2]` an `mjGEOM_ARROW` needs per metre of drawn length. MuJoCo 3.16 draws an
arrow geom half as long as the `size[2]` `mjv_connector` gives it: measured by us,
against a cylinder between the same points."""


def force_arrows(
    starts, forces, scale: float, min_force: float = 0.0, tip_at_point: bool = False
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Turn forces applied at points into arrow segments.

    Args:
        starts: Application points, ``(3,)`` or ``(n, 3)``, in metres.
        forces: Force vectors in the same axes, same shape, in newtons.
        scale: Arrow length per newton, in metres.
        min_force: Forces with a smaller magnitude get no arrow, which keeps
            sensor noise off the screen.
        tip_at_point: True to end each arrow at its point instead of starting it
            there, so a push on a surface is drawn outside it.

    Returns:
        One ``(start, end)`` pair per arrow drawn.
    """
    starts = np.atleast_2d(np.asarray(starts, dtype=float))
    forces = np.atleast_2d(np.asarray(forces, dtype=float))
    keep = np.linalg.norm(forces, axis=1) >= min_force
    arrows = [
        (start, start + force * scale)
        for start, force in zip(starts[keep], forces[keep])
    ]
    if tip_at_point:
        return [(2 * start - end, start) for start, end in arrows]
    return arrows


def add_connector(scene, geom_type, start, end, thickness: float, rgba) -> bool:
    """Append one connector geom, such as a cylinder or an arrow, between two points.

    Args:
        scene: ``MjvScene`` to draw into, in place.
        geom_type: A connector ``mjtGeom`` type.
        start: First point, in world coordinates.
        end: Second point, in world coordinates.
        thickness: Geom diameter, in metres.
        rgba: Color, as a float32 array.

    Returns:
        False if the scene is full and nothing was added.
    """
    if scene.ngeom >= scene.maxgeom:
        return False
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom, geom_type, np.zeros(3), np.zeros(3), np.eye(3).flatten(), rgba
    )
    # mjv_connector takes a radius.
    mujoco.mjv_connector(
        geom,
        geom_type,
        thickness / 2,
        np.asarray(start, dtype=float),
        np.asarray(end, dtype=float),
    )
    if geom_type == mujoco.mjtGeom.mjGEOM_ARROW:
        geom.size[2] *= ARROW_SIZE_PER_LENGTH
    scene.ngeom += 1
    return True


def split_arrow(
    start, end, head_length: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Split an arrow into a shaft and a head of fixed length.

    Args:
        start: Arrow start.
        end: Arrow tip.
        head_length: Length of the head, in metres. An arrow shorter than that is all
            head.

    Returns:
        ``(start, neck, end)``, where the head runs from ``neck`` to ``end``, or None
        for a zero-length arrow.
    """
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    length = float(np.linalg.norm(end - start))
    if length == 0.0:
        return None
    neck = end - (end - start) / length * min(head_length, length)
    return start, neck, end


def draw_arrows(
    scene, arrows, rgba, thickness: float = 0.002, head_length: float = 0.003
) -> int:
    """Append arrows to a ``MjvScene``, such as a passive viewer's ``user_scn``.

    MuJoCo's arrow geom scales its head with its length, so each arrow is drawn as a
    cylinder for the shaft plus a short arrow geom for the head, which keeps the head
    the same size whatever the force. Arrows that do not fit in the scene's ``maxgeom``
    are dropped.

    Args:
        scene: Scene to draw into, in place.
        arrows: ``(start, end)`` pairs in world coordinates, as from
            :func:`force_arrows`.
        rgba: Arrow color.
        thickness: Arrow shaft diameter, in metres.
        head_length: Length of the arrow head, in metres, the same for every arrow.

    Returns:
        The number of arrows drawn. Zero-length arrows are skipped.
    """
    color = np.asarray(rgba, dtype=np.float32)
    drawn = 0
    for arrow_start, arrow_end in arrows:
        parts = split_arrow(arrow_start, arrow_end, head_length)
        if parts is None:
            continue
        start, neck, end = parts
        has_shaft = not np.allclose(start, neck)
        if scene.ngeom + 1 + has_shaft > scene.maxgeom:
            break
        if has_shaft:
            add_connector(
                scene, mujoco.mjtGeom.mjGEOM_CYLINDER, start, neck, thickness, color
            )
        add_connector(scene, mujoco.mjtGeom.mjGEOM_ARROW, neck, end, thickness, color)
        drawn += 1
    return drawn
