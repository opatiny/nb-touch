"""Draw point markers in a MuJoCo scene, such as a passive viewer's `user_scn`.

Like the force arrows of `nb_touch.viz.arrows`, they are redrawn every frame and
change nothing in the model. Taken from `dextHand`, `src/dexthand/sim/markers.py`.
"""

import mujoco
import numpy as np


def draw_spheres(scene, centres, radius: float, rgba) -> int:
    """Append one sphere per point to a ``MjvScene``.

    Spheres that do not fit in the scene's ``maxgeom`` are dropped.

    Args:
        scene: Scene to draw into, in place.
        centres: Sphere centres ``(n, 3)``, in world coordinates.
        radius: Sphere radius, in metres.
        rgba: Sphere color.

    Returns:
        The number of spheres drawn.
    """
    color = np.asarray(rgba, dtype=np.float32)
    size = np.array([radius, 0.0, 0.0])
    drawn = 0
    for centre in np.asarray(centres, dtype=float).reshape(-1, 3):
        if scene.ngeom >= scene.maxgeom:
            break
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_SPHERE,
            size,
            centre,
            np.eye(3).flatten(),
            color,
        )
        scene.ngeom += 1
        drawn += 1
    return drawn
