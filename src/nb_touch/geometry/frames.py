"""Local taxel frames, with z along each taxel's outward normal.

`rotation_from_z` and `taxel_orientations` taken from `dextHand`,
`src/dexthand/touch/taxels.py`.
"""

import numpy as np

from nb_touch.geometry.normals import NEIGHBOURS, taxel_normals
from nb_touch.io.sensor_geometry import SensorGeometry


def rotation_from_z(normal) -> np.ndarray:
    """Return the smallest rotation that turns the z axis onto `normal`.

    Its columns are the local frame's axes in the parent frame: the local z is
    `normal`, the local x and y are the parent's x and y tilted as little as possible
    (Rodrigues' formula).

    Args:
        normal: Unit vector `(3,)`, in the parent frame.

    Returns:
        Rotation `(3, 3)` taking local vectors to the parent frame.
    """
    normal = np.asarray(normal, dtype=float)
    cosine = float(normal[2])
    if cosine < -1.0 + 1e-9:
        # Pointing straight down: any half turn about a horizontal axis works.
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross([0.0, 0.0, 1.0], normal)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + skew + skew @ skew / (1.0 + cosine)


def orientations_from_normals(normals) -> np.ndarray:
    """Return `rotation_from_z` of every normal `(n, 3)`, as `(n, 3, 3)`."""
    return np.array([rotation_from_z(normal) for normal in normals]).reshape(-1, 3, 3)


def normals_from_orientations(orientations) -> np.ndarray:
    """Return the local z axis of every orientation `(n, 3, 3)`, as `(n, 3)`."""
    return np.asarray(orientations, dtype=float)[..., :, 2]


def taxel_orientations(positions, neighbours: int = NEIGHBOURS) -> np.ndarray:
    """Build the local frame of every taxel, z along its estimated normal.

    Args:
        positions: Taxel positions `(n, 3)`, in the sensor frame.
        neighbours: Points per plane fit, see `taxel_normals`.

    Returns:
        Orientations `(n, 3, 3)` taking each taxel's local vectors to the sensor frame.
    """
    return orientations_from_normals(taxel_normals(positions, neighbours))


def with_estimated_orientations(
    geometry: SensorGeometry, neighbours: int = NEIGHBOURS
) -> SensorGeometry:
    """Return the geometry, with orientations estimated from the positions if missing.

    Args:
        geometry: Sensor geometry, with or without orientations.
        neighbours: Points per plane fit, see `taxel_normals`.
    """
    if geometry.orientations is not None:
        return geometry
    orientations = taxel_orientations(geometry.positions, neighbours)
    return SensorGeometry(geometry.name, geometry.positions, orientations)
