"""Synthetic sensors with known normals, for tests and demos: taxels on a plane, a
cylinder and a sphere. Positions in metres, sensor frame with z outward."""

import numpy as np

from nb_touch.geometry.frames import rotations_from_normals
from nb_touch.io.sensor_geometry import SensorGeometry


def grid(rows: int, cols: int, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the x and y coordinates `(rows * cols,)` of a grid centred on 0."""
    x = (np.arange(cols) - (cols - 1) / 2) * spacing
    y = (np.arange(rows) - (rows - 1) / 2) * spacing
    xx, yy = np.meshgrid(x, y)
    return xx.ravel(), yy.ravel()


def plane_sensor(
    rows: int = 5, cols: int = 5, spacing: float = 0.003
) -> SensorGeometry:
    """Taxels on a grid in the plane z = 0, normals along +z.

    Args:
        rows: Taxels along y.
        cols: Taxels along x.
        spacing: Distance between neighbours, in m.
    """
    x, y = grid(rows, cols, spacing)
    positions = np.stack([x, y, np.zeros_like(x)], axis=1)
    normals = np.tile([0.0, 0.0, 1.0], (len(positions), 1))
    return SensorGeometry("plane", positions, rotations_from_normals(normals))


def cylinder_sensor(
    radius: float = 0.008,
    rows: int = 6,
    cols: int = 7,
    spacing: float = 0.003,
    arc: float = np.pi / 2,
) -> SensorGeometry:
    """Taxels on an arc of a cylinder of axis y through the origin, centred on +z.

    Args:
        radius: Cylinder radius, in m.
        rows: Taxels along the axis.
        cols: Taxels around the axis, spread over `arc`.
        spacing: Distance between rows, in m.
        arc: Angle covered around the axis, in rad.
    """
    angles = np.linspace(-arc / 2, arc / 2, cols) if cols > 1 else np.zeros(1)
    angle, y = np.meshgrid(angles, (np.arange(rows) - (rows - 1) / 2) * spacing)
    angle, y = angle.ravel(), y.ravel()
    normals = np.stack([np.sin(angle), np.zeros_like(angle), np.cos(angle)], axis=1)
    positions = radius * normals + np.stack([np.zeros_like(y), y, np.zeros_like(y)], 1)
    return SensorGeometry("cylinder", positions, rotations_from_normals(normals))


def sphere_sensor(
    radius: float = 0.01, count: int = 40, cap: float = np.pi / 3
) -> SensorGeometry:
    """Taxels spread evenly on a spherical cap around +z, sphere centred on 0.

    Points of a Fibonacci spiral, equal area per point (González, "Measurement of
    areas on a sphere using Fibonacci and latitude–longitude lattices", Math. Geosci.
    42, 2010).

    Args:
        radius: Sphere radius, in m.
        count: Number of taxels.
        cap: Largest angle from +z, in rad.
    """
    index = np.arange(count) + 0.5
    cos_polar = 1.0 - (1.0 - np.cos(cap)) * index / count
    sin_polar = np.sqrt(1.0 - cos_polar**2)
    azimuth = np.pi * (1.0 + 5**0.5) * index
    normals = np.stack(
        [sin_polar * np.cos(azimuth), sin_polar * np.sin(azimuth), cos_polar], axis=1
    )
    return SensorGeometry("sphere", radius * normals, rotations_from_normals(normals))
