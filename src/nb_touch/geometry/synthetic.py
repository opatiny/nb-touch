"""Synthetic sensors with known normals, for tests and demos: taxels on a plane, a
cylinder and a sphere. Positions in metres, sensor frame with z outward."""

import numpy as np

from nb_touch.geometry.frames import rotations_from_normals
from nb_touch.io.sensor_geometry import SensorGeometry

SHAPES = ("plane", "cylinder", "sphere")
"""Shapes of `synthetic_sensor`."""


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


def near_square_grid(num_taxels: int) -> tuple[int, int]:
    """Return `(rows, cols)`, close to square, with at least `num_taxels` cells."""
    rows = max(1, round(num_taxels**0.5))
    return rows, -(-num_taxels // rows)


def first_taxels(geometry: SensorGeometry, num_taxels: int) -> SensorGeometry:
    """Return the geometry with only its first `num_taxels` taxels."""
    return SensorGeometry(
        geometry.name,
        geometry.positions[:num_taxels],
        geometry.rotations[:num_taxels],
    )


def synthetic_sensor(
    shape: str, num_taxels: int, spacing: float = 0.003, radius: float = 0.01
) -> SensorGeometry:
    """Return a synthetic sensor with exactly `num_taxels` taxels.

    Plane and cylinder: a near-square grid, its last row cut to the count. Sphere:
    a cap of 60° around +z.

    Args:
        shape: One of `SHAPES`.
        num_taxels: Number of taxels, at least 1.
        spacing: Distance between grid neighbours, in m (plane and cylinder).
        radius: Cylinder or sphere radius, in m.

    Raises:
        ValueError: If the shape is unknown or `num_taxels` below 1.
    """
    if num_taxels < 1:
        raise ValueError(f"need at least 1 taxel, got {num_taxels}")
    rows, cols = near_square_grid(num_taxels)
    if shape == "plane":
        return first_taxels(plane_sensor(rows, cols, spacing), num_taxels)
    if shape == "cylinder":
        arc = (cols - 1) * spacing / radius
        sensor = cylinder_sensor(radius, rows, cols, spacing, arc)
        return first_taxels(sensor, num_taxels)
    if shape == "sphere":
        return sphere_sensor(radius, num_taxels)
    raise ValueError(f"unknown shape {shape!r}, expected one of {SHAPES}")
