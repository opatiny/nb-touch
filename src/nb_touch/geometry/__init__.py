"""Sensor geometry: rotations, frame changes, normals, neighbours, synthetic sensors."""

from nb_touch.geometry.frames import (
    normals_from_rotations,
    taxel_rotations,
    with_estimated_rotations,
)
from nb_touch.geometry.neighbours import taxel_neighbours
from nb_touch.geometry.normals import taxel_normals
from nb_touch.geometry.synthetic import cylinder_sensor, plane_sensor, sphere_sensor
from nb_touch.geometry.transforms import rotate_vectors, transform_points

__all__ = [
    "cylinder_sensor",
    "normals_from_rotations",
    "plane_sensor",
    "rotate_vectors",
    "sphere_sensor",
    "taxel_neighbours",
    "taxel_normals",
    "taxel_rotations",
    "transform_points",
    "with_estimated_rotations",
]
