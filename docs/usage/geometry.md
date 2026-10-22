# Geometry

Sensor geometry utilities in `src/nb_touch/geometry/`, imported from `nb_touch.geometry`: rotations, frame changes, taxel normals and neighbours, and synthetic sensors.

## Estimating taxel orientations

- `taxel_normals(positions, neighbours=7)`: outward normal of every taxel. A plane is fitted to each taxel and its nearest neighbours (SVD), then the signs are made consistent from the highest taxel, which faces +z, along a minimum spanning tree of the neighbour graph (Hoppe et al., "Surface reconstruction from unorganized points", SIGGRAPH 1992). Needs at least 3 taxels.

## Finding neighbouring taxels

For each taxel: list the taxels that are closer than a given distance. This distance is computed based on a factor multiplying the median distance between all taxels. Using `factor=1.5` ensures a taxel in a square grid has 8 neighbours and not only 4.

Example: `taxel_neighbours(positions, factor=1.5)`

<!--- todo: improve example -->

## Rotation matrices and frame changes

- `project_to_so3(matrices)`: Transforms a 3x3 matrix into the closest rotation matrix (the determinant must be 1). Necessary because of number rounding issues. See [so3-projection.md](../theory/so3-projection.md).
- `rotate_vectors(rotations, vectors)`: rotates each vector by its own rotation matrix.
- `transform_points(rotation, translation, points)`: Rotates and translates an array of points, $p \mapsto R p + t$.

## Synthetic sensors

Mock sensors are implemented, used for tests and in the examples. Use the function `synthetic_sensor()` to create a `SensorGeometry` instance. We provide three basic sensor geometries:

- plane
- cylinder
- sphere

The number of taxels as well as the spacing can be chosen. Example:

<!--- todo -->
