# Geometry

Sensor geometry utilities in `src/nb_touch/geometry/`, imported from `nb_touch.geometry`: rotations, frame changes, taxel normals and neighbours, and synthetic sensors.

## Normals and taxel frames

- `taxel_normals(positions, neighbours=7)`: outward normal of every taxel. A plane is fitted to each taxel and its nearest neighbours (SVD), then the signs are made consistent from the highest taxel, which faces +z, along a minimum spanning tree of the neighbour graph (Hoppe et al., "Surface reconstruction from unorganized points", SIGGRAPH 1992). Needs at least 3 taxels.
- `taxel_rotations(positions, neighbours=7)`: one rotation per taxel, local z along its normal, the smallest rotation from the sensor's z.
- `with_estimated_rotations(geometry, neighbours=7)`: the `SensorGeometry`, with rotations estimated when missing.
- `normals_from_rotations(rotations)`: the local z axis of each rotation.

| Parameter    | What it does                                     | Default  |
| ------------ | ------------------------------------------------ | -------- |
| `positions`  | Taxel positions (n, 3), sensor frame, z outward. | required |
| `neighbours` | Taxels per plane fit, the taxel included.        | `7`      |

Taken from `dextHand`, `src/dexthand/touch/taxels.py`.

## Neighbours

`taxel_neighbours(positions, factor=1.5)`: per taxel, the taxels closer than `factor` times the median nearest-neighbour spacing, so diagonal neighbours on a square grid are linked too. Compute it once per sensor.

| Parameter   | What it does                                | Default  |
| ----------- | ------------------------------------------- | -------- |
| `positions` | Taxel positions (n, 3).                     | required |
| `factor`    | Link distance, in multiples of the spacing. | `1.5`    |

## Rotations and frame changes

- `project_to_so3(matrices)`: closest rotation to each 3×3 matrix, see [so3-projection.md](../theory/so3-projection.md).
- `rotate_vectors(rotations, vectors)`: rotates each vector by its own rotation, for example taxel forces from their taxel frames to the sensor frame.
- `transform_points(rotation, translation, points)`: $p \mapsto R p + t$.

## Synthetic sensors

Sensors with known normals, for tests and demos. Each returns a `SensorGeometry` with positions in metres and the true rotations.

- `synthetic_sensor(shape, num_taxels, spacing=0.003, radius=0.01)`: exactly `num_taxels` taxels on a `plane`, `cylinder` (near-square grid, last row cut) or `sphere` (60° cap).
- `plane_sensor(rows=5, cols=5, spacing=0.003)`: grid in the plane z = 0, centred on 0.
- `cylinder_sensor(radius=0.008, rows=6, cols=7, spacing=0.003, arc=π/2)`: arc of a cylinder of axis y through 0, centred on +z.
- `sphere_sensor(radius=0.01, count=40, cap=π/3)`: Fibonacci spiral on a spherical cap around +z (González, Math. Geosci. 42, 2010).

| Parameter | What it does                                    | Default         |
| --------- | ----------------------------------------------- | --------------- |
| `rows`    | Taxels along y (plane) or the axis (cylinder).  | `5`, `6`        |
| `cols`    | Taxels along x (plane) or around the axis.      | `5`, `7`        |
| `spacing` | Distance between rows, in m.                    | `0.003`         |
| `radius`  | Cylinder or sphere radius, in m.                | `0.008`, `0.01` |
| `arc`     | Angle covered around the cylinder axis, in rad. | π/2             |
| `count`   | Number of taxels on the sphere.                 | `40`            |
| `cap`     | Largest angle from +z on the sphere, in rad.    | π/3             |
