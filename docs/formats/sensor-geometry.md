# Sensor geometry format

Proposed YAML format for the taxel positions and orientations of one sensor model, in the sensor frame.

## Fields

| Field                | What it is                                                                                                                                |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `name`               | Sensor model name.                                                                                                                        |
| `frame`              | Always `sensor`.                                                                                                                          |
| `units`              | `mm` or `m`, for positions.                                                                                                               |
| `num_taxels`         | Number of taxels, checked against `taxels`.                                                                                               |
| `taxels`             | One entry per taxel, in the order of the force stream.                                                                                    |
| `taxels[i].position` | `[x, y, z]` in `units`.                                                                                                                   |
| `taxels[i].rotation` | Optional 3×3 matrix from the taxel frame to the sensor frame, local z along the outward normal. Estimated from the geometry when missing. |

Sensor frame: z points outward through the sensing surface, right-handed.
Inspired by the Orca hand `orca_core` file format.

## Usage

Implemented in `src/nb_touch/io/sensor_geometry.py`, imported from `nb_touch.io`.

- `SensorGeometry`: `name`, `positions` (n, 3) in metres, `rotations` (n, 3, 3) or `None`.
- `load_sensor_geometry(path)`: reads a file, converting positions to metres. Raises a `ValueError` if a field is missing or invalid, or if only some taxels have a rotation.
- `save_sensor_geometry(geometry, path, units="mm", decimals=6)`: writes a file and returns its path.

| Parameter  | What it does                                      | Default  |
| ---------- | ------------------------------------------------- | -------- |
| `geometry` | Geometry to write.                                | required |
| `path`     | Destination file, parent folders created.         | required |
| `units`    | Units of the positions in the file, `mm` or `m`.  | `"mm"`   |
| `decimals` | Decimals kept for positions and rotation entries. | `6`      |

Field checks adapted from `orca_core` (MIT), `orca_core/hardware/sensing/taxel_geometry.py`; rounding and matrix layout from `dextHand`, `src/dexthand/touch/taxel_frames.py`.
