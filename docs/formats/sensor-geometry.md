# Sensor geometry format

Proposed YAML format for the taxel positions and orientations of one sensor model, in the sensor frame.

## Fields

| Field | What it is |
|---|---|
| `name` | Sensor model name. |
| `frame` | Always `sensor`. |
| `units` | `mm` or `m`, for positions. |
| `num_taxels` | Number of taxels, checked against `taxels`. |
| `taxels` | One entry per taxel, in the order of the force stream. |
| `taxels[i].position` | `[x, y, z]` in `units`. |
| `taxels[i].rotation` | Optional 3×3 matrix from the taxel frame to the sensor frame, local z along the outward normal. Estimated from the geometry when missing. |

Sensor frame: z points outward through the sensing surface, right-handed.

## Sources

It merges two existing formats:

- `orca_core`, `orca_core/hardware/sensing/models/touch-sensor-finger/config.yaml`: `name`, `frame`, `units`, `num_taxels` and positions, read by `orca_core/hardware/sensing/taxel_geometry.py`. Sensor frame convention from the same file.
- `dextHand`, `data/software/taxel_frames.yaml`: one rotation per taxel, local z along the normal, written by `src/dexthand/touch/taxel_frames.py`.
