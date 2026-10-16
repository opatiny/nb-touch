# Recording format

Format for the data recorded for the [taxel orientation calibration](../theory/taxel-orientation.md): one `.npz` file for the arrays, plus a small YAML file for the metadata.

## Arrays (`.npz`)

| Array | Shape | Unit |
|---|---|---|
| `timestamps_s` | (N,) | s |
| `joint_pos_rad` | (N, J) | rad |
| `joint_torque_nm` | (N, J) | N·m |
| `taxel_forces_n` | (N, T, 3) | N, in each taxel's frame |

N samples, J joints, T taxels.

## Metadata (`.yaml`)

Sensor model, joint names, link the sensor is mounted on, sampling rate, date.

## Why npz

npz is numpy's own format: a zip of binary arrays, written with `np.savez_compressed` and read with `np.load` ([numpy docs](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html)).

| | npz | JSON |
|---|---|---|
| Size of the paper's 2400 × 87 × 3 forces | about 5 MB or less | about 11 MB |
| Shapes and dtypes | kept | lost (nested lists) |
| Precision, NaN | exact | rounded, no NaN in the standard |
| Human-readable | no | yes |

Sizes computed by us: 626,400 float64 values, and about 18 characters per number in JSON.

YAML or JSON remain the choice for small files, such as [sensor geometry](sensor-geometry.md) and metadata.
