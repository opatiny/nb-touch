# Taxel orientation calibration recording format

Format for the data recorded for the [taxel orientation calibration](../theory/taxel-orientation.md): one `.npz` file for the arrays, plus a small YAML file for the metadata.

## Arrays (`.npz`)

| Array             | Shape     | Unit                     |
| ----------------- | --------- | ------------------------ |
| `timestamps_s`    | (N,)      | s                        |
| `joint_pos_rad`   | (N, J)    | rad                      |
| `joint_torque_nm` | (N, J)    | N·m                      |
| `taxel_forces_n`  | (N, T, 3) | N, in each taxel's frame |

N samples, J joints, T taxels.

## Metadata (`.yaml`)

Sensor model, joint names, link the sensor is mounted on, sampling rate, date.

## Why npz?

npz is numpy's own format: a zip of binary arrays, written with `np.savez_compressed` and read with `np.load` ([numpy docs](https://numpy.org/doc/stable/reference/generated/numpy.savez_compressed.html)).

|                                          | npz                | JSON                            | CSV                                            |
| ---------------------------------------- | ------------------ | ------------------------------- | ---------------------------------------------- |
| Size of the paper's 2400 × 87 × 3 forces | about 5 MB or less | about 11 MB                     | about 11 MB                                    |
| Shapes and dtypes                        | kept               | lost (nested lists)             | lost, 2D only: forces flattened to 261 columns |
| Precision, NaN                           | exact              | rounded, no NaN in the standard | rounded unless written at full precision       |
| Human-readable                           | no                 | yes                             | yes, also in Excel                             |
| Files per recording                      | one                | one                             | one per array, rows to keep aligned            |

Sizes computed by us: 626,400 float64 values, and about 18 characters per number in JSON or CSV.

YAML or JSON remain the choice for small files, such as [sensor geometry](sensor-geometry.md) and metadata.

## Reading and writing a recording

Implemented in `src/nb_touch/io/recordings.py`, imported from `nb_touch.io`.

- `Recording`: the four arrays above plus `metadata`, a dict. Shapes are checked on creation.
- `save_recording(recording, path)`: writes `<path>.npz` and `<path>.yaml` and returns both paths.
- `load_recording(path)`: reads both files. Raises a `FileNotFoundError` if one is missing, and a `ValueError` if an array is missing or shapes disagree.
