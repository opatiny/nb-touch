"""Reading and writing recordings: arrays in `.npz`, metadata in a YAML sidecar.

See `docs/theory/taxel-orientation.md`.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nb_touch.utils.yaml_io import read_yaml, write_yaml

ARRAY_NAMES = ("timestamps_s", "joint_pos_rad", "joint_torque_nm", "taxel_forces_n")
"""Names of the arrays in the `.npz` file."""


@dataclass(frozen=True)
class Recording:
    """Data recorded for the taxel orientation calibration.

    Attributes:
        timestamps_s: `(N,)` sample times in s.
        joint_pos_rad: `(N, J)` joint positions in rad.
        joint_torque_nm: `(N, J)` joint torques in N·m.
        taxel_forces_n: `(N, T, 3)` taxel forces in N, in each taxel's frame.
        metadata: Sensor model, joint names, sensor link, sampling rate, date.
    """

    timestamps_s: np.ndarray
    joint_pos_rad: np.ndarray
    joint_torque_nm: np.ndarray
    taxel_forces_n: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        for name in ARRAY_NAMES:
            object.__setattr__(self, name, np.asarray(getattr(self, name), dtype=float))
        check_recording_shapes(
            self.timestamps_s,
            self.joint_pos_rad,
            self.joint_torque_nm,
            self.taxel_forces_n,
        )
        check_joint_names(self.metadata, self.num_joints)

    @property
    def num_samples(self) -> int:
        """Number of samples N."""
        return len(self.timestamps_s)

    @property
    def num_joints(self) -> int:
        """Number of joints J."""
        return self.joint_pos_rad.shape[1]

    @property
    def num_taxels(self) -> int:
        """Number of taxels T."""
        return self.taxel_forces_n.shape[1]


def check_recording_shapes(
    timestamps_s: np.ndarray,
    joint_pos_rad: np.ndarray,
    joint_torque_nm: np.ndarray,
    taxel_forces_n: np.ndarray,
) -> None:
    """Raise a `ValueError` unless the arrays have shapes `(N,)`, `(N, J)`, `(N, J)`
    and `(N, T, 3)`.
    """
    if timestamps_s.ndim != 1:
        raise ValueError(f"expected (N,) timestamps_s, got {timestamps_s.shape}")
    num_samples = len(timestamps_s)
    if joint_pos_rad.ndim != 2 or len(joint_pos_rad) != num_samples:
        raise ValueError(
            f"expected ({num_samples}, J) joint_pos_rad, got {joint_pos_rad.shape}"
        )
    if joint_torque_nm.shape != joint_pos_rad.shape:
        raise ValueError(
            f"expected {joint_pos_rad.shape} joint_torque_nm, "
            f"got {joint_torque_nm.shape}"
        )
    if (
        taxel_forces_n.ndim != 3
        or len(taxel_forces_n) != num_samples
        or taxel_forces_n.shape[2] != 3
    ):
        raise ValueError(
            f"expected ({num_samples}, T, 3) taxel_forces_n, "
            f"got {taxel_forces_n.shape}"
        )


def check_joint_names(metadata: dict, num_joints: int) -> None:
    """Raise a `ValueError` if `metadata` has `joint_names` not matching J."""
    names = metadata.get("joint_names")
    if names is not None and len(names) != num_joints:
        raise ValueError(f"{len(names)} joint_names for {num_joints} joints")


def recording_paths(path: str | Path) -> tuple[Path, Path]:
    """Return the `.npz` and `.yaml` paths of a recording.

    Args:
        path: Either file of the recording, or their common path without suffix.
    """
    path = Path(path)
    if path.suffix in (".npz", ".yaml"):
        path = path.with_suffix("")
    return path.with_name(path.name + ".npz"), path.with_name(path.name + ".yaml")


def save_recording(recording: Recording, path: str | Path) -> tuple[Path, Path]:
    """Write a recording to a `.npz` file and its YAML sidecar.

    Args:
        recording: The recording to write.
        path: Either file of the recording, or their common path without suffix.
            Parent folders are created.

    Returns:
        The `.npz` and `.yaml` paths written to.
    """
    npz_path, yaml_path = recording_paths(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: getattr(recording, name) for name in ARRAY_NAMES}
    np.savez_compressed(npz_path, **arrays)
    write_yaml(recording.metadata, yaml_path)
    return npz_path, yaml_path


def load_recording(path: str | Path) -> Recording:
    """Read a recording from its `.npz` file and YAML sidecar.

    Args:
        path: Either file of the recording, or their common path without suffix.

    Raises:
        FileNotFoundError: If either file is missing.
        ValueError: If an array is missing or the shapes disagree.
    """
    npz_path, yaml_path = recording_paths(path)
    metadata = read_yaml(yaml_path)
    with np.load(npz_path) as data:
        missing = [name for name in ARRAY_NAMES if name not in data]
        if missing:
            raise ValueError(f"{npz_path}: missing arrays {missing}")
        arrays = {name: data[name] for name in ARRAY_NAMES}
    return Recording(**arrays, metadata=metadata)
