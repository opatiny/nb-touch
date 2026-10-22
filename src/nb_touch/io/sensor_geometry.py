"""Reading and writing the sensor geometry YAML format.

See `docs/taxel-geometry.md`. Field checks adapted from `orca_core` (MIT,
ORCA Dexterity), `orca_core/hardware/sensing/taxel_geometry.py`, and rotation
rounding from `dextHand`, `src/dexthand/touch/taxel_frames.py`.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nb_touch.utils.units import length_to_metres_factor
from nb_touch.utils.yaml_io import read_yaml, write_yaml

SENSOR_FRAME = "sensor"
"""Frame of the positions and rotations in the file."""

DECIMALS = 6
"""Decimals kept for positions and rotation entries when writing."""


@dataclass(frozen=True)
class SensorGeometry:
    """Taxel positions and orientations of one sensor model, in the sensor frame.

    Attributes:
        name: Sensor model name.
        positions: `(n, 3)` taxel positions in metres, row `i` is taxel `i` of the
            force stream.
        rotations: `(n, 3, 3)` rotations from each taxel frame to the sensor frame,
            or `None` when unknown.
    """

    name: str
    positions: np.ndarray
    rotations: np.ndarray | None = None

    def __post_init__(self):
        positions = np.asarray(self.positions, dtype=float)
        check_positions(positions)
        object.__setattr__(self, "positions", positions)
        if self.rotations is not None:
            rotations = np.asarray(self.rotations, dtype=float)
            check_rotations(rotations, len(positions))
            object.__setattr__(self, "rotations", rotations)

    @property
    def num_taxels(self) -> int:
        """Number of taxels."""
        return len(self.positions)


def check_positions(positions: np.ndarray) -> None:
    """Raise a `ValueError` unless `positions` has shape `(n, 3)`."""
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"expected (n, 3) positions, got {positions.shape}")


def check_rotations(rotations: np.ndarray, num_taxels: int) -> None:
    """Raise a `ValueError` unless `rotations` has shape `(num_taxels, 3, 3)`."""
    if rotations.shape != (num_taxels, 3, 3):
        raise ValueError(
            f"expected ({num_taxels}, 3, 3) rotations, got {rotations.shape}"
        )


def require_field(data: dict, key: str):
    """Return `data[key]`, raising a `ValueError` naming the field when missing."""
    if key not in data:
        raise ValueError(f"missing field {key!r}")
    return data[key]


def check_frame(frame: str) -> None:
    """Raise a `ValueError` unless `frame` is the sensor frame."""
    if frame != SENSOR_FRAME:
        raise ValueError(f"unsupported frame {frame!r}, expected {SENSOR_FRAME!r}")


def check_num_taxels(declared: int, actual: int) -> None:
    """Raise a `ValueError` if the declared taxel count differs from the actual."""
    if declared != actual:
        raise ValueError(f"num_taxels={declared} disagrees with {actual} taxels")


def parse_positions(taxels: list[dict], units: str) -> np.ndarray:
    """Read the taxel positions and convert them to metres.

    Args:
        taxels: The `taxels` entries of the file.
        units: Units of the positions in the file.

    Returns:
        `(n, 3)` positions in metres.
    """
    factor = length_to_metres_factor(units)
    if not taxels:
        return np.empty((0, 3))
    positions = np.array(
        [require_field(taxel, "position") for taxel in taxels], dtype=float
    )
    check_positions(positions)
    return factor * positions


def parse_rotations(taxels: list[dict]) -> np.ndarray | None:
    """Read the taxel rotations, which all taxels have or none has.

    Args:
        taxels: The `taxels` entries of the file.

    Returns:
        `(n, 3, 3)` rotations, or `None` if no taxel has one.

    Raises:
        ValueError: If only some taxels have a rotation.
    """
    with_rotation = sum("rotation" in taxel for taxel in taxels)
    if with_rotation == 0:
        return None
    if with_rotation != len(taxels):
        raise ValueError(
            f"{with_rotation} of {len(taxels)} taxels have a rotation: "
            "give it for all taxels or none"
        )
    rotations = np.array([taxel["rotation"] for taxel in taxels], dtype=float)
    check_rotations(rotations, len(taxels))
    return rotations


def geometry_from_dict(data: dict) -> SensorGeometry:
    """Build a `SensorGeometry` from the content of a sensor geometry file.

    Raises:
        ValueError: If a field is missing or invalid.
    """
    check_frame(require_field(data, "frame"))
    taxels = require_field(data, "taxels")
    check_num_taxels(require_field(data, "num_taxels"), len(taxels))
    return SensorGeometry(
        name=require_field(data, "name"),
        positions=parse_positions(taxels, require_field(data, "units")),
        rotations=parse_rotations(taxels),
    )


def geometry_to_dict(
    geometry: SensorGeometry, units: str = "mm", decimals: int = DECIMALS
) -> dict:
    """Lay out a `SensorGeometry` as plain data for the YAML file.

    Args:
        geometry: The geometry to write.
        units: Units for the positions in the file.
        decimals: Decimals kept for positions and rotation entries.

    Returns:
        A dict of lists, numbers and strings.
    """
    positions = np.round(geometry.positions / length_to_metres_factor(units), decimals)
    taxels = [{"position": position} for position in positions.tolist()]
    if geometry.rotations is not None:
        rotations = np.round(geometry.rotations, decimals).tolist()
        for taxel, rotation in zip(taxels, rotations, strict=True):
            taxel["rotation"] = rotation
    return {
        "name": geometry.name,
        "frame": SENSOR_FRAME,
        "units": units,
        "num_taxels": geometry.num_taxels,
        "taxels": taxels,
    }


def load_sensor_geometry(path: str | Path) -> SensorGeometry:
    """Read a sensor geometry YAML file.

    Args:
        path: The file.

    Returns:
        The geometry, positions in metres.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If a field is missing or invalid.
    """
    return geometry_from_dict(read_yaml(path))


def save_sensor_geometry(
    geometry: SensorGeometry,
    path: str | Path,
    units: str = "mm",
    decimals: int = DECIMALS,
) -> Path:
    """Write a sensor geometry YAML file.

    Args:
        geometry: The geometry to write.
        path: Destination file. Parent folders are created.
        units: Units for the positions in the file, `mm` or `m`.
        decimals: Decimals kept for positions and rotation entries.

    Returns:
        The path written to.
    """
    return write_yaml(geometry_to_dict(geometry, units, decimals), path)
