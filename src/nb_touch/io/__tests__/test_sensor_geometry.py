import numpy as np
import pytest

from nb_touch.io.sensor_geometry import (
    SensorGeometry,
    geometry_from_dict,
    geometry_to_dict,
    load_sensor_geometry,
    save_sensor_geometry,
)

POSITIONS_M = np.array([[0.0, 0.0, 0.0], [0.001, 0.002, 0.003], [-0.004, 0.0, 0.01]])


def rotations_about_z(angles):
    """Return one rotation about z per angle, shape `(n, 3, 3)`."""
    c, s = np.cos(angles), np.sin(angles)
    zeros, ones = np.zeros_like(angles), np.ones_like(angles)
    return np.stack(
        [
            np.stack([c, -s, zeros], axis=-1),
            np.stack([s, c, zeros], axis=-1),
            np.stack([zeros, zeros, ones], axis=-1),
        ],
        axis=-2,
    )


ORIENTATIONS = rotations_about_z(np.array([0.0, 0.5, -2.0]))


def valid_dict():
    return {
        "name": "test-sensor",
        "frame": "sensor",
        "units": "mm",
        "num_taxels": 2,
        "taxels": [{"position": [1.0, 2.0, 3.0]}, {"position": [4.0, 5.0, 6.0]}],
    }


@pytest.mark.parametrize("units", ["mm", "m"])
@pytest.mark.parametrize("orientations", [None, ORIENTATIONS])
def test_save_load_round_trip(tmp_path, units, orientations):
    geometry = SensorGeometry("test-sensor", POSITIONS_M, orientations)
    path = save_sensor_geometry(geometry, tmp_path / "geometry.yaml", units=units)
    loaded = load_sensor_geometry(path)
    assert loaded.name == "test-sensor"
    np.testing.assert_allclose(loaded.positions, POSITIONS_M, atol=1e-9)
    if orientations is None:
        assert loaded.orientations is None
    else:
        np.testing.assert_allclose(loaded.orientations, orientations, atol=1e-6)


def test_positions_converted_from_mm_to_metres():
    geometry = geometry_from_dict(valid_dict())
    np.testing.assert_allclose(
        geometry.positions, [[1e-3, 2e-3, 3e-3], [4e-3, 5e-3, 6e-3]]
    )


def test_positions_written_in_requested_units():
    geometry = SensorGeometry("s", [[0.001, 0.0, 0.0]])
    assert geometry_to_dict(geometry, "mm")["taxels"][0]["position"] == [1.0, 0.0, 0.0]
    assert geometry_to_dict(geometry, "m")["taxels"][0]["position"] == [0.001, 0.0, 0.0]


def test_written_values_are_rounded():
    geometry = SensorGeometry("s", [[0.0011234567891, 0.0, 0.0]])
    assert (
        geometry_to_dict(geometry, "mm", decimals=3)["taxels"][0]["position"][0]
        == 1.123
    )


def test_written_dict_has_documented_fields():
    data = geometry_to_dict(SensorGeometry("s", POSITIONS_M))
    assert list(data) == ["name", "frame", "units", "num_taxels", "taxels"]
    assert data["frame"] == "sensor"
    assert data["num_taxels"] == 3


def test_zero_taxels():
    data = valid_dict() | {"num_taxels": 0, "taxels": []}
    geometry = geometry_from_dict(data)
    assert geometry.num_taxels == 0
    assert geometry.positions.shape == (0, 3)
    assert geometry.orientations is None


@pytest.mark.parametrize("key", ["name", "frame", "units", "num_taxels", "taxels"])
def test_missing_field_raises(key):
    data = valid_dict()
    del data[key]
    with pytest.raises(ValueError, match=f"missing field '{key}'"):
        geometry_from_dict(data)


def test_missing_position_raises():
    data = valid_dict()
    data["taxels"][1] = {}
    with pytest.raises(ValueError, match="missing field 'position'"):
        geometry_from_dict(data)


def test_wrong_frame_raises():
    with pytest.raises(ValueError, match="unsupported frame"):
        geometry_from_dict(valid_dict() | {"frame": "world"})


def test_wrong_units_raise():
    with pytest.raises(ValueError, match="unsupported length units"):
        geometry_from_dict(valid_dict() | {"units": "cm"})


def test_num_taxels_mismatch_raises():
    with pytest.raises(ValueError, match="num_taxels=3 disagrees with 2"):
        geometry_from_dict(valid_dict() | {"num_taxels": 3})


@pytest.mark.parametrize("position", [[1.0, 2.0], [1.0, 2.0, 3.0, 4.0]])
def test_position_with_wrong_length_raises(position):
    data = valid_dict()
    data["taxels"][0]["position"] = position
    data["taxels"][1]["position"] = position
    with pytest.raises(ValueError, match=r"\(n, 3\) positions"):
        geometry_from_dict(data)


def test_positions_with_different_lengths_raise():
    data = valid_dict()
    data["taxels"][0]["position"] = [1.0, 2.0]
    with pytest.raises(ValueError):
        geometry_from_dict(data)


def test_orientation_on_some_taxels_only_raises():
    data = valid_dict()
    data["taxels"][0]["rotation"] = np.eye(3).tolist()
    with pytest.raises(ValueError, match="1 of 2 taxels have a rotation"):
        geometry_from_dict(data)


def test_orientation_with_wrong_shape_raises():
    data = valid_dict()
    for taxel in data["taxels"]:
        taxel["rotation"] = [[1.0, 0.0], [0.0, 1.0]]
    with pytest.raises(ValueError, match=r"\(2, 3, 3\) orientations"):
        geometry_from_dict(data)


def test_geometry_rejects_orientation_count_mismatch():
    with pytest.raises(ValueError, match=r"\(3, 3, 3\) orientations"):
        SensorGeometry("s", POSITIONS_M, ORIENTATIONS[:2])


def test_geometry_converts_lists_to_float_arrays():
    geometry = SensorGeometry("s", [[1, 2, 3]], [np.eye(3, dtype=int).tolist()])
    assert geometry.positions.dtype == float
    assert geometry.orientations.dtype == float


def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sensor_geometry(tmp_path / "missing.yaml")
