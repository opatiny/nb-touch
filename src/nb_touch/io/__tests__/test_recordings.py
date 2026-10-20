import datetime
from pathlib import Path

import numpy as np
import pytest

from nb_touch.io.recordings import (
    Recording,
    load_recording,
    recording_paths,
    save_recording,
)

NUM_SAMPLES, NUM_JOINTS, NUM_TAXELS = 5, 4, 7


def make_recording(num_samples=NUM_SAMPLES, metadata=None):
    """Return a recording of random data with the given number of samples."""
    rng = np.random.default_rng(0)
    return Recording(
        timestamps_s=np.arange(num_samples) / 20.0,
        joint_pos_rad=rng.normal(size=(num_samples, NUM_JOINTS)),
        joint_torque_nm=rng.normal(size=(num_samples, NUM_JOINTS)),
        taxel_forces_n=rng.normal(size=(num_samples, NUM_TAXELS, 3)),
        metadata=metadata or {},
    )


METADATA = {
    "sensor_model": "test-sensor",
    "joint_names": ["j0", "j1", "j2", "j3"],
    "sensor_link": "fingertip",
    "sampling_rate_hz": 20.0,
    "date": datetime.date(2026, 10, 20),
}


def test_save_load_round_trip(tmp_path):
    recording = make_recording(metadata=METADATA)
    save_recording(recording, tmp_path / "rec")
    loaded = load_recording(tmp_path / "rec")
    for name in ("timestamps_s", "joint_pos_rad", "joint_torque_nm", "taxel_forces_n"):
        np.testing.assert_array_equal(getattr(loaded, name), getattr(recording, name))
    assert loaded.metadata == METADATA


def test_nan_kept_exactly(tmp_path):
    recording = make_recording()
    recording.taxel_forces_n[0, 0, 0] = np.nan
    save_recording(recording, tmp_path / "rec")
    assert np.isnan(load_recording(tmp_path / "rec").taxel_forces_n[0, 0, 0])


def test_zero_samples_round_trip(tmp_path):
    save_recording(make_recording(num_samples=0), tmp_path / "rec")
    loaded = load_recording(tmp_path / "rec")
    assert loaded.num_samples == 0
    assert loaded.taxel_forces_n.shape == (0, NUM_TAXELS, 3)


def test_counts():
    recording = make_recording()
    assert recording.num_samples == NUM_SAMPLES
    assert recording.num_joints == NUM_JOINTS
    assert recording.num_taxels == NUM_TAXELS


@pytest.mark.parametrize("name", ["rec", "rec.npz", "rec.yaml"])
def test_paths_from_any_file_or_stem(name):
    assert recording_paths(Path("d") / name) == (Path("d/rec.npz"), Path("d/rec.yaml"))


def test_paths_keep_dots_in_name():
    assert recording_paths("d/rec.v2") == (Path("d/rec.v2.npz"), Path("d/rec.v2.yaml"))


def test_save_creates_parent_folders(tmp_path):
    npz_path, yaml_path = save_recording(make_recording(), tmp_path / "a" / "b" / "rec")
    assert npz_path.is_file()
    assert yaml_path.is_file()


def test_load_from_npz_path(tmp_path):
    npz_path, _ = save_recording(make_recording(), tmp_path / "rec")
    assert load_recording(npz_path).num_samples == NUM_SAMPLES


def test_missing_yaml_raises(tmp_path):
    _, yaml_path = save_recording(make_recording(), tmp_path / "rec")
    yaml_path.unlink()
    with pytest.raises(FileNotFoundError):
        load_recording(tmp_path / "rec")


def test_missing_npz_raises(tmp_path):
    npz_path, _ = save_recording(make_recording(), tmp_path / "rec")
    npz_path.unlink()
    with pytest.raises(FileNotFoundError):
        load_recording(tmp_path / "rec")


def test_missing_array_raises(tmp_path):
    npz_path, _ = save_recording(make_recording(), tmp_path / "rec")
    np.savez_compressed(npz_path, timestamps_s=np.zeros(3))
    with pytest.raises(ValueError, match="missing arrays"):
        load_recording(tmp_path / "rec")


def recording_kwargs(**overrides):
    """Return valid `Recording` arguments, with some replaced."""
    kwargs = {
        "timestamps_s": np.zeros(3),
        "joint_pos_rad": np.zeros((3, 2)),
        "joint_torque_nm": np.zeros((3, 2)),
        "taxel_forces_n": np.zeros((3, 4, 3)),
    }
    return kwargs | overrides


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"timestamps_s": np.zeros((3, 1))}, "timestamps_s"),
        ({"joint_pos_rad": np.zeros(3)}, "joint_pos_rad"),
        ({"joint_pos_rad": np.zeros((2, 2))}, "joint_pos_rad"),
        ({"joint_torque_nm": np.zeros((3, 3))}, "joint_torque_nm"),
        ({"joint_torque_nm": np.zeros((2, 2))}, "joint_torque_nm"),
        ({"taxel_forces_n": np.zeros((3, 4))}, "taxel_forces_n"),
        ({"taxel_forces_n": np.zeros((2, 4, 3))}, "taxel_forces_n"),
        ({"taxel_forces_n": np.zeros((3, 4, 2))}, "taxel_forces_n"),
    ],
)
def test_wrong_shapes_raise(overrides, message):
    with pytest.raises(ValueError, match=message):
        Recording(**recording_kwargs(**overrides))


def test_joint_names_mismatch_raises():
    with pytest.raises(ValueError, match="3 joint_names for 2 joints"):
        Recording(**recording_kwargs(), metadata={"joint_names": ["a", "b", "c"]})


def test_arrays_converted_to_float():
    recording = Recording(**recording_kwargs(timestamps_s=[0, 1, 2]))
    assert recording.timestamps_s.dtype == float
