import pytest

from nb_touch.utils.yaml_io import read_yaml, write_yaml


def test_round_trip_keeps_content_and_order(tmp_path):
    data = {"b": 1, "a": [[1.0, 2.0], [3.0, 4.0]], "c": "text"}
    path = write_yaml(data, tmp_path / "file.yaml")
    loaded = read_yaml(path)
    assert loaded == data
    assert list(loaded) == ["b", "a", "c"]


def test_write_creates_parent_folders(tmp_path):
    path = write_yaml({"a": 1}, tmp_path / "x" / "y" / "file.yaml")
    assert path.is_file()


def test_matrix_rows_written_one_per_line(tmp_path):
    path = write_yaml({"m": [[1, 0], [0, 1]]}, tmp_path / "file.yaml")
    assert "- [1, 0]" in path.read_text()


def test_empty_file_reads_as_empty_dict(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("")
    assert read_yaml(path) == {}


def test_non_mapping_raises(tmp_path):
    path = tmp_path / "list.yaml"
    path.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="mapping"):
        read_yaml(path)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_yaml(tmp_path / "missing.yaml")
