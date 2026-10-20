"""Reading and writing YAML files."""

from pathlib import Path

import yaml


def read_yaml(path: str | Path) -> dict:
    """Read a YAML file holding a mapping.

    Args:
        path: The file.

    Returns:
        Its content, an empty dict for an empty file.

    Raises:
        FileNotFoundError: If the file is missing.
        ValueError: If the file holds something other than a mapping.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping, got {type(data).__name__}")
    return data


def write_yaml(data: dict, path: str | Path) -> Path:
    """Write a mapping to a YAML file, keeping the key order.

    The innermost lists are written in flow style, so a matrix takes one line per
    row. Taken from `dextHand`, `src/dexthand/touch/taxel_frames.py`.

    Args:
        data: Plain data (dicts, lists, numbers, strings).
        path: Destination file. Parent folders are created.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=200)
    path.write_text(text, encoding="utf-8")
    return path
