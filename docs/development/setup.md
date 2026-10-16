# Development setup

Creating the virtual environment and running the tests, with `uv`.

- `uv sync`: creates `.venv/` in the repo and installs the package and its dependencies.
- `uv run pytest`: runs the tests inside `.venv/`, without activating it.
- Tests live in `__tests__/` folders next to the code they test.

Source: [uv docs](https://docs.astral.sh/uv/).
