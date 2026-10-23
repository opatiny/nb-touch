# Development doc

## Activate .venv

If necessary: deactivate previously activated venv with `deactivate`.

Activate the local venv with `source .venv/bin/activate`.

## Running tests

In an activated venv terminal: `pytest`

Otherwise: `uv run pytest`

## Publishing to PyPI

The workflow [.github/workflows/publish.yml](../.github/workflows/publish.yml) builds the package and uploads it to PyPI each time a GitHub release is published, using trusted publishing (no token).

Setup, once:

- PyPI, [publishing settings](https://pypi.org/manage/account/publishing/): create a pending publisher with project `nb-touch`, owner `opatiny`, repository `nb-touch`, workflow `publish.yml`, environment `pypi`.
- GitHub, Settings → Environments: create an environment named `pypi`. Optional: restrict it to tags `v*`, add yourself as required reviewer (leave "Prevent self-review" unchecked).

Releasing:

1. Bump `version` in `pyproject.toml`, commit and push. PyPI refuses a version twice.
2. Create the release: `gh release create vX.Y.Z --target main --generate-notes`, or on GitHub: Releases → Draft a new release.
3. Follow the Publish workflow in the Actions tab.
