# CI and publishing

Running the tests on GitHub, releasing automatically and publishing the package to PyPI.

## Tests on GitHub

A workflow in `.github/workflows/tests.yml` runs on every push and pull request:

- `actions/checkout`, then `astral-sh/setup-uv`.
- `uv sync`, then `uv run pytest`.
- Optionally on several Python versions (matrix).

Source: [uv GitHub Actions guide](https://docs.astral.sh/uv/guides/integration/github/).

## Publishing to PyPI

- `uv build`: builds a wheel and a source archive in `dist/`.
- `uv publish`: uploads them to PyPI. Try [TestPyPI](https://test.pypi.org/) first.
- Trusted publishing: link the GitHub repo to the project on pypi.org, and a GitHub Action publishes without any token.
- Check on pypi.org that the name `nb-touch` is free.

Sources: [uv packaging guide](https://docs.astral.sh/uv/guides/package/), [PyPI trusted publishers](https://docs.pypi.org/trusted-publishers/).

## Automatic releases

[release-please](https://github.com/googleapis/release-please-action) reads conventional commits (`feat:`, `fix:`):

1. It opens a release PR that bumps the version and updates the changelog.
2. Merging the PR tags the release.
3. A publish job then builds and uploads to PyPI with `pypa/gh-action-pypi-publish`.
