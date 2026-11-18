# Contributing to simcitynet

Thanks for helping! Bug reports, questions, documentation fixes and new features are all welcome.

## Reporting a problem

Open an [issue](https://github.com/hpa-code/simcitynet/issues) with:

- what you ran (a minimal code snippet or command),
- what you expected and what happened (full traceback),
- the output of `simcitynet doctor`.

## Development setup

```bash
git clone https://github.com/hpa-code/simcitynet.git
cd simcitynet
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ",all" to run the real-model tests
```

## Checks (the same ones CI runs)

```bash
ruff format --check src tests     # formatting (run `ruff format` to fix)
ruff check src tests              # linting
mypy                              # types
pytest --cov                      # fast tests with fake models; coverage must stay >= 80%
mkdocs build --strict             # documentation
```

The fast test suite never downloads models: the feature blocks are tested on small synthetic
images with known answers, and the pipeline with fake models (`tests/conftest.py`).

Before a release, run the real-model tests on a machine with internet (GPU recommended):

```bash
pip install -e ".[all,dev]"
python -m tests.test_real_models   # downloads the example photos and (re)creates
                                   # tests/data/reference_embeddings.json - commit that file
pytest -m slow
```

## Pull requests

1. Create a branch from `main`.
2. Add tests for new behaviour and update the docs and `CHANGELOG.md` (under *Unreleased*).
3. Keep changes that alter embedding values clearly marked: they change research results.
   Explain them in the pull request and in `docs/faithfulness.md`.
4. Make sure all checks pass.

## Releasing (maintainers)

See [PUBLISHING.md](https://github.com/hpa-code/simcitynet/blob/main/PUBLISHING.md). In short: bump `src/simcitynet/_version.py`,
`CITATION.cff` and `CHANGELOG.md` together, then push a tag `vX.Y.Z`.

By contributing you agree that your contributions are licensed under the MIT License and
that you will follow the [Code of Conduct](https://github.com/hpa-code/simcitynet/blob/main/CODE_OF_CONDUCT.md).
