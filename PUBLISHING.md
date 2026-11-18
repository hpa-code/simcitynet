# Publishing simcitynet

This guide takes the repository from GitHub to GitHub Pages, TestPyPI and PyPI. The
repository is `hpa-code/simcitynet`. If it ever moves to another owner, replace the old
owner everywhere:

```bash
grep -rl "hpa-code/simcitynet" --exclude-dir=.git . | xargs sed -i "s#hpa-code/simcitynet#OWNER/simcitynet#g"
grep -rl "hpa-code.github.io" --exclude-dir=.git . | xargs sed -i "s#hpa-code.github.io#OWNER.github.io#g"
```

(on macOS use `sed -i ''`).

## 0. Before the first release (one time, about 30 minutes)

These need internet and, ideally, a GPU — Google Colab is enough.

1. **Run the real-model tests and create the reference embeddings.**

    ```bash
    pip install -e ".[all,dev]"
    python -m tests.test_real_models     # downloads models + example photos, writes tests/data/reference_embeddings.json
    pytest -m slow                       # must pass
    ```

    Commit `tests/data/reference_embeddings.json` (the photos themselves are git-ignored).

2. **Run the four tutorials and save their outputs**, so the website shows the figures.
   Open each notebook in `docs/tutorials/` in Colab (Runtime → GPU), run all cells, then
   *File → Download .ipynb* and replace the file in `docs/tutorials/`. Until the package is on
   PyPI, the first cell installs from GitHub automatically.

3. **Pick a README hero image (optional).** Save a scene card from tutorial 1
   (`fig.savefig("docs/assets/scene_card.png", dpi=150)`) and add it under the pipeline
   diagram in `README.md` and `docs/index.md`. Credit the photo (see `CREDITS.csv`).

## 1. The GitHub repository

The code lives at <https://github.com/hpa-code/simcitynet> (public). To work on it locally:

```bash
git clone https://github.com/hpa-code/simcitynet.git
cd simcitynet
pip install -e ".[dev]"
```

On the repository page, click the gear next to **About** and set:

- Website: `https://hpa-code.github.io/simcitynet/`
- Topics: `python`, `computer-vision`, `urban-analytics`, `street-view`, `image-embedding`,
  `generative-ai`, `algorithmic-fairness`, `computational-social-science`

## 2. Turn on GitHub Pages

**Settings → Pages → Build and deployment → Source: GitHub Actions.** The *Docs* workflow
publishes the site on every push to `main` (re-run it from the **Actions** tab the first time).

## 3. Check CI

Open the **Actions** tab. *CI* runs formatting, linting, type checks, tests on Python
3.10–3.13 (Linux) and 3.12 (macOS, Windows), builds the package and the docs. Fix anything red
before releasing. *repo-review* is informational.

## 4. Connect PyPI and TestPyPI (Trusted Publishing — no passwords or tokens)

1. Create accounts on <https://pypi.org> and <https://test.pypi.org> (with two-factor auth).
2. On **PyPI → Your account → Publishing → Add a new pending publisher**, enter:
    - PyPI project name: `simcitynet`
    - Owner: `hpa-code` · Repository: `simcitynet`
    - Workflow name: `release.yml` · Environment name: `pypi`
3. Do the same on **TestPyPI** with environment name `testpypi`.
4. On GitHub, **Settings → Environments → New environment**: create `pypi` and `testpypi`.
   For `pypi`, tick *Required reviewers* and add yourself, so every release waits for your click.

## 5. Archive releases on Zenodo (DOI)

1. Sign in to <https://zenodo.org> with GitHub, open **Settings → GitHub**, and switch on
   `hpa-code/simcitynet`.
2. Every GitHub Release now gets a DOI. After the first one, add the DOI badge to `README.md`
   and an `identifiers:` entry to `CITATION.cff`.

## 6. Release

1. Make sure these agree on the version: `src/simcitynet/_version.py`, `CITATION.cff`
   (`version` and `date-released`) and `CHANGELOG.md` (move *Unreleased* items under the new
   version). The release workflow refuses to run if they differ.
2. Tag and push:

    ```bash
    git tag v0.1.0
    git push origin v0.1.0
    ```

3. The *Release* workflow builds the package, publishes to TestPyPI, waits for your approval
   on the `pypi` environment, publishes to PyPI, then creates the GitHub Release (which
   triggers Zenodo).
4. Check: `pip install simcitynet==0.1.0` in a fresh environment, the PyPI page, the website,
   and the Zenodo record.

## Later releases

Bump the three version places, update the changelog, push, tag `vX.Y.Z`. Use a patch version
for fixes, a minor version for new features, and say clearly in the changelog when a change
alters embedding values.
