"""Slow tests with the real models and the six example photos from Wikimedia Commons.

Run before each release on a machine with internet:

    pip install -e ".[all,dev]"
    pytest -m slow

They download about 3 GB of weights on first run. A GPU is recommended but not required.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from simcitynet import SimCityNet

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        any(importlib.util.find_spec(m) is None for m in ("torch", "transformers", "ultralytics")),
        reason='needs pip install "simcitynet[all]"',
    ),
]

DATA = Path(__file__).parent / "data"
REFERENCE = DATA / "reference_embeddings.json"


@pytest.fixture(scope="module")
def net() -> SimCityNet:
    return SimCityNet()


def _example_images() -> list[Path]:
    """The six Wikimedia Commons examples, downloaded once into tests/data (git-ignored)."""
    from simcitynet.datasets import fetch_examples

    return fetch_examples(DATA)


@pytest.fixture(scope="module")
def images() -> list[Path]:
    return _example_images()


def test_real_pipeline_produces_plausible_values(net, images):
    emb = net.embed(images)
    raw = emb.raw
    assert raw.shape == (len(images), 33)
    assert not raw.isna().any().any()
    assert raw["vegetation"].between(0, 1).all()
    assert raw[["road_share", "sidewalk_share", "building_share"]].sum(axis=1).le(1 + 1e-9).all()
    assert (raw[["person", "car"]] >= 0).all().all()


def test_matches_reference_embeddings(net, images):
    """Regression guard: library upgrades must not silently change results.

    Create the reference once with ``python -m tests.test_real_models`` and commit it.
    """
    if not REFERENCE.exists():
        pytest.skip("No reference file yet; run `python -m tests.test_real_models` to create it")
    expected = json.loads(REFERENCE.read_text())
    emb = net.embed(images)
    for name, values in expected.items():
        got = emb.raw.loc[name].to_numpy(dtype=float)
        np.testing.assert_allclose(got, np.array(values), rtol=0.05, atol=0.02, err_msg=name)


if __name__ == "__main__":  # pragma: no cover - helper to (re)create the reference file
    paths = _example_images()
    result = SimCityNet().embed(paths)
    REFERENCE.write_text(
        json.dumps({n: result.raw.loc[n].tolist() for n in result.names}, indent=1)
    )
    print(f"Wrote {REFERENCE} for {len(paths)} images")
