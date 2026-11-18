"""Normalisation, similarity and colour EMD."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from simcitynet.features import BLOCKS, EMBEDDING_DIM, FEATURE_NAMES, FEATURES
from simcitynet.normalize import complete_columns, fixed_bounds, normalize
from simcitynet.similarity import (
    color_emd,
    color_emd_matrix,
    cosine_similarity_matrix,
    euclidean_distance_matrix,
)


def test_feature_names_match_paper_layout():
    assert EMBEDDING_DIM == 33
    assert len(set(FEATURE_NAMES)) == 33
    assert FEATURE_NAMES[0] == "vegetation"
    assert FEATURE_NAMES[1:21] == BLOCKS["color"]
    assert FEATURE_NAMES[21] == "person"
    assert FEATURE_NAMES[22:30] == (
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
    )
    assert FEATURE_NAMES[30:] == ("road_share", "sidewalk_share", "building_share")
    assert sum(len(v) for v in BLOCKS.values()) == 33
    assert [f.name for f in FEATURES] == list(FEATURE_NAMES)
    assert {f.block for f in FEATURES} == set(BLOCKS)


def _frame(rows: list[list[float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=list(FEATURE_NAMES[: len(rows[0])]))


def test_batch_normalisation_divides_by_column_max_and_keeps_zero_columns():
    df = _frame([[0.5, 100.0, 0.0], [0.25, 200.0, 0.0]])
    out = normalize(df, "batch")
    np.testing.assert_allclose(out.to_numpy(), [[1.0, 0.5, 0.0], [0.5, 1.0, 0.0]])


def test_fixed_normalisation_uses_bounds_and_clips():
    raw = pd.DataFrame([np.zeros(33)], columns=list(FEATURE_NAMES))
    raw.loc[0, "color1_r"] = 255
    raw.loc[0, "person"] = 100  # above the cap of 50 -> clipped
    raw.loc[0, "car"] = 15
    out = normalize(raw, "fixed")
    assert out.loc[0, "color1_r"] == 1.0
    assert out.loc[0, "person"] == 1.0
    assert out.loc[0, "car"] == pytest.approx(0.5)
    bounds = fixed_bounds({"person": 20})
    assert bounds["person"] == 20 and bounds["bus"] == 10 and bounds["road_share"] == 1


def test_normalise_none_and_invalid():
    df = _frame([[1.0, 2.0]])
    pd.testing.assert_frame_equal(normalize(df, "none"), df.astype(float))
    with pytest.raises(ValueError):
        normalize(df, "zscore")  # type: ignore[arg-type]


def test_complete_columns_drops_missing():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [np.nan, 1.0]})
    assert list(complete_columns(df).columns) == ["a"]


def test_cosine_similarity_properties():
    x = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 3.0], [0.0, 0.0]])
    s = cosine_similarity_matrix(x)
    np.testing.assert_allclose(s, s.T)
    np.testing.assert_allclose(np.diag(s), 1.0)
    assert s[0, 1] == pytest.approx(1.0)
    assert s[0, 2] == pytest.approx(0.0)
    assert s[3, 0] == 0.0  # zero vector
    with pytest.raises(ValueError):
        cosine_similarity_matrix(np.ones(3))


def test_euclidean_distance_properties():
    x = np.array([[0.0, 0.0], [3.0, 4.0]])
    d = euclidean_distance_matrix(x)
    np.testing.assert_allclose(d, [[0, 5], [5, 0]])
    with pytest.raises(ValueError):
        euclidean_distance_matrix(np.ones(3))


def test_color_emd_identity_symmetry_and_known_value():
    a_c = np.array([[0, 0, 0], [255, 255, 255]])
    a_w = np.array([0.5, 0.5])
    assert color_emd(a_c, a_w, a_c, a_w) == pytest.approx(0.0, abs=1e-9)
    b_c = np.array([[0, 0, 0], [0, 0, 0]])
    b_w = np.array([0.5, 0.5])
    d = color_emd(a_c, a_w, b_c, b_w)
    assert d == pytest.approx(0.5 * np.sqrt(3) * 255)
    assert color_emd(b_c, b_w, a_c, a_w) == pytest.approx(d)
    m = color_emd_matrix([(a_c, a_w), (b_c, b_w)])
    np.testing.assert_allclose(m, [[0, d], [d, 0]])


def test_color_emd_validation():
    with pytest.raises(ValueError):
        color_emd([[0, 0, 0]], [1, 0], [[0, 0, 0]], [1])
    with pytest.raises(ValueError):
        color_emd([[0, 0, 0]], [0], [[0, 0, 0]], [1])
