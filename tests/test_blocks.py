"""Unit tests for the five measurement blocks, on hand-made inputs with known answers."""

from __future__ import annotations

import numpy as np
import pytest

from simcitynet.backends import Detection, find_label_ids
from simcitynet.blocks import (
    compute_vegetation,
    count_objects,
    depth_weighted_index,
    dominant_colors,
    normalize_depth,
    surface_shares,
)
from simcitynet.config import Config

from .conftest import ADE_LABELS, CITY_LABELS, TREE, FakeADESegmenter, FakeCityscapesSegmenter

CFG = Config()


# --------------------------------------------------------------------------------- colour
def test_solid_image_gives_one_colour_with_full_share():
    img = np.full((30, 40, 3), (255, 0, 0), dtype=np.uint8)
    res = dominant_colors(img)
    assert res.centroids.shape == (5, 3)
    np.testing.assert_allclose(res.centroids[0], [255, 0, 0])
    np.testing.assert_allclose(res.shares, [1, 0, 0, 0, 0])
    assert res.vector().shape == (20,)


def test_two_colour_split_is_sorted_by_share():
    img = np.zeros((10, 100, 3), dtype=np.uint8)
    img[:, :70] = (10, 200, 10)
    img[:, 70:] = (0, 0, 250)
    res = dominant_colors(img)
    np.testing.assert_allclose(res.shares[:2], [0.7, 0.3])
    np.testing.assert_allclose(res.centroids[0], [10, 200, 10])
    np.testing.assert_allclose(res.centroids[1], [0, 0, 250])
    assert res.hex()[:2] == ["#0ac80a", "#0000fa"]


def test_kmeans_path_is_deterministic_and_shares_sum_to_one():
    img = np.random.default_rng(1).integers(0, 256, (60, 60, 3), dtype=np.uint8)
    a = dominant_colors(img, random_state=3)
    b = dominant_colors(img, random_state=3)
    np.testing.assert_allclose(a.vector(), b.vector())
    assert a.shares.sum() == pytest.approx(1.0)
    assert np.all(np.diff(a.shares) <= 1e-12)
    assert np.all((a.centroids >= 0) & (a.centroids <= 255))


def test_colour_sampling_limits_pixels():
    img = np.random.default_rng(2).integers(0, 256, (200, 200, 3), dtype=np.uint8)
    res = dominant_colors(img, sample_pixels=500)
    assert res.shares.sum() == pytest.approx(1.0)


def test_colour_errors():
    with pytest.raises(ValueError):
        dominant_colors(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        dominant_colors(np.zeros((4, 4, 3), dtype=np.uint8), k=0)


# ----------------------------------------------------------------------------- vegetation
def test_normalize_depth_minmax_and_constant():
    np.testing.assert_allclose(
        normalize_depth(np.array([[2.0, 4.0], [6.0, 10.0]])), [[0, 0.25], [0.5, 1]]
    )
    np.testing.assert_allclose(normalize_depth(np.full((2, 2), 3.0)), np.ones((2, 2)))


def test_depth_weighted_index_basic_cases():
    mask = np.zeros((4, 4), dtype=bool)
    mask[:, :2] = True
    assert depth_weighted_index(mask, np.ones((4, 4))) == pytest.approx(0.5)
    assert depth_weighted_index(np.ones((4, 4), bool), np.ones((4, 4))) == pytest.approx(1.0)
    weights = np.zeros((4, 4))
    weights[:, :2] = 1.0  # only the masked half has weight
    assert depth_weighted_index(mask, weights) == pytest.approx(1.0)
    assert depth_weighted_index(mask, np.zeros((4, 4))) == pytest.approx(0.5)  # fallback


def test_vegetation_uniform_depth_equals_pixel_fraction(scene):
    seg = FakeADESegmenter().segment(scene)
    res = compute_vegetation(
        scene, seg, ADE_LABELS, None, CFG.vegetation_hsv, CFG.vegetation_fallback_ids
    )
    expected = float(np.all(scene == np.array(TREE, np.uint8), axis=-1).mean())
    assert res.value == pytest.approx(expected)
    assert res.per_class["tree"] == pytest.approx(expected)
    assert res.per_class["grass"] == 0.0
    assert res.mask.dtype == bool


def test_vegetation_depth_weighting_changes_value(scene):
    seg = FakeADESegmenter().segment(scene)
    h, w = scene.shape[:2]
    near_top = np.repeat(np.linspace(10, 0, h)[:, None], w, axis=1)  # top is close
    flat = compute_vegetation(
        scene, seg, ADE_LABELS, None, CFG.vegetation_hsv, CFG.vegetation_fallback_ids
    )
    weighted = compute_vegetation(
        scene, seg, ADE_LABELS, near_top, CFG.vegetation_hsv, CFG.vegetation_fallback_ids
    )
    assert weighted.value > flat.value  # trees sit in the upper half


def test_hsv_refinement_removes_non_green_pixels():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[:, :5] = TREE
    img[:, 5:] = (200, 60, 60)  # red, but labelled "tree" below
    seg = np.full((10, 10), 4)
    res = compute_vegetation(
        img, seg, ADE_LABELS, None, CFG.vegetation_hsv, CFG.vegetation_fallback_ids
    )
    assert res.value == pytest.approx(0.5)


def test_vegetation_shape_checks(scene):
    seg = FakeADESegmenter().segment(scene)
    with pytest.raises(ValueError):
        compute_vegetation(
            scene, seg[:-1], ADE_LABELS, None, CFG.vegetation_hsv, CFG.vegetation_fallback_ids
        )
    with pytest.raises(ValueError):
        compute_vegetation(
            scene, seg, ADE_LABELS, np.ones((3, 3)), CFG.vegetation_hsv, CFG.vegetation_fallback_ids
        )


def test_vegetation_uses_fallback_ids_without_names():
    img = np.full((4, 4, 3), TREE, dtype=np.uint8)
    res = compute_vegetation(
        img, np.full((4, 4), 4), {}, None, CFG.vegetation_hsv, CFG.vegetation_fallback_ids
    )
    assert res.value == pytest.approx(1.0)
    none = compute_vegetation(img, np.full((4, 4), 4), {}, None, CFG.vegetation_hsv, {})
    assert none.value == 0.0


# ------------------------------------------------------------------------------- labels
def test_find_label_ids_by_name_partial_and_fallback():
    labels = {3: "Tree", 7: "grass;lawn", 12: "flower"}
    ids = find_label_ids(labels, ["tree", "grass", "plant"], {"plant": 17})
    assert ids == {"tree": 3, "grass": 7, "plant": 17}


# ------------------------------------------------------------------------------ dynamics
def test_count_objects_threshold_aliases_and_ignored_classes():
    dets = [
        Detection("person", 0.9, (0, 0, 1, 1)),
        Detection("Person", 0.25, (0, 0, 1, 1)),
        Detection("person", 0.24, (0, 0, 1, 1)),
        Detection("motorbike", 0.5, (0, 0, 1, 1)),
        Detection("aeroplane", 0.5, (0, 0, 1, 1)),
        Detection("dog", 0.99, (0, 0, 1, 1)),
    ]
    res = count_objects(dets, 0.25)
    assert res.counts["person"] == 2
    assert res.counts["motorcycle"] == 1
    assert res.counts["airplane"] == 1
    assert sum(res.counts.values()) == 4
    assert len(res.detections) == 4


def test_count_objects_rejects_bad_threshold():
    with pytest.raises(ValueError):
        count_objects([], 1.5)


# ------------------------------------------------------------------------------ surfaces
def test_surface_shares(scene):
    seg = FakeCityscapesSegmenter().segment(scene)
    shares = surface_shares(seg, CITY_LABELS, CFG.surface_fallback_ids)
    h = scene.shape[0]
    assert shares["road"] == pytest.approx((h - 2 * h // 3) / h)
    assert shares["sidewalk"] == pytest.approx((2 * h // 3 - h // 2) / h)
    assert shares["building"] == pytest.approx(
        ((h // 2 - h // 6) * (scene.shape[1] // 2)) / scene[..., 0].size
    )
    assert sum(shares.values()) <= 1.0


def test_surface_shares_errors_and_missing_class():
    with pytest.raises(ValueError):
        surface_shares(np.zeros((0, 0), int), CITY_LABELS, {})
    shares = surface_shares(np.zeros((2, 2), int), {0: "road"}, {})
    assert shares == {"road": 1.0, "sidewalk": 0.0, "building": 0.0}
