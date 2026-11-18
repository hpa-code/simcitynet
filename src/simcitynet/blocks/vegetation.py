"""Vegetation block ``f_veg``: depth-weighted Green View Index (paper Section 4.1).

Steps: semantic masks for tree / grass / plant -> HSV refinement against green-painted
false positives -> weighting by min-max normalised monocular depth (larger = closer) ->

    GVI_depth = Σ M(i,j) D(i,j) / Σ D(i,j)

over the union of the refined masks.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from ..backends.base import RGBImage, find_label_ids
from ..config import HSVRange
from ..features import VEGETATION_CLASSES


@dataclass
class VegetationResult:
    """Output of :func:`compute_vegetation`."""

    #: Combined depth-weighted GVI over all vegetation classes (the embedding value).
    value: float
    #: Depth-weighted GVI per class (diagnostic; not part of the 33 dimensions).
    per_class: dict[str, float]
    #: Union of the refined vegetation masks, ``(H, W)`` bool.
    mask: NDArray[np.bool_]
    #: Depth weights in [0, 1] used for the index, ``(H, W)`` float.
    weights: NDArray[np.float32]


def normalize_depth(depth: NDArray[np.floating]) -> NDArray[np.float32]:
    """Min-max normalise a depth map to [0, 1]; a constant map becomes all ones."""
    d = np.nan_to_num(np.asarray(depth, dtype=np.float32))
    lo, hi = float(d.min()), float(d.max())
    if hi - lo <= 1e-12:
        return np.ones_like(d, dtype=np.float32)
    return ((d - lo) / (hi - lo)).astype(np.float32)


def hsv_mask(image: RGBImage, hsv_range: HSVRange) -> NDArray[np.bool_]:
    """Pixels whose HSV value lies within ``hsv_range`` (OpenCV units)."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lo, hi = (np.array(v, dtype=np.uint8) for v in hsv_range)
    return cv2.inRange(hsv, lo, hi).astype(bool)


def depth_weighted_index(mask: NDArray[np.bool_], weights: NDArray[np.floating]) -> float:
    """``Σ mask·w / Σ w``; falls back to the plain pixel fraction if all weights are zero."""
    total = float(weights.sum())
    if total <= 0:
        return float(mask.mean()) if mask.size else 0.0
    return float((mask * weights).sum() / total)


def compute_vegetation(
    image: RGBImage,
    segmentation: NDArray[np.integer],
    labels: Mapping[int, str],
    depth: NDArray[np.floating] | None,
    hsv_ranges: Mapping[str, HSVRange],
    fallback_ids: Mapping[str, int],
) -> VegetationResult:
    """Compute the vegetation block from a segmentation map and (optionally) a depth map.

    If ``depth`` is ``None`` every pixel gets weight 1, which gives the classic Green View Index.
    """
    if segmentation.shape != image.shape[:2]:
        raise ValueError(
            f"Segmentation shape {segmentation.shape} does not match image {image.shape[:2]}"
        )
    ids = find_label_ids(labels, VEGETATION_CLASSES, fallback_ids)
    weights = (
        np.ones(image.shape[:2], dtype=np.float32) if depth is None else normalize_depth(depth)
    )
    if weights.shape != image.shape[:2]:
        raise ValueError(f"Depth shape {weights.shape} does not match image {image.shape[:2]}")

    union = np.zeros(image.shape[:2], dtype=bool)
    per_class: dict[str, float] = {}
    for name in VEGETATION_CLASSES:
        if name not in ids:
            per_class[name] = 0.0
            continue
        refined = segmentation == ids[name]
        if name in hsv_ranges:
            refined &= hsv_mask(image, hsv_ranges[name])
        per_class[name] = depth_weighted_index(refined, weights)
        union |= refined
    return VegetationResult(
        value=depth_weighted_index(union, weights),
        per_class=per_class,
        mask=union,
        weights=weights,
    )
