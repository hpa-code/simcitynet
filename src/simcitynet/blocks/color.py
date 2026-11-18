"""Colour block ``f_color``: K-means dominant colours and their pixel shares (paper Section 4.2).

``f_color = [μ1, …, μ5, P1, …, P5] ∈ R^20`` with RGB centroids ``μk ∈ [0, 255]^3`` and shares
``Pk`` summing to 1. K-means returns clusters in arbitrary order, so they are **sorted by share
(largest first, ties by brightness)** to make dimensions comparable across images.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning

from ..backends.base import RGBImage


@dataclass
class ColorResult:
    """Dominant colours of an image."""

    #: ``(k, 3)`` RGB centroids in [0, 255], sorted by share.
    centroids: NDArray[np.float64]
    #: ``(k,)`` pixel shares summing to 1, descending.
    shares: NDArray[np.float64]

    def vector(self) -> NDArray[np.float64]:
        """The ``3k + k`` values in embedding order: all centroids, then all shares."""
        return np.concatenate([self.centroids.reshape(-1), self.shares])

    def hex(self) -> list[str]:
        """Centroids as ``#rrggbb`` strings."""
        return ["#{:02x}{:02x}{:02x}".format(*(round(c) for c in rgb)) for rgb in self.centroids]


def _sort(centroids: NDArray[np.float64], shares: NDArray[np.float64]) -> ColorResult:
    luminance = centroids @ np.array([0.2126, 0.7152, 0.0722])
    order = np.lexsort((-luminance, -shares))  # primary: share desc, secondary: brightness desc
    return ColorResult(centroids[order], shares[order])


def dominant_colors(
    image: RGBImage, k: int = 5, sample_pixels: int = 50_000, random_state: int = 0
) -> ColorResult:
    """Cluster the pixels of ``image`` into ``k`` colours.

    A random sample of at most ``sample_pixels`` pixels is clustered (deterministic for a
    given ``random_state``). Images with fewer than ``k`` distinct colours are padded with
    zero-share copies of the most common colour, so the vector length is always ``4k``.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    pixels: NDArray[np.uint8] = np.asarray(image, dtype=np.uint8).reshape(-1, 3)
    if pixels.shape[0] == 0:
        raise ValueError("Cannot compute colours of an empty image")
    rng = np.random.default_rng(random_state)
    if pixels.shape[0] > sample_pixels:
        pixels = pixels[rng.choice(pixels.shape[0], size=sample_pixels, replace=False)]

    unique, counts = np.unique(pixels, axis=0, return_counts=True)
    if unique.shape[0] <= k:
        centroids = unique.astype(np.float64)
        shares = counts / counts.sum()
        if unique.shape[0] < k:
            pad = k - unique.shape[0]
            top = centroids[np.argmax(shares)]
            centroids = np.vstack([centroids, np.repeat(top[None, :], pad, axis=0)])
            shares = np.concatenate([shares, np.zeros(pad)])
        return _sort(centroids, shares.astype(np.float64))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        km = KMeans(n_clusters=k, n_init=4, random_state=random_state)
        assignment = km.fit_predict(pixels.astype(np.float64))
    shares = np.bincount(assignment, minlength=k) / assignment.size
    centroids = np.clip(km.cluster_centers_, 0, 255)
    return _sort(centroids.astype(np.float64), shares.astype(np.float64))
