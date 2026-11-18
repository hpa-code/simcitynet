"""Similarity and distance measures between embeddings (paper Sections 4 and 6.1).

* cosine similarity and equal-weight Euclidean distance on normalised vectors (Section 4);
* first-order Wasserstein / Earth Mover's distance between colour palettes (Section 6.1),
  with ground cost ``‖μk − νm‖₂`` in RGB space.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog


def cosine_similarity_matrix(x: ArrayLike) -> NDArray[np.float64]:
    """Pairwise cosine similarity of the rows of ``x``.

    A row of zeros has similarity 0 with every other row and 1 with itself.
    """
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("Expected a 2-D array")
    norms = np.linalg.norm(a, axis=1)
    safe = np.where(norms > 0, norms, 1.0)
    unit = a / safe[:, None]
    sim = unit @ unit.T
    sim = np.clip(sim, -1.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim


def euclidean_distance_matrix(x: ArrayLike) -> NDArray[np.float64]:
    """Pairwise Euclidean distance of the rows of ``x`` (all coordinates weighted equally)."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("Expected a 2-D array")
    sq = np.sum(a * a, axis=1)
    d2 = np.maximum(sq[:, None] + sq[None, :] - 2.0 * (a @ a.T), 0.0)
    d = np.sqrt(d2)
    np.fill_diagonal(d, 0.0)
    return d


def color_emd(
    centroids_a: ArrayLike, shares_a: ArrayLike, centroids_b: ArrayLike, shares_b: ArrayLike
) -> float:
    """Earth Mover's distance between two weighted colour palettes (in RGB units).

    Solves ``min Σ f_km c_km`` s.t. rows sum to ``p``, columns to ``q``, ``f ≥ 0``.
    Weights are renormalised to sum to 1.
    """
    ca = np.asarray(centroids_a, dtype=np.float64).reshape(-1, 3)
    cb = np.asarray(centroids_b, dtype=np.float64).reshape(-1, 3)
    p = np.asarray(shares_a, dtype=np.float64).ravel()
    q = np.asarray(shares_b, dtype=np.float64).ravel()
    if p.size != ca.shape[0] or q.size != cb.shape[0]:
        raise ValueError("Each palette needs one share per centroid")
    if p.sum() <= 0 or q.sum() <= 0 or np.any(p < 0) or np.any(q < 0):
        raise ValueError("Shares must be non-negative with a positive sum")
    p, q = p / p.sum(), q / q.sum()
    cost = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=2)
    k, m = cost.shape
    a_eq = np.zeros((k + m, k * m))
    for i in range(k):
        a_eq[i, i * m : (i + 1) * m] = 1.0
    for j in range(m):
        a_eq[k + j, j::m] = 1.0
    b_eq = np.concatenate([p, q])
    res = linprog(cost.ravel(), A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not res.success:  # pragma: no cover - balanced transport problems are always feasible
        raise RuntimeError(f"EMD optimisation failed: {res.message}")
    return float(max(res.fun, 0.0))


def color_emd_matrix(
    palettes: Sequence[tuple[ArrayLike, ArrayLike]],
) -> NDArray[np.float64]:
    """Pairwise colour EMD for a list of ``(centroids, shares)`` palettes."""
    n = len(palettes)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = color_emd(*palettes[i], *palettes[j])
    return out
