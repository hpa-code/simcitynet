"""Surface block ``u``: pixel proportions of road, sidewalk and building (paper Section 4.4)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import NDArray

from ..backends.base import find_label_ids
from ..features import SURFACE_CLASSES


def surface_shares(
    segmentation: NDArray[np.integer],
    labels: Mapping[int, str],
    fallback_ids: Mapping[str, int],
) -> dict[str, float]:
    """Fraction of pixels labelled road, sidewalk and building (sum ≤ 1)."""
    total = segmentation.size
    if total == 0:
        raise ValueError("Empty segmentation map")
    ids = find_label_ids(labels, SURFACE_CLASSES, fallback_ids)
    return {
        name: (float(np.count_nonzero(segmentation == ids[name])) / total if name in ids else 0.0)
        for name in SURFACE_CLASSES
    }
