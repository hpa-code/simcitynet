"""Per-image result with the 33-dimensional vector and diagnostics for plotting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .backends.base import Detection, RGBImage
from .blocks.color import ColorResult
from .features import FEATURE_NAMES

#: Codes used in :attr:`SceneResult.surface_map`.
SURFACE_CODES = {0: "other", 1: "road", 2: "sidewalk", 3: "building"}


@dataclass
class SceneResult:
    """Everything SimCityNet computed for one image.

    Maps are stored at preview resolution (``Config.preview_side``) to keep memory small;
    detection boxes are rescaled to the same resolution.
    """

    name: str
    #: The raw (un-normalised) 33-dimensional embedding, in :data:`FEATURE_NAMES` order.
    vector: NDArray[np.float64]
    colors: ColorResult
    #: Downsized RGB copy of the analysed image.
    preview: RGBImage
    #: Vegetation mask at preview resolution (``None`` if the block was skipped).
    vegetation_mask: NDArray[np.bool_] | None = None
    #: Depth weights in [0, 1] at preview resolution.
    depth: NDArray[np.float32] | None = None
    #: Per-class depth-weighted GVI (tree, grass, plant).
    vegetation_per_class: dict[str, float] = field(default_factory=dict)
    #: Surface codes at preview resolution, see :data:`SURFACE_CODES`.
    surface_map: NDArray[np.uint8] | None = None
    #: Counted detections, boxes in preview pixel coordinates.
    detections: list[Detection] = field(default_factory=list)
    #: Seconds into the video, for frames from :meth:`SimCityNet.embed_video`.
    timestamp: float | None = None

    def as_series(self) -> pd.Series:
        """The raw embedding as a named Series."""
        return pd.Series(self.vector, index=list(FEATURE_NAMES), name=self.name)


def downsize(image: NDArray[Any], max_side: int, nearest: bool = False) -> NDArray[Any]:
    """Shrink an image or label map so its longer side is at most ``max_side``."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_AREA
    src: Any = image.astype(np.uint8) if image.dtype == bool else image
    out = cv2.resize(src, size, interpolation=interp)
    return out.astype(bool) if image.dtype == bool else out
