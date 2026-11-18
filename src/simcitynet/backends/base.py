"""Model interfaces used by the pipeline.

Each heavy model sits behind a tiny protocol so that the feature blocks can be tested with
fake models and so users can plug in their own (e.g. a fine-tuned vegetation segmenter).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

#: An RGB image as a ``(H, W, 3)`` uint8 array.
RGBImage = NDArray[np.uint8]


@dataclass(frozen=True)
class Detection:
    """One object detection."""

    label: str
    confidence: float
    #: Bounding box ``(x0, y0, x1, y1)`` in pixel coordinates of the analysed image.
    box: tuple[float, float, float, float]


@runtime_checkable
class Segmenter(Protocol):
    """Semantic segmentation model: image -> per-pixel integer class ids."""

    @property
    def labels(self) -> Mapping[int, str]:
        """Class id -> class name (lower case)."""
        ...

    def segment(self, image: RGBImage) -> NDArray[np.integer]:
        """Return an ``(H, W)`` array of class ids aligned with ``image``."""
        ...


@runtime_checkable
class DepthEstimator(Protocol):
    """Monocular depth model."""

    def estimate(self, image: RGBImage) -> NDArray[np.floating]:
        """Return an ``(H, W)`` map where **larger values mean closer** to the camera.

        DPT/MiDaS models predict relative inverse depth, which already has this property.
        """
        ...


@runtime_checkable
class Detector(Protocol):
    """Object detector returning labelled boxes."""

    def detect(self, image: RGBImage) -> Sequence[Detection]:
        """Return all detections (any confidence; the pipeline applies the threshold)."""
        ...


def find_label_ids(
    labels: Mapping[int, str], wanted: Sequence[str], fallback: Mapping[str, int]
) -> dict[str, int]:
    """Map wanted class names to ids, by name first and by ``fallback`` ids otherwise.

    Matching is case-insensitive and also accepts ADE20K-style names such as
    ``"tree"`` inside ``"tree;trees"``.
    """
    normalized = {i: str(n).lower() for i, n in labels.items()}
    out: dict[str, int] = {}
    for name in wanted:
        exact = [i for i, n in normalized.items() if n == name]
        if exact:
            out[name] = exact[0]
            continue
        partial = [
            i
            for i, n in normalized.items()
            if name in [p.strip() for p in n.replace(";", ",").split(",")]
        ]
        if partial:
            out[name] = partial[0]
        elif name in fallback:
            out[name] = fallback[name]
    return out
