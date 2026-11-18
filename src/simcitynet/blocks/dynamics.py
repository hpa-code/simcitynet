"""Pedestrian and vehicle blocks ``c_ped`` and ``d_¬ped`` (paper Section 4.3).

Counts detections whose confidence is at least τ, for ``person`` and the eight COCO classes
bicycle, car, motorcycle, airplane, bus, train, truck and boat.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from ..backends.base import Detection
from ..features import VEHICLE_CLASSES

COUNTED_CLASSES: tuple[str, ...] = ("person", *VEHICLE_CLASSES)

# Common alternative spellings produced by some detectors.
_ALIASES = {"motorbike": "motorcycle", "aeroplane": "airplane", "pedestrian": "person"}


@dataclass
class DynamicsResult:
    """Counts per class plus the detections that were counted."""

    counts: dict[str, int]
    detections: list[Detection]


def count_objects(detections: Iterable[Detection], threshold: float) -> DynamicsResult:
    """Count detections of the nine dynamic classes with ``confidence >= threshold``."""
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    counts = dict.fromkeys(COUNTED_CLASSES, 0)
    kept: list[Detection] = []
    for det in detections:
        label = _ALIASES.get(det.label.lower(), det.label.lower())
        if label in counts and det.confidence >= threshold:
            counts[label] += 1
            kept.append(Detection(label, det.confidence, det.box))
    return DynamicsResult(counts, kept)
