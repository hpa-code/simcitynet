"""Shared fixtures: a synthetic street scene and fake models (no downloads, no GPU)."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib
import numpy as np
import pytest

from simcitynet import SimCityNet
from simcitynet.backends import Detection

matplotlib.use("Agg")

# Exact colours used to paint the synthetic scene; fake segmenters read them back.
SKY = (120, 170, 230)
BUILDING = (180, 120, 90)
TREE = (40, 140, 50)  # hue ~ 63 in OpenCV units: inside the vegetation HSV range
ROAD = (90, 90, 90)
SIDEWALK = (200, 190, 170)

ADE_LABELS = {
    0: "wall",
    1: "building",
    2: "sky",
    4: "tree",
    6: "road",
    9: "grass",
    11: "sidewalk",
    17: "plant",
}
CITY_LABELS = {0: "road", 1: "sidewalk", 2: "building", 8: "vegetation", 10: "sky"}


def street_scene(h: int = 120, w: int = 160) -> np.ndarray:
    """A tiny drawn street: sky, building, tree, sidewalk and road bands."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = SKY
    img[h // 6 : h // 2, w // 2 :] = BUILDING
    img[h // 6 : h // 2, : w // 4] = TREE
    img[h // 2 : 2 * h // 3] = SIDEWALK
    img[2 * h // 3 :] = ROAD
    return img


def _color_map(
    image: np.ndarray, table: dict[tuple[int, int, int], int], default: int
) -> np.ndarray:
    out = np.full(image.shape[:2], default, dtype=np.int32)
    for rgb, cls in table.items():
        out[np.all(image == np.array(rgb, dtype=np.uint8), axis=-1)] = cls
    return out


class FakeADESegmenter:
    labels = ADE_LABELS

    def segment(self, image: np.ndarray) -> np.ndarray:
        return _color_map(image, {SKY: 2, BUILDING: 1, TREE: 4, ROAD: 6, SIDEWALK: 11}, 0)


class FakeCityscapesSegmenter:
    labels = CITY_LABELS

    def segment(self, image: np.ndarray) -> np.ndarray:
        return _color_map(image, {SKY: 10, BUILDING: 2, TREE: 8, ROAD: 0, SIDEWALK: 1}, 10)


class FakeDepth:
    """Far at the top, close at the bottom (larger = closer)."""

    def estimate(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return np.repeat(np.linspace(0.0, 10.0, h, dtype=np.float32)[:, None], w, axis=1)


class FakeDetector:
    def __init__(self, detections: Sequence[Detection] | None = None) -> None:
        self.detections = list(
            detections
            if detections is not None
            else [
                Detection("person", 0.9, (10, 60, 20, 90)),
                Detection("person", 0.8, (30, 60, 40, 90)),
                Detection("person", 0.1, (50, 60, 60, 90)),  # below threshold
                Detection("car", 0.7, (60, 85, 110, 115)),
                Detection("motorbike", 0.6, (120, 90, 140, 115)),  # alias of motorcycle
                Detection("dog", 0.99, (0, 0, 5, 5)),  # not a counted class
            ]
        )

    def detect(self, image: np.ndarray) -> list[Detection]:
        return list(self.detections)


@pytest.fixture
def scene() -> np.ndarray:
    return street_scene()


@pytest.fixture
def fake_net() -> SimCityNet:
    return SimCityNet(
        vegetation_segmenter=FakeADESegmenter(),
        depth_estimator=FakeDepth(),
        surface_segmenter=FakeCityscapesSegmenter(),
        detector=FakeDetector(),
        device="cpu",
    )


@pytest.fixture
def scenes() -> list[np.ndarray]:
    """Three different synthetic scenes."""
    a = street_scene()
    b = street_scene()
    b[:60, :40] = SKY  # no tree
    c = street_scene()
    c[60:, :] = ROAD  # all road, no sidewalk
    return [a, b, c]
