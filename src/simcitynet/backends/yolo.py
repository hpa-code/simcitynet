"""Ultralytics YOLO adapter (paper default: YOLOv9e).

Requires the ``yolo`` extra: ``pip install "simcitynet[yolo]"``. Note that Ultralytics is
licensed under AGPL-3.0; see MODEL_LICENSES.md.
"""

from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from .base import Detection, RGBImage


class YOLODetector:
    """Object detection with Ultralytics YOLO weights (downloaded on first use)."""

    def __init__(self, weights: str = "yolov9e.pt", device: str = "cpu") -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "Pedestrian and vehicle counts need Ultralytics YOLO. "
                'Install it with: pip install "simcitynet[yolo]"'
            ) from exc
        self.weights = weights
        self.device = device
        self.model = YOLO(weights)

    def detect(self, image: RGBImage) -> Sequence[Detection]:
        # conf=0.01 keeps low-confidence boxes; the pipeline applies the configured threshold.
        result = self.model.predict(
            Image.fromarray(image), conf=0.01, device=self.device, verbose=False
        )[0]
        names = result.names
        boxes = result.boxes
        out: list[Detection] = []
        for cls, conf, xyxy in zip(
            boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), strict=True
        ):
            x0, y0, x1, y1 = (float(v) for v in xyxy)
            out.append(Detection(str(names[int(cls)]).lower(), float(conf), (x0, y0, x1, y1)))
        return out
