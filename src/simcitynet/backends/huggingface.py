"""Hugging Face Transformers adapters: Mask2Former segmentation and DPT depth.

Requires the ``models`` extra: ``pip install "simcitynet[models]"``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .base import RGBImage


def _require_transformers() -> tuple[Any, Any]:
    try:
        import torch
        import transformers
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "The vegetation, depth and surface models need PyTorch and Transformers. "
            'Install them with: pip install "simcitynet[models]"'
        ) from exc
    return torch, transformers


class Mask2FormerSegmenter:
    """Semantic segmentation with a Mask2Former checkpoint from the Hugging Face Hub."""

    def __init__(self, model_id: str, device: str = "cpu") -> None:
        torch, transformers = _require_transformers()
        self._torch = torch
        self.model_id = model_id
        self.device = device
        self.processor = transformers.AutoImageProcessor.from_pretrained(model_id)
        self.model = transformers.Mask2FormerForUniversalSegmentation.from_pretrained(model_id)
        self.model.to(device).eval()
        self._labels = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}

    @property
    def labels(self) -> Mapping[int, str]:
        return self._labels

    def segment(self, image: RGBImage) -> NDArray[np.integer]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with self._torch.inference_mode():
            outputs = self.model(**inputs)
        seg = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.shape[:2]]
        )[0]
        return np.asarray(seg.cpu().numpy(), dtype=np.int32)


class DPTDepthEstimator:
    """Monocular relative inverse depth with a DPT checkpoint (larger = closer)."""

    def __init__(self, model_id: str, device: str = "cpu") -> None:
        torch, transformers = _require_transformers()
        self._torch = torch
        self.model_id = model_id
        self.device = device
        self.processor = transformers.AutoImageProcessor.from_pretrained(model_id)
        self.model = transformers.AutoModelForDepthEstimation.from_pretrained(model_id)
        self.model.to(device).eval()

    def estimate(self, image: RGBImage) -> NDArray[np.floating]:
        torch = self._torch
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            predicted = self.model(**inputs).predicted_depth
        resized = torch.nn.functional.interpolate(
            predicted.unsqueeze(1), size=image.shape[:2], mode="bicubic", align_corners=False
        )
        return np.asarray(resized.squeeze().float().cpu().numpy(), dtype=np.float32)
