"""Model backends. Heavy libraries are imported only when a backend is created."""

from __future__ import annotations

from .base import DepthEstimator, Detection, Detector, RGBImage, Segmenter, find_label_ids

__all__ = [
    "DepthEstimator",
    "Detection",
    "Detector",
    "RGBImage",
    "Segmenter",
    "find_label_ids",
    "resolve_device",
]


def resolve_device(device: str = "auto") -> str:
    """Pick ``cuda`` or ``mps`` when available for ``"auto"``, else ``cpu``."""
    if device != "auto":
        return device
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"
