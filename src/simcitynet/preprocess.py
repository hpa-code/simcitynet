"""Image normalisation: RGB conversion, black-border cropping and resizing."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .backends.base import RGBImage
from .config import Config


def to_rgb_array(image: Image.Image | NDArray[np.generic]) -> RGBImage:
    """Convert a PIL image or array (gray, RGB or RGBA; uint8 or float in [0, 1]) to RGB uint8."""
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"Expected an (H, W), (H, W, 3) or (H, W, 4) image, got shape {arr.shape}")
    arr = arr[..., :3]
    if np.issubdtype(arr.dtype, np.floating):
        if arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(np.nan_to_num(arr), 0, 255)
    return np.ascontiguousarray(arr.astype(np.uint8))


def crop_black_borders(
    image: RGBImage, threshold: int = 10, min_fraction: float = 0.02
) -> RGBImage:
    """Remove black borders (rows/columns whose bright-pixel fraction is below ``min_fraction``).

    Mirrors the paper's first step ("thresholded to identify the primary content window").
    Returns the input unchanged if nothing would remain.
    """
    gray = image.mean(axis=2)
    bright = gray > threshold
    rows = np.flatnonzero(bright.mean(axis=1) > min_fraction)
    cols = np.flatnonzero(bright.mean(axis=0) > min_fraction)
    if rows.size == 0 or cols.size == 0:
        return image
    return image[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def resize(image: RGBImage, size: tuple[int, int]) -> RGBImage:
    """Resize to ``(width, height)`` with Lanczos interpolation."""
    if (image.shape[1], image.shape[0]) == size:
        return image
    out = Image.fromarray(image).resize(size, Image.Resampling.LANCZOS)
    return np.asarray(out, dtype=np.uint8)


def cap_long_side(image: RGBImage, max_side: int) -> RGBImage:
    """Shrink so the longer side is at most ``max_side`` pixels (never enlarges)."""
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    return resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))


def prepare(image: Image.Image | NDArray[np.generic], config: Config) -> RGBImage:
    """Apply the configured preprocessing to one image."""
    rgb = to_rgb_array(image)
    if config.crop_borders:
        rgb = crop_black_borders(rgb, config.border_threshold)
    if config.panorama:
        return resize(rgb, config.panorama_size)
    return cap_long_side(rgb, config.max_side)
