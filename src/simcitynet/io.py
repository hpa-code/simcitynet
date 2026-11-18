"""Loading images (files, folders, globs, arrays) and sampling frames from videos."""

from __future__ import annotations

import glob
import os
import sys
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageOps

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"})

#: Anything :meth:`simcitynet.SimCityNet.embed` accepts as one image.
ImageLike = Union[str, os.PathLike, Image.Image, NDArray[np.generic]]  # noqa: UP007


_PIL_OPEN = Image.open


def _pil_open() -> Callable[..., Image.Image]:
    """Pillow's own ``Image.open``.

    Ultralytics replaces ``PIL.Image.open`` with a wrapper that tries to ``pip install`` a HEIF
    plugin whenever a file cannot be identified. Reading user images must not install packages,
    so the original function is used even after Ultralytics has been imported.
    """
    current = Image.open
    if getattr(current, "__module__", "").startswith("ultralytics"):
        patches = sys.modules.get("ultralytics.utils.patches")
        original = getattr(patches, "_image_open", None)
        return original if callable(original) else _PIL_OPEN
    return current


def load_image(path: str | os.PathLike[str]) -> Image.Image:
    """Open an image file as RGB, applying its EXIF orientation."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Image not found: {p}")
    try:
        with _pil_open()(p) as im:
            return ImageOps.exif_transpose(im).convert("RGB")
    except OSError as exc:  # PIL raises UnidentifiedImageError (an OSError) for bad files
        raise ValueError(f"Could not read image {p}: {exc}") from exc


def expand_paths(sources: Iterable[str | os.PathLike[str]]) -> list[Path]:
    """Expand folders (non-recursive) and glob patterns into a sorted list of image files."""
    out: list[Path] = []
    for src in sources:
        s = str(src)
        p = Path(s)
        if p.is_dir():
            out.extend(sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES))
        elif any(ch in s for ch in "*?["):
            out.extend(
                sorted(Path(m) for m in glob.glob(s) if Path(m).suffix.lower() in IMAGE_SUFFIXES)
            )
        else:
            out.append(p)
    return out


def iter_inputs(
    images: ImageLike | Iterable[ImageLike], names: Iterable[str] | None = None
) -> Iterator[tuple[str, Image.Image | NDArray[np.generic]]]:
    """Yield ``(name, image)`` pairs from any supported input.

    Paths and folders are named by file stem; in-memory images are named ``image_0``,
    ``image_1``... unless ``names`` is given.
    """
    if isinstance(images, (str, os.PathLike, Image.Image, np.ndarray)):
        items: list[ImageLike] = [images]
    else:
        items = list(images)

    expanded: list[tuple[str | None, ImageLike]] = []
    for item in items:
        if isinstance(item, (str, os.PathLike)):
            for path in expand_paths([item]):
                expanded.append((path.stem, path))
        else:
            expanded.append((None, item))

    given = list(names) if names is not None else None
    if given is not None and len(given) != len(expanded):
        raise ValueError(f"Got {len(given)} names for {len(expanded)} images")

    seen: dict[str, int] = {}
    for i, (stem, item) in enumerate(expanded):
        name = given[i] if given is not None else (stem or f"image_{i}")
        if name in seen:  # keep names unique so they can index a DataFrame
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        if isinstance(item, (str, os.PathLike)):
            yield name, load_image(item)
        else:
            yield name, item


def sample_video_frames(
    path: str | os.PathLike[str], every: float = 1.0, max_frames: int | None = None
) -> Iterator[tuple[float, NDArray[np.uint8]]]:
    """Yield ``(timestamp_seconds, rgb_frame)`` every ``every`` seconds of a video."""
    if every <= 0:
        raise ValueError("`every` must be positive")
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Video not found: {p}")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError(f"Could not open video {p}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = 30.0
        step = max(1, round(every * fps))
        index = 0
        yielded = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % step == 0:
                yield (
                    index / fps,
                    np.asarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), dtype=np.uint8),
                )
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            index += 1
    finally:
        cap.release()
