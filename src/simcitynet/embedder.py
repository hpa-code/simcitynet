"""The :class:`SimCityNet` pipeline: images in, interpretable 33-D embeddings out."""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image

from . import provenance
from .backends import resolve_device
from .backends.base import DepthEstimator, Detection, Detector, Segmenter, find_label_ids
from .blocks import compute_vegetation, count_objects, dominant_colors, surface_shares
from .config import Config
from .embeddings import Embeddings
from .features import FEATURE_NAMES, SURFACE_CLASSES, VEHICLE_CLASSES
from .io import ImageLike, iter_inputs, sample_video_frames
from .preprocess import prepare
from .scene import SceneResult, downsize

log = logging.getLogger(__name__)

ALL_BLOCKS: tuple[str, ...] = ("vegetation", "color", "pedestrians", "vehicles", "surfaces")


class SimCityNet:
    """Compute SimCityNet embeddings for street-level images.

    Models are loaded lazily, the first time a block needs them. Pass your own model objects
    (anything implementing :class:`~simcitynet.backends.Segmenter`,
    :class:`~simcitynet.backends.DepthEstimator` or :class:`~simcitynet.backends.Detector`)
    to swap a model or to run without downloads.

    Args:
        config: Pipeline settings; see :class:`simcitynet.Config`.
        blocks: Which of ``"vegetation"``, ``"color"``, ``"pedestrians"``, ``"vehicles"``,
            ``"surfaces"`` to compute. Skipped blocks are ``NaN`` in the output. ``"color"``
            alone needs no deep-learning models.
        vegetation_segmenter: Overrides the ADE20K Mask2Former model.
        depth_estimator: Overrides the DPT depth model.
        surface_segmenter: Overrides the Cityscapes Mask2Former model.
        detector: Overrides the YOLOv9e detector.
        **config_overrides: Shortcut for ``config.with_(...)``, e.g. ``device="cuda"``.

    Example:
        >>> from simcitynet import SimCityNet
        >>> scn = SimCityNet()                                  # doctest: +SKIP
        >>> emb = scn.embed(["zurich.jpg", "doha.jpg"])         # doctest: +SKIP
        >>> emb.similarity()                                    # doctest: +SKIP
    """

    def __init__(
        self,
        config: Config | None = None,
        *,
        blocks: Sequence[str] = ALL_BLOCKS,
        vegetation_segmenter: Segmenter | None = None,
        depth_estimator: DepthEstimator | None = None,
        surface_segmenter: Segmenter | None = None,
        detector: Detector | None = None,
        **config_overrides: Any,
    ) -> None:
        cfg = config or Config()
        self.config = cfg.with_(**config_overrides) if config_overrides else cfg
        unknown = set(blocks) - set(ALL_BLOCKS)
        if unknown:
            raise ValueError(f"Unknown blocks {sorted(unknown)}; choose from {list(ALL_BLOCKS)}")
        if not blocks:
            raise ValueError("At least one block is required")
        self.blocks = tuple(b for b in ALL_BLOCKS if b in blocks)
        self._vegetation_segmenter = vegetation_segmenter
        self._depth_estimator = depth_estimator
        self._surface_segmenter = surface_segmenter
        self._detector = detector
        self._device: str | None = None

    # ----------------------------------------------------------------------------- models
    @property
    def device(self) -> str:
        """The resolved compute device (``cuda``, ``mps`` or ``cpu``)."""
        if self._device is None:
            self._device = resolve_device(self.config.device)
        return self._device

    @property
    def vegetation_segmenter(self) -> Segmenter:
        if self._vegetation_segmenter is None:
            from .backends.huggingface import Mask2FormerSegmenter

            log.info("Loading vegetation segmenter %s", self.config.vegetation_model)
            self._vegetation_segmenter = Mask2FormerSegmenter(
                self.config.vegetation_model, self.device
            )
        return self._vegetation_segmenter

    @property
    def depth_estimator(self) -> DepthEstimator:
        if self._depth_estimator is None:
            from .backends.huggingface import DPTDepthEstimator

            log.info("Loading depth model %s", self.config.depth_model)
            self._depth_estimator = DPTDepthEstimator(self.config.depth_model, self.device)
        return self._depth_estimator

    @property
    def surface_segmenter(self) -> Segmenter:
        if self._surface_segmenter is None:
            from .backends.huggingface import Mask2FormerSegmenter

            log.info("Loading surface segmenter %s", self.config.surface_model)
            self._surface_segmenter = Mask2FormerSegmenter(self.config.surface_model, self.device)
        return self._surface_segmenter

    @property
    def detector(self) -> Detector:
        if self._detector is None:
            from .backends.yolo import YOLODetector

            log.info("Loading detector %s", self.config.detector_weights)
            self._detector = YOLODetector(self.config.detector_weights, self.device)
        return self._detector

    # ---------------------------------------------------------------------------- compute
    def analyze(self, image: Image.Image | NDArray[np.generic], name: str = "image") -> SceneResult:
        """Run the pipeline on one in-memory image and return all diagnostics."""
        cfg = self.config
        rgb = prepare(image, cfg)
        vector = np.full(len(FEATURE_NAMES), np.nan)
        index = {n: i for i, n in enumerate(FEATURE_NAMES)}
        preview = downsize(rgb, cfg.preview_side)
        scale = preview.shape[1] / rgb.shape[1]

        colors = dominant_colors(rgb, cfg.n_colors, cfg.color_sample_pixels, cfg.random_state)
        if "color" in self.blocks:
            if cfg.n_colors != 5:
                raise ValueError("The 33-D embedding requires n_colors=5")
            vector[index["color1_r"] : index["color1_r"] + 20] = colors.vector()

        result = SceneResult(name=name, vector=vector, colors=colors, preview=preview)

        if "vegetation" in self.blocks:
            seg = self.vegetation_segmenter
            depth = self.depth_estimator.estimate(rgb) if cfg.depth_weighting else None
            veg = compute_vegetation(
                rgb,
                seg.segment(rgb),
                seg.labels,
                depth,
                cfg.vegetation_hsv,
                cfg.vegetation_fallback_ids,
            )
            vector[index["vegetation"]] = veg.value
            result.vegetation_per_class = veg.per_class
            result.vegetation_mask = downsize(veg.mask, cfg.preview_side, nearest=True)
            result.depth = downsize(veg.weights, cfg.preview_side)

        if "pedestrians" in self.blocks or "vehicles" in self.blocks:
            dyn = count_objects(self.detector.detect(rgb), cfg.detection_conf)
            if "pedestrians" in self.blocks:
                vector[index["person"]] = dyn.counts["person"]
            if "vehicles" in self.blocks:
                for cls in VEHICLE_CLASSES:
                    vector[index[cls]] = dyn.counts[cls]
            wanted = {"person"} if "pedestrians" in self.blocks else set()
            if "vehicles" in self.blocks:
                wanted |= set(VEHICLE_CLASSES)
            result.detections = [
                Detection(
                    d.label,
                    d.confidence,
                    (d.box[0] * scale, d.box[1] * scale, d.box[2] * scale, d.box[3] * scale),
                )
                for d in dyn.detections
                if d.label in wanted
            ]

        if "surfaces" in self.blocks:
            seg = self.surface_segmenter
            label_map = seg.segment(rgb)
            shares = surface_shares(label_map, seg.labels, cfg.surface_fallback_ids)
            for cls in SURFACE_CLASSES:
                vector[index[f"{cls}_share"]] = shares[cls]
            result.surface_map = _surface_codes(
                label_map, seg.labels, cfg, (preview.shape[0], preview.shape[1])
            )

        return result

    def embed(
        self,
        images: ImageLike | Iterable[ImageLike],
        *,
        names: Iterable[str] | None = None,
        keep_scenes: bool = True,
        progress: bool = False,
    ) -> Embeddings:
        """Embed one or more images.

        Args:
            images: A path, folder, glob pattern, PIL image or array, or a list of these.
            names: Row names (default: file stems, or ``image_0``, ``image_1``...).
            keep_scenes: Keep previews and maps for plotting (set False for large batches).
            progress: Log one line per image.

        Returns:
            An :class:`~simcitynet.Embeddings` with one row per image.
        """
        scenes: list[SceneResult] = []
        for i, (name, image) in enumerate(iter_inputs(images, names)):
            if progress:
                log.info("[%d] embedding %s", i + 1, name)
            scenes.append(self.analyze(image, name))
        if not scenes:
            raise ValueError("No images found")
        return self._collect(scenes, keep_scenes)

    def embed_video(
        self,
        path: str | os.PathLike[str],
        *,
        every: float = 1.0,
        max_frames: int | None = None,
        keep_scenes: bool = True,
    ) -> Embeddings:
        """Embed frames sampled every ``every`` seconds from a video file.

        Rows are named ``<stem>@<seconds>s``; the ``timestamp`` column of
        :attr:`Embeddings.meta` holds the time in seconds.
        """
        stem = os.path.splitext(os.path.basename(os.fspath(path)))[0]
        scenes = []
        for t, frame in sample_video_frames(path, every, max_frames):
            scene = self.analyze(frame, f"{stem}@{t:.1f}s")
            scene.timestamp = t
            scenes.append(scene)
        if not scenes:
            raise ValueError(f"No frames could be read from {path}")
        return self._collect(scenes, keep_scenes)

    def _collect(self, scenes: list[SceneResult], keep_scenes: bool) -> Embeddings:
        raw = pd.DataFrame(
            np.vstack([s.vector for s in scenes]),
            index=pd.Index([s.name for s in scenes], name="image"),
            columns=list(FEATURE_NAMES),
        )
        meta = pd.DataFrame(
            {
                "timestamp": np.array(
                    [np.nan if s.timestamp is None else s.timestamp for s in scenes], dtype=float
                ),
                **{
                    f"vegetation_{k}": [s.vegetation_per_class.get(k, np.nan) for s in scenes]
                    for k in ("tree", "grass", "plant")
                },
            },
            index=raw.index,
        )
        needs_models = any(b != "color" for b in self.blocks)
        record = provenance.collect(
            self.config.to_dict(), self.device if needs_models else "cpu", list(self.blocks)
        )
        return Embeddings(raw, scenes=scenes if keep_scenes else None, meta=meta, provenance=record)


def _surface_codes(
    label_map: NDArray[np.integer],
    labels: Any,
    cfg: Config,
    shape: tuple[int, int],
) -> NDArray[np.uint8]:
    ids = find_label_ids(labels, SURFACE_CLASSES, cfg.surface_fallback_ids)
    codes = np.zeros(label_map.shape, dtype=np.uint8)
    for code, cls in enumerate(SURFACE_CLASSES, start=1):
        if cls in ids:
            codes[label_map == ids[cls]] = code
    small: NDArray[np.uint8] = downsize(codes, max(shape), nearest=True)
    return small
