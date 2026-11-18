"""Configuration for the SimCityNet pipeline.

Every default is documented in ``docs/faithfulness.md``. Values the paper states are marked
"paper"; values the paper leaves open are marked "default" with the reason.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

from .features import VEHICLE_CLASSES

Device = Literal["auto", "cpu", "cuda", "mps"]

#: Hue/saturation/value bounds in OpenCV units (H 0-179, S 0-255, V 0-255).
HSVRange = tuple[tuple[int, int, int], tuple[int, int, int]]


def _default_hsv() -> dict[str, HSVRange]:
    # Paper: trees use hue-saturation [20, 40]-[95, 255]. The paper gives no thresholds for
    # grass and plants, so the same range is used for all three (default).
    tree: HSVRange = ((20, 40, 0), (95, 255, 255))
    return {"tree": tree, "grass": tree, "plant": tree}


def _default_count_caps() -> dict[str, float]:
    # Used only by normalize="fixed". The paper divides by the corpus maximum, which is not
    # published, so these caps are round numbers for busy urban scenes (default).
    caps = {"person": 50.0, "car": 30.0}
    caps.update({c: 10.0 for c in VEHICLE_CLASSES if c not in caps})
    return caps


@dataclass(frozen=True)
class Config:
    """All tunable settings of :class:`simcitynet.SimCityNet`.

    Use :meth:`with_` to derive a modified copy, e.g. ``Config().with_(detection_conf=0.4)``.
    """

    # --- hardware -------------------------------------------------------------------------
    device: Device = "auto"

    # --- models (Hugging Face model IDs / Ultralytics weights) -----------------------------
    #: Mask2Former trained on ADE20K, used for vegetation masks (paper).
    vegetation_model: str = "facebook/mask2former-swin-large-ade-semantic"
    #: Monocular depth model used to weight vegetation (paper: a Dense Prediction Transformer).
    depth_model: str = "Intel/dpt-large"
    #: Street-scene segmenter for road / sidewalk / building. The paper names DeepLabV3+ on
    #: Cityscapes "e.g."; Mask2Former on Cityscapes keeps one model library (default).
    surface_model: str = "facebook/mask2former-swin-large-cityscapes-semantic"
    #: Object detector weights (paper: YOLOv9e).
    detector_weights: str = "yolov9e.pt"

    # --- preprocessing ---------------------------------------------------------------------
    #: Remove black borders before analysis (paper).
    crop_borders: bool = True
    #: Grey level below which a border pixel counts as black.
    border_threshold: int = 10
    #: Treat inputs as panoramas and resize to ``panorama_size`` (paper uses 4000x2000).
    panorama: bool = False
    panorama_size: tuple[int, int] = (4000, 2000)
    #: For ordinary photos, cap the longer side (keeps aspect ratio; default).
    max_side: int = 2048

    # --- vegetation ------------------------------------------------------------------------
    vegetation_hsv: dict[str, HSVRange] = field(default_factory=_default_hsv)
    #: Paper ADE20K indices for tree, grass, plant; used when label names are unavailable.
    vegetation_fallback_ids: dict[str, int] = field(
        default_factory=lambda: {"tree": 4, "grass": 9, "plant": 17}
    )
    #: Weight vegetation by monocular depth (paper). If False, a plain Green View Index.
    depth_weighting: bool = True

    # --- colour ----------------------------------------------------------------------------
    n_colors: int = 5
    #: Pixels sampled for K-means (speed; default).
    color_sample_pixels: int = 50_000
    random_state: int = 0

    # --- detection -------------------------------------------------------------------------
    #: Confidence threshold τ (not stated in the paper; Ultralytics default).
    detection_conf: float = 0.25

    # --- surfaces --------------------------------------------------------------------------
    surface_fallback_ids: dict[str, int] = field(
        default_factory=lambda: {"road": 0, "sidewalk": 1, "building": 2}
    )

    # --- normalisation ---------------------------------------------------------------------
    count_caps: dict[str, float] = field(default_factory=_default_count_caps)

    # --- outputs ---------------------------------------------------------------------------
    #: Longest side of thumbnails and diagnostic maps kept for plotting.
    preview_side: int = 512

    def with_(self, **changes: Any) -> Config:
        """Return a copy with some fields changed."""
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly dictionary of all settings."""
        return asdict(self)
