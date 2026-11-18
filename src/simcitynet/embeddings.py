"""The :class:`Embeddings` result: a table of 33-D vectors plus similarity helpers and plots."""

from __future__ import annotations

import json
import os
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from .blocks.color import ColorResult
from .features import BLOCKS, FEATURE_NAMES, VEHICLE_CLASSES
from .normalize import NormalizeMethod, complete_columns, normalize
from .similarity import color_emd_matrix, cosine_similarity_matrix, euclidean_distance_matrix

if TYPE_CHECKING:
    from .scene import SceneResult
    from .viz import Plotter

DistanceMetric = Literal["euclidean", "cosine", "color_emd"]


class Embeddings:
    """SimCityNet embeddings for a set of images.

    Attributes:
        raw: ``(n, 33)`` DataFrame of un-normalised values, one row per image.
        meta: Extra per-image columns (video timestamp, per-class vegetation indices).
        scenes: Per-image diagnostics used by the plots (``None`` if not kept).
        provenance: Versions, settings and device that produced the embeddings.
    """

    def __init__(
        self,
        raw: pd.DataFrame,
        *,
        scenes: Sequence[SceneResult] | None = None,
        meta: pd.DataFrame | None = None,
        provenance: Mapping[str, Any] | None = None,
        count_caps: Mapping[str, float] | None = None,
    ) -> None:
        missing = [c for c in FEATURE_NAMES if c not in raw.columns]
        if missing:
            raise ValueError(f"Embedding table is missing columns: {missing}")
        if raw.index.has_duplicates:
            raise ValueError("Image names must be unique")
        self.raw = raw[list(FEATURE_NAMES)].astype(float)
        self.scenes = list(scenes) if scenes is not None else None
        self.meta = meta if meta is not None else pd.DataFrame(index=self.raw.index)
        self.provenance = dict(provenance or {})
        if count_caps is None:
            count_caps = self.provenance.get("config", {}).get("count_caps")
        self._count_caps = dict(count_caps) if count_caps else None

    # --------------------------------------------------------------------------- basics
    def __len__(self) -> int:
        return len(self.raw)

    def __repr__(self) -> str:
        return f"<Embeddings: {len(self)} images × {self.raw.shape[1]} features>"

    @property
    def names(self) -> list[str]:
        """Image names (row labels)."""
        return [str(n) for n in self.raw.index]

    def _row(self, image: int | str) -> int:
        if isinstance(image, (int, np.integer)):
            if not -len(self) <= image < len(self):
                raise IndexError(f"Image index {image} out of range for {len(self)} images")
            return int(image) % len(self)
        if image not in self.raw.index:
            raise KeyError(f"No image named {image!r}")
        return int(self.raw.index.get_loc(image))  # type: ignore[arg-type]

    def scene(self, image: int | str) -> SceneResult:
        """Diagnostics (preview, maps, detections) for one image."""
        if self.scenes is None:
            raise ValueError(
                "No scene diagnostics stored; embed with keep_scenes=True to use image plots"
            )
        return self.scenes[self._row(image)]

    def block(self, name: str) -> pd.DataFrame:
        """Columns of one block: vegetation, color, pedestrians, vehicles or surfaces."""
        if name not in BLOCKS:
            raise KeyError(f"Unknown block {name!r}; choose from {list(BLOCKS)}")
        return self.raw[list(BLOCKS[name])]

    def colors(self, image: int | str) -> ColorResult:
        """Dominant colours of one image, rebuilt from its row."""
        row = self.raw.iloc[self._row(image)]
        cent = row[list(BLOCKS["color"][:15])].to_numpy(dtype=float).reshape(5, 3)
        shares = row[list(BLOCKS["color"][15:])].to_numpy(dtype=float)
        return ColorResult(cent, shares)

    # ------------------------------------------------------------------------ transforms
    def normalized(self, method: NormalizeMethod = "batch") -> pd.DataFrame:
        """Scaled copy of :attr:`raw` (see :func:`simcitynet.normalize.normalize`)."""
        return normalize(self.raw, method, self._count_caps)

    def _matrix(
        self, normalize_method: NormalizeMethod, blocks: Sequence[str] | None
    ) -> pd.DataFrame:
        cols = list(FEATURE_NAMES) if blocks is None else [c for b in blocks for c in BLOCKS[b]]
        data = self.normalized(normalize_method)[cols]
        complete = complete_columns(data)
        dropped = sorted(set(data.columns) - set(complete.columns))
        if dropped:
            warnings.warn(
                f"Ignoring {len(dropped)} features with missing values (blocks not computed?)",
                stacklevel=3,
            )
        if complete.shape[1] == 0:
            raise ValueError("No complete features to compare")
        return complete

    def similarity(
        self,
        *,
        normalize: NormalizeMethod = "batch",
        blocks: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Pairwise cosine similarity on normalised vectors (paper Section 4).

        Args:
            normalize: Scaling applied first; ``"batch"`` is the paper's rule.
            blocks: Restrict to some blocks, e.g. ``["vegetation", "surfaces"]``.
        """
        x = self._matrix(normalize, blocks)
        return pd.DataFrame(cosine_similarity_matrix(x.to_numpy()), index=x.index, columns=x.index)

    def distance(
        self,
        metric: DistanceMetric = "euclidean",
        *,
        normalize: NormalizeMethod = "batch",
        blocks: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Pairwise distances.

        * ``"euclidean"`` — equal-weight Euclidean distance on normalised vectors (Section 4).
        * ``"cosine"`` — ``1 − cosine similarity``.
        * ``"color_emd"`` — Earth Mover's distance between colour palettes in RGB units
          (Section 6.1); ignores ``normalize`` and ``blocks``.
        """
        index = self.raw.index
        if metric == "color_emd":
            if bool(self.block("color").isna().to_numpy().any()):
                raise ValueError("The colour block was not computed for every image")
            palettes = [(c.centroids, c.shares) for c in (self.colors(i) for i in range(len(self)))]
            return pd.DataFrame(color_emd_matrix(palettes), index=index, columns=index)
        if metric == "cosine":
            return 1.0 - self.similarity(normalize=normalize, blocks=blocks)
        if metric == "euclidean":
            x = self._matrix(normalize, blocks)
            return pd.DataFrame(
                euclidean_distance_matrix(x.to_numpy()), index=x.index, columns=x.index
            )
        raise ValueError(f"Unknown metric {metric!r}")

    def nearest(
        self,
        image: int | str,
        k: int = 5,
        *,
        normalize: NormalizeMethod = "batch",
        blocks: Sequence[str] | None = None,
    ) -> pd.Series:
        """The ``k`` most similar other images by cosine similarity, best first."""
        sim = self.similarity(normalize=normalize, blocks=blocks)
        name = self.raw.index[self._row(image)]
        return sim[name].drop(index=name).sort_values(ascending=False).head(k)

    def summary(self) -> pd.DataFrame:
        """Eight interpretable scalars per image, used by the radar charts.

        greenery, pedestrians, vehicles (all eight classes), road, sidewalk, building,
        brightness (share-weighted luminance of the palette, 0-1) and colourfulness
        (share-weighted HSV saturation of the palette, 0-1).
        """
        raw = self.raw
        out = pd.DataFrame(index=raw.index)
        out["greenery"] = raw["vegetation"]
        out["pedestrians"] = raw["person"]
        out["vehicles"] = raw[list(VEHICLE_CLASSES)].sum(axis=1, min_count=len(VEHICLE_CLASSES))
        out["road"] = raw["road_share"]
        out["sidewalk"] = raw["sidewalk_share"]
        out["building"] = raw["building_share"]
        bright, sat = [], []
        for i in range(len(self)):
            c = self.colors(i)
            rgb = c.centroids / 255.0
            lum = rgb @ np.array([0.2126, 0.7152, 0.0722])
            mx, mn = rgb.max(axis=1), rgb.min(axis=1)
            s = np.where(mx > 0, (mx - mn) / np.where(mx > 0, mx, 1), 0.0)
            w = c.shares
            bright.append(float(np.nansum(lum * w)))
            sat.append(float(np.nansum(s * w)))
        out["brightness"] = bright
        out["colourfulness"] = sat
        if bool(raw[list(BLOCKS["color"])].isna().to_numpy().any()):
            out.loc[
                raw[list(BLOCKS["color"])].isna().any(axis=1), ["brightness", "colourfulness"]
            ] = np.nan
        return out

    # ---------------------------------------------------------------------------- plots
    @property
    def plot(self) -> Plotter:
        """Plotting helpers: ``scene_card``, ``palette``, ``radar``, ``heatmap``, ``map``,
        ``neighbors``. See :class:`simcitynet.viz.Plotter`."""
        from .viz import Plotter

        return Plotter(self)

    # ------------------------------------------------------------------------------- I/O
    def to_frame(self, normalize_method: NormalizeMethod | None = None) -> pd.DataFrame:
        """Raw (default) or normalised embeddings as a DataFrame."""
        return self.raw.copy() if normalize_method is None else self.normalized(normalize_method)

    def save(self, path: str | os.PathLike[str]) -> Path:
        """Write ``<path>.csv`` (embeddings + meta) and ``<path>.json`` (provenance).

        Returns the CSV path. Scene diagnostics are not saved.
        """
        base = Path(path)
        if base.suffix.lower() in {".csv", ".json"}:
            base = base.with_suffix("")
        base.parent.mkdir(parents=True, exist_ok=True)
        csv = base.with_suffix(".csv")
        pd.concat([self.raw, self.meta], axis=1).to_csv(csv)
        base.with_suffix(".json").write_text(json.dumps(self.provenance, indent=2, default=str))
        return csv

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> Embeddings:
        """Read embeddings written by :meth:`save` (or any CSV with the 33 columns)."""
        base = Path(path)
        csv = base if base.suffix.lower() == ".csv" else base.with_suffix(".csv")
        table = pd.read_csv(csv, index_col=0)
        prov_path = csv.with_suffix(".json")
        prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}
        meta_cols = [c for c in table.columns if c not in FEATURE_NAMES]
        return cls(table[list(FEATURE_NAMES)], meta=table[meta_cols], provenance=prov)

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> Embeddings:
        """Wrap an existing ``(n, 33)`` DataFrame."""
        return cls(frame)
