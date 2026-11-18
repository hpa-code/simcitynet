"""Plots for SimCityNet embeddings (matplotlib; each method returns a ``Figure``).

Access them through :attr:`Embeddings.plot`::

    emb.plot.scene_card("zurich")   # one image: overlays, palette, key numbers
    emb.plot.heatmap()              # pairwise similarity with thumbnails
    emb.plot.map()                  # images placed by similarity (PCA)
    emb.plot.neighbors("zurich")    # most similar images
    emb.plot.radar(["zurich", "doha"])
    emb.plot.palette("zurich")

Colours follow a colour-blind-checked categorical order; magnitudes use a single-hue ramp.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.figure import Figure
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Rectangle
from numpy.typing import NDArray

from .normalize import NormalizeMethod

if TYPE_CHECKING:
    from .embeddings import Embeddings

# --- palette (light surface) ----------------------------------------------------------------
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
MUTED = "#8a8985"
GRID = "#e4e3df"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GREEN = "#008300"
BLUES = LinearSegmentedColormap.from_list(
    "simcitynet_blues",
    ["#f4f8fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
)
#: Surface classes in the overlay, in categorical order (road, sidewalk, building).
SURFACE_COLORS = {1: SERIES[0], 2: SERIES[1], 3: SERIES[2]}
SURFACE_NAMES = {1: "Road", 2: "Sidewalk", 3: "Building"}

RADAR_AXES = (
    "greenery",
    "pedestrians",
    "vehicles",
    "road",
    "sidewalk",
    "building",
    "brightness",
    "colourfulness",
)


def _figure(width: float, height: float) -> Figure:
    fig = Figure(figsize=(width, height), dpi=110, facecolor=SURFACE, layout="constrained")
    return fig


def _clean(ax: Axes, title: str | None = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=10, color=TEXT, pad=6, fontweight="bold")


def _gray(rgb: NDArray[np.uint8]) -> NDArray[np.float64]:
    g = rgb.astype(float) @ np.array([0.299, 0.587, 0.114]) / 255.0
    return np.repeat(g[..., None], 3, axis=2)


def _blend(
    base: NDArray[np.float64], color: str, mask: NDArray[np.bool_], alpha: Any
) -> NDArray[np.float64]:
    out = base.copy()
    col = np.array(to_rgb(color))
    a = np.asarray(alpha, dtype=float)
    a = np.broadcast_to(a, mask.shape) if a.ndim == 0 else a
    w = (mask * a)[..., None]
    return out * (1 - w) + col * w


def _text_on(value: float) -> str:
    rgba = BLUES(value)
    lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
    return TEXT if lum > 0.55 else "#ffffff"


def _fmt_pct(v: float) -> str:
    return "–" if np.isnan(v) else f"{100 * v:.0f}%"


def _fmt_int(v: float) -> str:
    return "–" if np.isnan(v) else f"{v:.0f}"


class Plotter:
    """Plots bound to one :class:`~simcitynet.Embeddings` object."""

    def __init__(self, embeddings: Embeddings) -> None:
        self.emb = embeddings

    # ---------------------------------------------------------------------- single image
    def palette(self, image: int | str = 0, ax: Axes | None = None) -> Figure:
        """The five dominant colours as a strip whose widths are their pixel shares."""
        colors = self.emb.colors(image)
        fig: Figure
        if ax is None:
            fig = _figure(6, 1.4)
            ax = fig.add_subplot()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]
        left = 0.0
        for rgb, share, hx in zip(colors.centroids, colors.shares, colors.hex(), strict=True):
            if share <= 0 or np.isnan(share):
                continue
            ax.add_patch(
                Rectangle(
                    (left, 0),
                    share,
                    1,
                    facecolor=np.clip(rgb / 255, 0, 1),
                    edgecolor=SURFACE,
                    linewidth=2,
                )
            )
            if share >= 0.08:
                ink = TEXT if (rgb @ np.array([0.2126, 0.7152, 0.0722])) / 255 > 0.55 else "#ffffff"
                ax.text(
                    left + share / 2,
                    0.5,
                    f"{hx}\n{share:.0%}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=ink,
                )
            left += share
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        name = self.emb.names[self.emb._row(image)]
        _clean(ax, f"Colour palette · {name}")
        return fig

    def scene_card(self, image: int | str = 0) -> Figure:
        """One-page summary of an image: photo, vegetation, surfaces, detections, palette, key numbers."""
        scene = self.emb.scene(image)
        row = self.emb.raw.iloc[self.emb._row(image)]
        fig = _figure(13, 7.6)
        grid = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.42])
        photo = scene.preview
        base = _gray(photo) * 0.85 + 0.15

        ax = fig.add_subplot(grid[0, 0])
        ax.imshow(photo)
        _clean(ax, scene.name)

        ax = fig.add_subplot(grid[0, 1])
        if scene.vegetation_mask is not None:
            depth = scene.depth if scene.depth is not None else np.ones(scene.vegetation_mask.shape)
            ax.imshow(_blend(base, GREEN, scene.vegetation_mask, 0.35 + 0.55 * depth))
            _clean(ax, f"Vegetation · {_fmt_pct(row['vegetation'])} depth-weighted")
        else:
            ax.imshow(base)
            _clean(ax, "Vegetation · not computed")

        ax = fig.add_subplot(grid[0, 2])
        if scene.surface_map is not None:
            over = base
            for code, color in SURFACE_COLORS.items():
                over = _blend(over, color, scene.surface_map == code, 0.6)
            ax.imshow(over)
            handles = [Rectangle((0, 0), 1, 1, color=SURFACE_COLORS[c]) for c in SURFACE_COLORS]
            labels = [
                f"{SURFACE_NAMES[c]} {_fmt_pct(row[f'{SURFACE_NAMES[c].lower()}_share'])}"
                for c in SURFACE_COLORS
            ]
            ax.legend(handles, labels, loc="lower left", fontsize=8, frameon=True, framealpha=0.9)
            _clean(ax, "Surfaces")
        else:
            ax.imshow(base)
            _clean(ax, "Surfaces · not computed")

        ax = fig.add_subplot(grid[1, 0])
        ax.imshow(photo)
        for det in scene.detections:
            x0, y0, x1, y1 = det.box
            color = SERIES[0] if det.label == "person" else SERIES[1]
            ax.add_patch(
                Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=1.6)
            )
        n_people = _fmt_int(row["person"])
        vehicles = row[list(self.emb.block("vehicles").columns)]
        n_veh = _fmt_int(float(vehicles.sum()) if bool(vehicles.notna().all()) else float("nan"))
        handles = [
            Rectangle((0, 0), 1, 1, fill=False, edgecolor=c, linewidth=1.6) for c in SERIES[:2]
        ]
        ax.legend(
            handles,
            [f"People {n_people}", f"Vehicles {n_veh}"],
            loc="lower left",
            fontsize=8,
            framealpha=0.9,
        )
        _clean(ax, "Detections")

        ax = fig.add_subplot(grid[1, 1])
        if scene.depth is not None:
            ax.imshow(scene.depth, cmap=BLUES, vmin=0, vmax=1)
            _clean(ax, "Depth weight (darker = closer)")
        else:
            ax.imshow(photo)
            _clean(ax, "Depth · not computed")

        ax = fig.add_subplot(grid[1, 2])
        self._stats(ax, image)

        self.palette(image, ax=fig.add_subplot(grid[2, :]))
        return fig

    def _stats(self, ax: Axes, image: int | str) -> None:
        s = self.emb.summary().iloc[self.emb._row(image)]
        tiles = [
            ("Greenery", _fmt_pct(s["greenery"])),
            ("People", _fmt_int(s["pedestrians"])),
            ("Vehicles", _fmt_int(s["vehicles"])),
            ("Road", _fmt_pct(s["road"])),
            ("Sidewalk", _fmt_pct(s["sidewalk"])),
            ("Building", _fmt_pct(s["building"])),
            ("Brightness", _fmt_pct(s["brightness"])),
            ("Colourfulness", _fmt_pct(s["colourfulness"])),
        ]
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 4)
        for i, (label, value) in enumerate(tiles):
            x, y = i % 2, 3 - i // 2
            ax.add_patch(
                Rectangle((x + 0.04, y + 0.06), 0.92, 0.88, facecolor="#f1f0ec", edgecolor="none")
            )
            ax.text(x + 0.12, y + 0.62, label, fontsize=8.5, color=TEXT_2, va="center")
            ax.text(
                x + 0.12,
                y + 0.3,
                value,
                fontsize=15,
                color=TEXT,
                va="center",
                fontweight="bold",
            )
        _clean(ax, "Key numbers")

    # ----------------------------------------------------------------------- comparisons
    def radar(
        self, images: Sequence[int | str] | None = None, normalize: NormalizeMethod = "batch"
    ) -> Figure:
        """Eight interpretable scalars per image on a radar chart.

        Up to three images are overlaid; more are drawn as small multiples. Values are scaled
        with ``normalize`` over all images in the set (``"batch"``: 1 = the set's maximum).
        """
        summary = self.emb.summary()
        scaled = _scale_summary(summary, normalize)
        picks = list(range(len(self.emb))) if images is None else [self.emb._row(i) for i in images]
        angles = np.linspace(0, 2 * np.pi, len(RADAR_AXES), endpoint=False)
        closed = np.concatenate([angles, angles[:1]])

        def draw(ax: Axes, rows: list[int], legend: bool) -> None:
            for slot, r in enumerate(rows):
                vals = np.nan_to_num(scaled.iloc[r][list(RADAR_AXES)].to_numpy(dtype=float))
                vals = np.concatenate([vals, vals[:1]])
                color = SERIES[slot]
                ax.plot(closed, vals, color=color, linewidth=2, label=self.emb.names[r])
                ax.fill(closed, vals, color=color, alpha=0.12)
            ax.set_xticks(angles)
            ax.set_xticklabels([a.title() for a in RADAR_AXES], fontsize=8, color=TEXT_2)
            ax.tick_params(axis="x", pad=10)
            ax.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax.set_yticklabels([])
            ax.set_ylim(0, 1)
            ax.grid(color=GRID, linewidth=0.8)
            ax.spines["polar"].set_color(GRID)
            if legend:
                ax.legend(loc="upper left", bbox_to_anchor=(1.08, 1.0), fontsize=8, frameon=False)

        if len(picks) <= 3:
            fig = _figure(6.4, 5)
            ax = fig.add_subplot(projection="polar")
            draw(ax, picks, legend=True)
            return fig
        cols = min(4, len(picks))
        rows = int(np.ceil(len(picks) / cols))
        fig = _figure(3.2 * cols, 3.3 * rows)
        for k, r in enumerate(picks):
            ax = fig.add_subplot(rows, cols, k + 1, projection="polar")
            draw(ax, [r], legend=False)
            ax.set_title(self.emb.names[r], fontsize=9, color=TEXT, pad=14)
        return fig

    def heatmap(
        self,
        *,
        normalize: NormalizeMethod = "batch",
        blocks: Sequence[str] | None = None,
        thumbnails: bool = True,
        annotate: bool | None = None,
    ) -> Figure:
        """Pairwise cosine similarity as a heatmap, with image thumbnails on both axes."""
        sim = self.emb.similarity(normalize=normalize, blocks=blocks)
        n = len(sim)
        size = max(4.5, min(13.0, 0.75 * n + 2.5))
        fig = _figure(size + 1.2, size)
        ax = fig.add_subplot()
        lo = float(np.floor(min(sim.to_numpy().min(), 0.9) * 10) / 10)
        im = ax.imshow(sim.to_numpy(), cmap=BLUES, vmin=lo, vmax=1.0)
        names = [str(c) for c in sim.columns]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8, color=TEXT_2)
        ax.set_yticklabels(names, fontsize=8, color=TEXT_2)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks(np.arange(-0.5, n), minor=True)
        ax.set_yticks(np.arange(-0.5, n), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="both", length=0)
        if annotate is None:
            annotate = n <= 12
        if annotate:
            values = sim.to_numpy()
            for i in range(n):
                for j in range(n):
                    v = values[i, j]
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color=_text_on((v - lo) / (1 - lo) if lo < 1 else 1.0),
                    )
        if thumbnails and self.emb.scenes is not None and n <= 25:
            # Thumbnails sit under the columns; row names stay readable on the left.
            longest = max(max(s.preview.shape[:2]) for s in self.emb.scenes)
            zoom = 54 / longest
            for k, name in enumerate(names):
                thumb = self.emb.scene(name).preview
                ab = AnnotationBbox(
                    OffsetImage(thumb, zoom=zoom),
                    (k, n - 0.5),
                    xybox=(0, -26),
                    xycoords="data",
                    boxcoords="offset points",
                    frameon=False,
                    annotation_clip=False,
                )
                ax.add_artist(ab)
            ax.tick_params(axis="x", pad=50)
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        outline: Any = cbar.outline  # a Spine; typed differently across matplotlib versions
        outline.set_visible(False)
        cbar.ax.tick_params(labelsize=8, colors=TEXT_2)
        cbar.set_label("Cosine similarity", fontsize=9, color=TEXT_2)
        ax.set_title("Pairwise similarity", loc="left", fontsize=11, color=TEXT, fontweight="bold")
        return fig

    def map(
        self,
        *,
        normalize: NormalizeMethod = "batch",
        blocks: Sequence[str] | None = None,
        thumbnails: bool = True,
        thumb_size: float = 64,
    ) -> Figure:
        """Images placed in 2-D by PCA of their normalised embeddings (close = similar)."""
        x = self.emb._matrix(normalize, blocks).to_numpy()
        names = self.emb.names
        coords, explained = _pca2(x)
        fig = _figure(9, 7)
        ax = fig.add_subplot()
        ax.scatter(coords[:, 0], coords[:, 1], s=60, color=SERIES[0], zorder=2)
        if thumbnails and self.emb.scenes is not None:
            for (cx, cy), name in zip(coords, names, strict=True):
                thumb = self.emb.scene(name).preview
                zoom = thumb_size / max(thumb.shape[:2])
                ax.add_artist(
                    AnnotationBbox(
                        OffsetImage(thumb, zoom=zoom),
                        (cx, cy),
                        frameon=True,
                        bboxprops={"edgecolor": SURFACE, "linewidth": 2},
                        pad=0.05,
                    )
                )
        for (cx, cy), name in zip(coords, names, strict=True):
            ax.annotate(
                name,
                (cx, cy),
                xytext=(0, -thumb_size * 0.62 if thumbnails else -10),
                textcoords="offset points",
                ha="center",
                va="top",
                fontsize=8,
                color=TEXT_2,
            )
        pad_x = (np.ptp(coords[:, 0]) or 1) * 0.15
        pad_y = (np.ptp(coords[:, 1]) or 1) * 0.15
        ax.set_xlim(coords[:, 0].min() - pad_x, coords[:, 0].max() + pad_x)
        ax.set_ylim(coords[:, 1].min() - pad_y, coords[:, 1].max() + pad_y)
        ax.set_xlabel(f"Component 1 ({explained[0]:.0%} of variance)", fontsize=9, color=TEXT_2)
        ax.set_ylabel(f"Component 2 ({explained[1]:.0%} of variance)", fontsize=9, color=TEXT_2)
        ax.grid(color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("Similarity map", loc="left", fontsize=11, color=TEXT, fontweight="bold")
        return fig

    def neighbors(
        self,
        image: int | str = 0,
        k: int = 4,
        *,
        normalize: NormalizeMethod = "batch",
        blocks: Sequence[str] | None = None,
    ) -> Figure:
        """An image followed by its ``k`` most similar images, with cosine similarity."""
        near = self.emb.nearest(image, k, normalize=normalize, blocks=blocks)
        query = self.emb.names[self.emb._row(image)]
        cols = len(near) + 1
        aspect = float(
            np.mean([s.preview.shape[0] / s.preview.shape[1] for s in self.emb.scenes or []])
            or 0.75
        )
        fig = _figure(3.0 * cols, 3.0 * aspect + 0.7)
        for j, (name, score) in enumerate([(query, None), *near.items()]):
            ax = fig.add_subplot(1, cols, j + 1)
            ax.imshow(self.emb.scene(str(name)).preview)
            title = f"{name}\nquery" if score is None else f"{name}\nsimilarity {score:.3f}"
            _clean(ax)
            ax.set_title(title, fontsize=9, color=TEXT if score is None else TEXT_2)
            if score is None:
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color(SERIES[0])
                    spine.set_linewidth(3)
        return fig


def _scale_summary(summary: Any, normalize: NormalizeMethod) -> Any:
    out = summary.astype(float).copy()
    if normalize == "none":
        return out
    for col in out.columns:
        if (
            col in ("greenery", "road", "sidewalk", "building", "brightness", "colourfulness")
            and normalize == "fixed"
        ):
            continue
        top = out[col].abs().max()
        if normalize == "fixed" and col in ("pedestrians", "vehicles"):
            top = 50.0 if col == "pedestrians" else 80.0
        if top and not np.isnan(top):
            out[col] = (out[col] / top).clip(0, 1)
    return out


def _pca2(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """First two principal components (padded with zeros for tiny inputs)."""
    centered = x - x.mean(axis=0)
    if x.shape[0] < 2 or not np.any(centered):
        return np.zeros((x.shape[0], 2)), np.zeros(2)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    coords = u * s
    var = s**2
    explained = var / var.sum() if var.sum() > 0 else np.zeros_like(var)
    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((coords.shape[0], 2 - coords.shape[1]))])
        explained = np.concatenate([explained, np.zeros(2 - explained.size)])
    return coords[:, :2], explained[:2]
