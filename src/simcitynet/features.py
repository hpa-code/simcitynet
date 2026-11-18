"""Names, blocks and descriptions of the 33 SimCityNet dimensions.

The order follows Section 4 of Caliskan et al. (2025):

``Φ(I) = [f_veg, f_color, c_ped, d_¬ped, u]`` with sizes ``1 + 20 + 1 + 8 + 3 = 33``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: COCO classes counted in the "other dynamic objects" block, in paper order.
VEHICLE_CLASSES: tuple[str, ...] = (
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
)

#: Surface classes measured as pixel proportions, in paper order.
SURFACE_CLASSES: tuple[str, ...] = ("road", "sidewalk", "building")

#: Vegetation classes (ADE20K) merged into the depth-weighted Green View Index.
VEGETATION_CLASSES: tuple[str, ...] = ("tree", "grass", "plant")

N_COLORS = 5


def _color_names() -> list[str]:
    names = [f"color{k}_{ch}" for k in range(1, N_COLORS + 1) for ch in ("r", "g", "b")]
    names += [f"color{k}_share" for k in range(1, N_COLORS + 1)]
    return names


#: The 33 feature names, in embedding order.
FEATURE_NAMES: tuple[str, ...] = (
    "vegetation",
    *_color_names(),
    "person",
    *VEHICLE_CLASSES,
    *(f"{c}_share" for c in SURFACE_CLASSES),
)

#: Blocks of the embedding, mapping block name -> feature names.
BLOCKS: dict[str, tuple[str, ...]] = {
    "vegetation": ("vegetation",),
    "color": tuple(_color_names()),
    "pedestrians": ("person",),
    "vehicles": VEHICLE_CLASSES,
    "surfaces": tuple(f"{c}_share" for c in SURFACE_CLASSES),
}

EMBEDDING_DIM = len(FEATURE_NAMES)
assert EMBEDDING_DIM == 33, EMBEDDING_DIM


@dataclass(frozen=True)
class FeatureInfo:
    """Human-readable description of one embedding dimension."""

    name: str
    block: str
    description: str
    unit: str


def _describe(name: str) -> FeatureInfo:
    if name == "vegetation":
        return FeatureInfo(
            name,
            "vegetation",
            "Depth-weighted Green View Index over trees, grass and plants",
            "fraction [0, 1]",
        )
    if name.startswith("color"):
        k, part = name[len("color") :].split("_")
        if part == "share":
            return FeatureInfo(
                name,
                "color",
                f"Share of pixels in dominant colour {k} (colours sorted by share)",
                "fraction [0, 1]",
            )
        channel = {"r": "red", "g": "green", "b": "blue"}[part]
        return FeatureInfo(
            name, "color", f"{channel.title()} channel of dominant colour {k}", "0-255"
        )
    if name == "person":
        return FeatureInfo(name, "pedestrians", "Number of detected people", "count")
    if name in VEHICLE_CLASSES:
        return FeatureInfo(
            name, "vehicles", f"Number of detected objects of class '{name}'", "count"
        )
    surface = name.removesuffix("_share")
    return FeatureInfo(name, "surfaces", f"Share of pixels labelled '{surface}'", "fraction [0, 1]")


#: Descriptions for all 33 features, in embedding order.
FEATURES: tuple[FeatureInfo, ...] = tuple(_describe(n) for n in FEATURE_NAMES)
