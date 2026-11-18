"""The five measurement blocks of the SimCityNet embedding."""

from .color import ColorResult, dominant_colors
from .dynamics import COUNTED_CLASSES, DynamicsResult, count_objects
from .surfaces import surface_shares
from .vegetation import VegetationResult, compute_vegetation, depth_weighted_index, normalize_depth

__all__ = [
    "COUNTED_CLASSES",
    "ColorResult",
    "DynamicsResult",
    "VegetationResult",
    "compute_vegetation",
    "count_objects",
    "depth_weighted_index",
    "dominant_colors",
    "normalize_depth",
    "surface_shares",
]
