"""Per-dimension scaling ``Φ̃(I) = N Φ(I)`` (paper Section 4).

* ``"batch"`` — the paper's rule ``n_k = 1 / max_I Φ_k(I)`` over the images being compared.
* ``"fixed"`` — the paper's "(or a chosen upper bound)": natural bounds for colours (255) and
  shares (1), configurable caps for counts. Comparable across separate runs. Values are
  clipped to [0, 1].
* ``"none"`` — raw values.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd

from .config import Config
from .features import BLOCKS, FEATURE_NAMES

NormalizeMethod = Literal["batch", "fixed", "none"]


def fixed_bounds(count_caps: Mapping[str, float] | None = None) -> pd.Series:
    """Upper bound used by ``normalize="fixed"`` for each of the 33 features."""
    caps = dict(Config().count_caps if count_caps is None else count_caps)
    bounds: dict[str, float] = {}
    for name in FEATURE_NAMES:
        if name in BLOCKS["color"]:
            bounds[name] = 1.0 if name.endswith("_share") else 255.0
        elif name in BLOCKS["pedestrians"] or name in BLOCKS["vehicles"]:
            bounds[name] = float(caps.get(name, 10.0))
        else:  # vegetation and surface shares are already fractions
            bounds[name] = 1.0
    return pd.Series(bounds, dtype=float)


def normalize(
    raw: pd.DataFrame,
    method: NormalizeMethod = "batch",
    count_caps: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Scale an ``(n, 33)`` embedding table. Missing values stay missing."""
    if method == "none":
        return raw.astype(float).copy()
    if method == "batch":
        maxima = raw.abs().max(axis=0, skipna=True).astype(float)
        maxima = maxima.where(maxima > 0, 1.0).fillna(1.0)
        return raw.astype(float).div(maxima, axis=1)
    if method == "fixed":
        bounds = fixed_bounds(count_caps).reindex(raw.columns).fillna(1.0)
        out = raw.astype(float).div(bounds, axis=1)
        return out.clip(lower=0.0, upper=1.0)
    raise ValueError(f"Unknown normalize method {method!r}; use 'batch', 'fixed' or 'none'")


def complete_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns with any missing value (e.g. blocks that were not computed)."""
    keep = [c for c in df.columns if not bool(np.any(df[c].isna()))]
    return df[keep]
