"""Record what produced an embedding (package, model and library versions, settings, device)."""

from __future__ import annotations

import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from ._version import __version__

_LIBRARIES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "opencv-python-headless",
    "Pillow",
    "torch",
    "transformers",
    "ultralytics",
)


def library_versions() -> dict[str, str | None]:
    """Installed versions of the libraries that influence results (``None`` if absent)."""
    versions: dict[str, str | None] = {}
    for lib in _LIBRARIES:
        try:
            versions[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            versions[lib] = None
    return versions


def collect(config: dict[str, Any], device: str | None, blocks: list[str]) -> dict[str, Any]:
    """Build a JSON-serialisable provenance record."""
    return {
        "simcitynet_version": __version__,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "device": device,
        "blocks": blocks,
        "config": config,
        "libraries": library_versions(),
        "paper": "Caliskan et al. (2025), SimCityNet, SSRN 5369073, doi:10.2139/ssrn.5369073",
    }
