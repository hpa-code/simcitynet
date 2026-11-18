"""Openly licensed example street photos from Wikimedia Commons, for tutorials.

Images are downloaded on demand (never shipped with the package) together with their author
and licence, read from the Wikimedia Commons API. Keep the generated ``CREDITS.csv`` with any
figure you publish: most Commons photos require attribution and some are share-alike.
"""

from __future__ import annotations

import csv
import html
import json
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ._version import __version__

API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = f"simcitynet/{__version__} (https://github.com/hpa-code/simcitynet)"


@dataclass(frozen=True)
class Example:
    """One example photo on Wikimedia Commons."""

    name: str
    city: str
    region: str
    commons_file: str


#: Street scenes from cities studied in the paper (three in Europe, three in the Middle East).
EXAMPLES: tuple[Example, ...] = (
    Example("amsterdam", "Amsterdam", "Europe", "GerardDoustraat.jpg"),
    Example("dublin", "Dublin", "Europe", "Templebar-dublin-1.jpg"),
    Example("zurich", "Zürich", "Europe", "Niederdorfstrasse Zürich 2022-09-24 01.jpg"),
    Example("doha", "Doha", "Middle East", "Doha Souq Waqif Main Street 02.jpg"),
    Example("dubai", "Dubai", "Middle East", "Al Fahidi Historical Neighbourhood, Dubai.jpg"),
    Example(
        "tel_aviv",
        "Tel Aviv",
        "Middle East",
        "Visit to Rothschild Boulevard, Tel Aviv, Israel 26.jpg",
    ),
)


@dataclass(frozen=True)
class Credit:
    """Attribution for one downloaded example."""

    name: str
    city: str
    region: str
    author: str
    license: str
    license_url: str
    source_url: str


def _get_json(url: str) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data: dict[str, Any] = json.loads(resp.read().decode("utf-8"))
        return data


def _download(url: str, path: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:
        path.write_bytes(resp.read())


def _strip_html(text: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


def _lookup(commons_file: str, width: int) -> tuple[str, dict[str, str]]:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "titles": f"File:{commons_file}",
            "prop": "imageinfo",
            "iiprop": "url|extmetadata",
            "iiurlwidth": str(width),
            "format": "json",
        }
    )
    data = _get_json(f"{API}?{query}")
    page = next(iter(data["query"]["pages"].values()))
    if "imageinfo" not in page:
        raise FileNotFoundError(f"Wikimedia Commons has no file named {commons_file!r}")
    info = page["imageinfo"][0]
    meta = {k: str(v.get("value", "")) for k, v in info.get("extmetadata", {}).items()}
    meta["descriptionurl"] = info.get("descriptionurl", "")
    return info.get("thumburl") or info["url"], meta


def fetch_examples(
    dest: str | os.PathLike[str] = "simcitynet_examples",
    *,
    width: int = 1280,
    names: list[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download example street photos and write ``CREDITS.csv`` next to them.

    Args:
        dest: Folder to save into (created if needed).
        width: Width of the downloaded copy in pixels.
        names: Subset of :data:`EXAMPLES` names (default: all).
        overwrite: Download again even if the file exists.

    Returns:
        Paths of the downloaded JPEG files, named ``<name>.jpg``.
    """
    folder = Path(dest)
    folder.mkdir(parents=True, exist_ok=True)
    chosen = [e for e in EXAMPLES if names is None or e.name in names]
    if names is not None and len(chosen) != len(set(names)):
        known = sorted(e.name for e in EXAMPLES)
        raise ValueError(f"Unknown example name(s); choose from {known}")
    paths: list[Path] = []
    credits: list[Credit] = []
    for ex in chosen:
        url, meta = _lookup(ex.commons_file, width)
        path = folder / f"{ex.name}.jpg"
        if overwrite or not path.exists():
            _download(url, path)
        paths.append(path)
        credits.append(
            Credit(
                name=ex.name,
                city=ex.city,
                region=ex.region,
                author=_strip_html(meta.get("Artist", "")) or "unknown",
                license=_strip_html(meta.get("LicenseShortName", "")) or "see source",
                license_url=meta.get("LicenseUrl", ""),
                source_url=meta.get("descriptionurl", ""),
            )
        )
    with (folder / "CREDITS.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["name", "city", "region", "author", "license", "license_url", "source_url"]
        )
        for c in credits:
            writer.writerow(
                [c.name, c.city, c.region, c.author, c.license, c.license_url, c.source_url]
            )
    return paths


def credits_table(dest: str | os.PathLike[str] = "simcitynet_examples") -> list[Credit]:
    """Read back the credits written by :func:`fetch_examples`."""
    with (Path(dest) / "CREDITS.csv").open(encoding="utf-8") as fh:
        return [Credit(**row) for row in csv.DictReader(fh)]
