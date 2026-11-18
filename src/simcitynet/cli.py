"""Command-line interface: ``simcitynet embed | similarity | card | features | doctor``."""

from __future__ import annotations

import importlib.util
import logging
import platform
import sys
from pathlib import Path
from typing import Annotated

import typer

from ._version import __version__
from .config import Config
from .embedder import ALL_BLOCKS, SimCityNet
from .embeddings import Embeddings
from .features import FEATURES

app = typer.Typer(
    name="simcitynet",
    help="Interpretable 33-D embeddings and similarity for street-level images.",
    no_args_is_help=True,
    add_completion=False,
)

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"simcitynet {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Show version."),
    ] = False,
) -> None:
    """SimCityNet command-line tools."""


def _blocks(value: str | None) -> tuple[str, ...]:
    if not value:
        return ALL_BLOCKS
    return tuple(b.strip() for b in value.split(",") if b.strip())


@app.command()
def embed(
    inputs: Annotated[list[Path], typer.Argument(help="Images, folders or videos.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output CSV path.")] = Path(
        "embeddings.csv"
    ),
    device: Annotated[str, typer.Option(help="auto, cpu, cuda or mps.")] = "auto",
    blocks: Annotated[
        str | None, typer.Option(help="Comma-separated subset, e.g. 'color,surfaces'.")
    ] = None,
    panorama: Annotated[bool, typer.Option(help="Treat inputs as 2:1 panoramas.")] = False,
    every: Annotated[float, typer.Option(help="Seconds between sampled video frames.")] = 1.0,
) -> None:
    """Compute embeddings and write them to a CSV (+ a JSON provenance file)."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    scn = SimCityNet(Config(device=device, panorama=panorama), blocks=_blocks(blocks))  # type: ignore[arg-type]
    parts: list[Embeddings] = []
    images: list[Path] = []
    for path in inputs:
        if path.suffix.lower() in _VIDEO_SUFFIXES:
            parts.append(scn.embed_video(path, every=every, keep_scenes=False))
        else:
            images.append(path)
    if images:
        parts.append(scn.embed(images, keep_scenes=False, progress=True))
    if not parts:
        raise typer.BadParameter("No inputs given")
    import pandas as pd

    raw = pd.concat([p.raw for p in parts])
    meta = pd.concat([p.meta for p in parts])
    result = Embeddings(raw, meta=meta, provenance=parts[0].provenance)
    path = result.save(output)
    typer.echo(f"Wrote {len(result)} embeddings to {path}")


@app.command()
def similarity(
    embeddings: Annotated[Path, typer.Argument(help="CSV written by `simcitynet embed`.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="Output CSV path.")] = Path(
        "similarity.csv"
    ),
    metric: Annotated[str, typer.Option(help="cosine, euclidean or color_emd.")] = "cosine",
    normalize: Annotated[str, typer.Option(help="batch, fixed or none.")] = "batch",
) -> None:
    """Compute a pairwise similarity (cosine) or distance matrix."""
    emb = Embeddings.load(embeddings)
    if metric == "cosine":
        table = emb.similarity(normalize=normalize)  # type: ignore[arg-type]
    elif metric in ("euclidean", "color_emd"):
        table = emb.distance(metric, normalize=normalize)  # type: ignore[arg-type]
    else:
        raise typer.BadParameter("metric must be cosine, euclidean or color_emd")
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output)
    typer.echo(f"Wrote {table.shape[0]}x{table.shape[1]} {metric} matrix to {output}")


@app.command()
def card(
    image: Annotated[Path, typer.Argument(help="One image.")],
    output: Annotated[Path, typer.Option("--output", "-o", help="PNG path.")] = Path("card.png"),
    device: Annotated[str, typer.Option(help="auto, cpu, cuda or mps.")] = "auto",
) -> None:
    """Save a scene card (overlays, palette, key numbers) for one image."""
    emb = SimCityNet(Config(device=device)).embed(image)  # type: ignore[arg-type]
    fig = emb.plot.scene_card(0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    typer.echo(f"Saved {output}")


@app.command()
def features() -> None:
    """List the 33 embedding dimensions."""
    for i, f in enumerate(FEATURES, start=1):
        typer.echo(f"{i:>2}  {f.name:<16} {f.block:<12} {f.unit:<16} {f.description}")


@app.command()
def doctor() -> None:
    """Check the installation: Python, optional extras and GPU."""
    typer.echo(f"simcitynet {__version__}")
    typer.echo(f"Python     {sys.version.split()[0]} on {platform.platform()}")
    ok = True
    for module, extra, purpose in (
        ("torch", "models", "vegetation, depth and surface models"),
        ("transformers", "models", "vegetation, depth and surface models"),
        ("ultralytics", "yolo", "pedestrian and vehicle counts"),
    ):
        found = importlib.util.find_spec(module) is not None
        ok &= found
        mark = "ok     " if found else "missing"
        hint = "" if found else f'  -> pip install "simcitynet[{extra}]"'
        typer.echo(f"[{mark}] {module:<13} ({purpose}){hint}")
    if importlib.util.find_spec("torch") is not None:
        import torch

        if torch.cuda.is_available():
            typer.echo(f"[ok     ] CUDA GPU      {torch.cuda.get_device_name(0)}")
        else:
            typer.echo("[note   ] No CUDA GPU found; models will run on CPU (slow but works).")
    if not ok:
        typer.echo(
            'Only the colour block works without extras. Full install: pip install "simcitynet[all]"'
        )


def _run() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    _run()
