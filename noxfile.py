#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox>=2025.2.9"]
# ///
"""Task runner: `pip install nox` then `nox` (default sessions) or `nox -s tests`."""

from __future__ import annotations

import nox

nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "uv|virtualenv"

PYTHONS = ["3.10", "3.11", "3.12", "3.13"]


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
    """Fast tests with fake models and coverage."""
    session.install("-e", ".[dev]")
    session.run("pytest", "--cov", *session.posargs)


@nox.session(default=False)
def slow(session: nox.Session) -> None:
    """Real-model tests (downloads ~3 GB of weights and the example photos)."""
    session.install("-e", ".[all,dev]")
    session.run("pytest", "-m", "slow", *session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
    """Formatting, linting and type checks."""
    session.install("-e", ".[dev]")
    session.run("ruff", "format", "--check", "src", "tests")
    session.run("ruff", "check", "src", "tests")
    session.run("mypy")


@nox.session
def docs(session: nox.Session) -> None:
    """Build the documentation; `nox -s docs -- serve` for a live preview."""
    session.install("-e", ".[docs]")
    if session.posargs and session.posargs[0] == "serve":
        session.run("mkdocs", "serve")
    else:
        session.run("mkdocs", "build", "--strict")


@nox.session(default=False)
def build(session: nox.Session) -> None:
    """Build and check the sdist and wheel."""
    session.install("build", "twine")
    session.run("python", "-m", "build")
    session.run("twine", "check", "--strict", "dist/*")


if __name__ == "__main__":
    nox.main()
