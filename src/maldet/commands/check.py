"""`maldet check` — validate maldet.toml and entrypoint references."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Annotated

import typer

from maldet.manifest import ManifestNotFoundError, load_manifest, search_manifest


def _check_symbol(dotted: str) -> str | None:
    if ":" not in dotted:
        return f"{dotted!r}: expected 'module:attribute' format"
    mod, attr = dotted.split(":", 1)
    try:
        m = importlib.import_module(mod)
    except ImportError as exc:
        return f"{dotted!r}: cannot import module '{mod}': {exc}"
    if not hasattr(m, attr):
        return f"{dotted!r}: module '{mod}' has no attribute '{attr}'"
    return None


def check(
    config: Annotated[
        Path | None, typer.Option("--config", "-c", help="Optional Hydra config to validate too")
    ] = None,
) -> None:
    """Validate the detector's manifest and optional Hydra config."""
    try:
        manifest_path = search_manifest()
    except ManifestNotFoundError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1) from exc

    try:
        manifest = load_manifest(manifest_path)
    except Exception as exc:
        typer.echo(f"ERROR: manifest invalid: {exc}", err=True)
        raise typer.Exit(1) from exc

    errors: list[str] = []
    for stage_name, spec in manifest.stages.items():
        for field in ("reader", "extractor", "model", "trainer", "evaluator", "predictor"):
            dotted = getattr(spec, field, None)
            if dotted is None:
                continue
            err = _check_symbol(dotted)
            if err is not None:
                errors.append(f"[stages.{stage_name}.{field}] {err}")

    if errors:
        for e in errors:
            typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("OK")
