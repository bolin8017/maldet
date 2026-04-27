"""`maldet check` — validate maldet.toml and entrypoint references."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel

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


def _check_stage_config_class_strict(stage_name: str, dotted: str) -> list[str]:
    """Return list of error strings; empty if OK.

    Validates that ``stage_name``'s ``config_class`` points at a Pydantic
    ``BaseModel`` subclass with ``model_config['extra'] == 'forbid'``.

    Import errors are reported by ``_check_symbol`` (called separately on the
    same string), so this helper stays silent on those — the symbol-loop owner
    will already have surfaced an error for the missing module / attribute.
    """
    if ":" not in dotted:
        # Format error already covered by _check_symbol; stay silent here.
        return []
    mod_name, attr = dotted.split(":", 1)
    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        # Import error already reported by _check_symbol.
        return []
    cls = getattr(mod, attr, None)
    if cls is None:
        # Missing attribute already reported by _check_symbol.
        return []
    errors: list[str] = []
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        errors.append(
            f"[stages.{stage_name}.config_class] {dotted} is not a pydantic.BaseModel subclass"
        )
        return errors
    current = cls.model_config.get("extra")
    if current != "forbid":
        hint = (
            "missing — add model_config = ConfigDict(extra='forbid')"
            if current is None
            else f"got {current!r}"
        )
        errors.append(
            f"[stages.{stage_name}.config_class] {dotted}: model_config['extra'] must be 'forbid' ({hint})"
        )
    return errors


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

        # Phase 11e — strict lint on stage's typed config_class.
        # _check_symbol covers import / attribute errors; this adds Pydantic
        # subclass + extra='forbid' enforcement.
        sym_err = _check_symbol(spec.config_class)
        if sym_err is not None:
            errors.append(f"[stages.{stage_name}.config_class] {sym_err}")
        errors.extend(_check_stage_config_class_strict(stage_name, spec.config_class))

    if errors:
        for e in errors:
            typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("OK")
