"""``maldet introspect-schema`` — auto-derive a stage's JSON Schema from its Pydantic config class.

Used by ``maldet build`` to populate ``manifest.stages.{stage}.params_schema``
without forcing detector authors to hand-write JSON Schema.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel


def _load_class(dotted: str) -> object:
    if ":" not in dotted:
        raise typer.BadParameter(f"expected 'module.sub:Class', got {dotted!r}")
    mod_name, attr = dotted.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def introspect_schema(
    config_class: Annotated[
        str,
        typer.Option(
            "--config-class",
            help="Import path 'module.sub:ClassName' to a Pydantic BaseModel.",
        ),
    ],
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write schema JSON to this file (else stdout)."),
    ] = None,
) -> None:
    """Auto-derive JSON Schema from a stage's Pydantic config class."""
    cls = _load_class(config_class)
    if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
        typer.echo(f"error: {config_class} is not a pydantic.BaseModel subclass", err=True)
        raise typer.Exit(2)
    if cls.model_config.get("extra") != "forbid":
        typer.echo(
            f"error: {config_class} must set model_config = ConfigDict(extra='forbid')",
            err=True,
        )
        raise typer.Exit(2)
    schema: dict[str, Any] = cls.model_json_schema(mode="serialization")
    text = json.dumps(schema, indent=2, sort_keys=True)
    if out is not None:
        out.write_text(text + "\n")
    else:
        sys.stdout.write(text + "\n")
