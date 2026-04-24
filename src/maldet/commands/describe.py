"""`maldet describe` — print the manifest as JSON or TOML."""

from __future__ import annotations

import json
from typing import Annotated

import typer

from maldet.manifest import load_manifest, search_manifest


def describe(
    format: Annotated[
        str, typer.Option("--format", "-f", help="Output format: json or toml")
    ] = "json",
) -> None:
    m = load_manifest(search_manifest())
    if format == "json":
        typer.echo(json.dumps(m.model_dump(mode="json"), indent=2, default=str))
    elif format == "toml":
        try:
            import tomli_w
        except ImportError:
            raise typer.BadParameter("TOML output requires: pip install tomli_w") from None
        typer.echo(tomli_w.dumps(m.model_dump(mode="json")))
    else:
        raise typer.BadParameter(f"unknown format: {format}")
