"""`maldet scaffold` — generate a new detector repo from a Jinja2 template."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATES_ROOT = Path(__file__).parent.parent / "templates"


def scaffold(
    template: Annotated[str, typer.Option("--template", "-t", help="rf | cnn")] = "rf",
    name: Annotated[str, typer.Option("--name", "-n", help="Package / detector name")] = "",
    out: Annotated[Path, typer.Option("--out", "-o", help="Output directory")] = Path("."),
) -> None:
    """Generate a fresh detector repo under ``out/``."""
    if not name:
        raise typer.BadParameter("--name is required")
    tpl_dir = TEMPLATES_ROOT / template
    if not tpl_dir.is_dir():
        raise typer.BadParameter(f"unknown template: {template}")
    env = Environment(
        loader=FileSystemLoader(str(tpl_dir)),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
        autoescape=False,  # we're generating Python/TOML/Dockerfile, not HTML
    )
    out.mkdir(parents=True, exist_ok=True)

    for src in tpl_dir.rglob("*.j2"):
        rel = src.relative_to(tpl_dir)
        parts = list(rel.parts)
        if parts and parts[0] == "src":
            parts = ["src", name, *parts[1:]]
        target = out.joinpath(*parts).with_suffix("")  # strip .j2
        target.parent.mkdir(parents=True, exist_ok=True)
        rendered = env.get_template(str(rel)).render(name=name)
        target.write_text(rendered, encoding="utf-8")

    typer.echo(f"Scaffolded {template} detector to {out}")
