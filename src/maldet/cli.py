"""maldet CLI — root Typer app.

Subcommands live in ``maldet.commands.*``.
"""

from __future__ import annotations

import typer

from maldet._version import __version__
from maldet.commands import check as _check
from maldet.commands import describe as _describe
from maldet.commands import introspect_schema as _introspect
from maldet.commands import run as _run
from maldet.commands import scaffold as _scaffold

app = typer.Typer(
    name="maldet",
    help="Plug-and-play malware detector framework",
    no_args_is_help=True,
    add_completion=False,
)

app.add_typer(_run.app, name="run")
app.command("describe")(_describe.describe)
app.command("check")(_check.check)
app.command("scaffold")(_scaffold.scaffold)
app.command("introspect-schema")(_introspect.introspect_schema)


@app.callback(invoke_without_command=True)
def _root(
    version: bool = typer.Option(False, "--version", help="Print version and exit"),
) -> None:
    if version:
        typer.echo(__version__)
        raise typer.Exit(0)
