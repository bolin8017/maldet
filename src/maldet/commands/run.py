"""`maldet run <stage>` — drives a stage via StageRunner."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from maldet.runner import StageRunner

app = typer.Typer(help="Run a detector lifecycle stage")


def _run_stage(
    stage: str,
    config: Annotated[Path, typer.Option("--config", "-c", help="Path to Hydra YAML config")],
) -> None:
    StageRunner().run(stage=stage, config_path=config)


@app.command("train")
def train(config: Annotated[Path, typer.Option("--config", "-c")]) -> None:
    """Train the detector."""
    _run_stage("train", config)


@app.command("evaluate")
def evaluate(config: Annotated[Path, typer.Option("--config", "-c")]) -> None:
    """Evaluate the detector against test data."""
    _run_stage("evaluate", config)


@app.command("predict")
def predict(config: Annotated[Path, typer.Option("--config", "-c")]) -> None:
    """Run prediction on input samples."""
    _run_stage("predict", config)
