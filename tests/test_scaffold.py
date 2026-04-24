"""maldet scaffold generates a working detector repo."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from maldet.cli import app

runner = CliRunner()


def test_scaffold_rf_produces_maldet_toml(tmp_path: Path) -> None:
    target = tmp_path / "mydet"
    r = runner.invoke(
        app, ["scaffold", "--template", "rf", "--name", "mydet", "--out", str(target)]
    )
    assert r.exit_code == 0, r.stdout
    assert (target / "maldet.toml").exists()
    assert (target / "pyproject.toml").exists()
    assert (target / "Dockerfile").exists()
    assert (target / "src" / "mydet" / "__init__.py").exists()
    assert (target / "src" / "mydet" / "features.py").exists()
    assert (target / "src" / "mydet" / "models.py").exists()

    content = (target / "maldet.toml").read_text()
    assert 'name = "mydet"' in content
    assert 'framework = "sklearn"' in content


def test_scaffold_cnn_declares_lightning(tmp_path: Path) -> None:
    target = tmp_path / "mynn"
    r = runner.invoke(
        app, ["scaffold", "--template", "cnn", "--name", "mynn", "--out", str(target)]
    )
    assert r.exit_code == 0
    content = (target / "maldet.toml").read_text()
    assert 'framework = "lightning"' in content
    assert 'supports_distributed = "ddp"' in content
