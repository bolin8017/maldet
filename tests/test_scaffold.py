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


def test_scaffold_rf_dockerfile_copies_readme(tmp_path: Path) -> None:
    """Phase 11d regression: detectors that declare readme=README.md in pyproject need
    README.md present at wheel-build time, or hatchling raises "Readme file does not exist"."""
    target = tmp_path / "mydet"
    r = runner.invoke(
        app, ["scaffold", "--template", "rf", "--name", "mydet", "--out", str(target)]
    )
    assert r.exit_code == 0, r.stdout
    dockerfile = (target / "Dockerfile").read_text()
    assert "README.md" in dockerfile


def test_scaffold_cnn_dockerfile_copies_readme(tmp_path: Path) -> None:
    target = tmp_path / "mynn"
    r = runner.invoke(
        app, ["scaffold", "--template", "cnn", "--name", "mynn", "--out", str(target)]
    )
    assert r.exit_code == 0, r.stdout
    dockerfile = (target / "Dockerfile").read_text()
    assert "README.md" in dockerfile
