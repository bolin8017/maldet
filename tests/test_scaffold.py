"""maldet scaffold generates a working detector repo."""

from __future__ import annotations

from pathlib import Path

import pytest
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


def test_scaffold_rf_emits_pydantic_configs(tmp_path: Path) -> None:
    """Phase 11e: scaffolded sklearn (rf) detector ships a Pydantic configs.py with extra='forbid'."""
    target = tmp_path / "newdet"
    r = runner.invoke(
        app, ["scaffold", "--template", "rf", "--name", "newdet", "--out", str(target)]
    )
    assert r.exit_code == 0, r.stdout
    cfg = (target / "src" / "newdet" / "configs.py").read_text()
    assert "TrainConfig" in cfg
    assert "EvaluateConfig" in cfg
    assert "PredictConfig" in cfg
    assert "extra='forbid'" in cfg or 'extra="forbid"' in cfg


def test_scaffold_cnn_emits_pydantic_configs(tmp_path: Path) -> None:
    """Phase 11e: scaffolded lightning (cnn) detector ships a Pydantic configs.py with extra='forbid'."""
    target = tmp_path / "newnn"
    r = runner.invoke(
        app, ["scaffold", "--template", "cnn", "--name", "newnn", "--out", str(target)]
    )
    assert r.exit_code == 0, r.stdout
    cfg = (target / "src" / "newnn" / "configs.py").read_text()
    assert "TrainConfig" in cfg
    assert "epochs" in cfg
    assert "extra='forbid'" in cfg or 'extra="forbid"' in cfg


def test_scaffold_rf_check_passes_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Phase 11e: a freshly scaffolded rf detector survives `maldet check` (strict lint)."""
    target = tmp_path / "freshrf"
    r = runner.invoke(
        app, ["scaffold", "--template", "rf", "--name", "freshrf", "--out", str(target)]
    )
    assert r.exit_code == 0
    monkeypatch.syspath_prepend(str(target / "src"))  # so freshrf.* is importable
    monkeypatch.chdir(target)
    res = runner.invoke(app, ["check"])
    text = (res.stdout or "") + (res.stderr or "")
    # Other symbols (features.Text256Extractor, models.make_rf) must also resolve, which
    # they should because the templates ship those as stub modules. The strict lint
    # specifically verifies configs.py classes are BaseModel + extra='forbid'.
    assert res.exit_code == 0, text
