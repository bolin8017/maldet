"""CLI integration (in-process)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from maldet.cli import app

runner = CliRunner()
FIX = Path(__file__).parent / "fixtures"


def test_version() -> None:
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0
    assert r.stdout.strip()


def test_describe_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = tmp_path / "maldet.toml"
    m.write_text((FIX / "valid_manifest.toml").read_text())
    monkeypatch.setenv("MALDET_MANIFEST", str(m))
    r = runner.invoke(app, ["describe", "--format", "json"])
    assert r.exit_code == 0
    data = json.loads(r.stdout)
    assert data["detector"]["name"] == "elfrfdet"


def test_check_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = tmp_path / "maldet.toml"
    m.write_text((FIX / "valid_manifest.toml").read_text())
    monkeypatch.setenv("MALDET_MANIFEST", str(m))
    r = runner.invoke(app, ["check"])
    assert r.exit_code == 0
    assert "OK" in r.stdout


def test_check_fails_on_broken_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    m = tmp_path / "maldet.toml"
    m.write_text('[detector]\nname = ""\n')
    monkeypatch.setenv("MALDET_MANIFEST", str(m))
    r = runner.invoke(app, ["check"])
    assert r.exit_code != 0
