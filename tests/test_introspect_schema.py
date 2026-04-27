"""maldet introspect-schema — derive JSON Schema from a stage's config_class."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from maldet.cli import app


@pytest.fixture
def sample_pkg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Drop a minimal Pydantic config class on sys.path."""
    pkg = tmp_path / "samplepkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "configs.py").write_text(
        "from pydantic import BaseModel, ConfigDict, Field\n"
        "class TrainConfig(BaseModel):\n"
        "    model_config = ConfigDict(extra='forbid')\n"
        "    n_estimators: int = Field(default=100, ge=1)\n"
        "    max_depth: int | None = None\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    return pkg


def test_introspect_emits_json_schema_to_file(sample_pkg: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    out = tmp_path / "schema.json"
    result = runner.invoke(
        app,
        ["introspect-schema", "--config-class", "samplepkg.configs:TrainConfig", "--out", str(out)],
    )
    assert result.exit_code == 0, result.stdout
    schema = json.loads(out.read_text())
    assert schema["additionalProperties"] is False
    assert "n_estimators" in schema["properties"]
    assert schema["properties"]["n_estimators"]["minimum"] == 1


def test_introspect_emits_json_schema_to_stdout(sample_pkg: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["introspect-schema", "--config-class", "samplepkg.configs:TrainConfig"]
    )
    assert result.exit_code == 0
    schema = json.loads(result.stdout)
    assert "n_estimators" in schema["properties"]


def test_introspect_rejects_non_basemodel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "badpkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "x.py").write_text("class NotAModel:\n    pass\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(app, ["introspect-schema", "--config-class", "badpkg.x:NotAModel"])
    assert result.exit_code != 0
    # error message should mention BaseModel — check both stdout & stderr
    text = (result.stdout or "") + (result.stderr or "")
    assert "BaseModel" in text


def test_introspect_rejects_extra_allow(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "loose"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "c.py").write_text(
        "from pydantic import BaseModel, ConfigDict\n"
        "class LooseConfig(BaseModel):\n"
        "    model_config = ConfigDict(extra='allow')\n"
        "    x: int = 1\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    runner = CliRunner()
    result = runner.invoke(app, ["introspect-schema", "--config-class", "loose.c:LooseConfig"])
    assert result.exit_code != 0
    text = (result.stdout or "") + (result.stderr or "")
    assert "extra" in text


def test_introspect_rejects_bad_dotted_format() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["introspect-schema", "--config-class", "missing-colon-here"])
    assert result.exit_code != 0
    assert "expected" in (result.stdout + (result.stderr or ""))


def test_introspect_rejects_missing_module() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app, ["introspect-schema", "--config-class", "definitely_not_a_module:Foo"]
    )
    assert result.exit_code != 0
    text = result.stdout + (result.stderr or "")
    assert "cannot import" in text
    assert "definitely_not_a_module" in text


def test_introspect_rejects_missing_attribute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = tmp_path / "modwithoutattr"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "c.py").write_text("# empty\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(app, ["introspect-schema", "--config-class", "modwithoutattr.c:Missing"])
    assert result.exit_code != 0
    text = result.stdout + (result.stderr or "")
    assert "has no attribute" in text
    assert "Missing" in text


def test_introspect_rejects_unset_extra(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "default_extra"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "c.py").write_text(
        "from pydantic import BaseModel\nclass DefaultConfig(BaseModel):\n    x: int = 1\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(
        app, ["introspect-schema", "--config-class", "default_extra.c:DefaultConfig"]
    )
    assert result.exit_code != 0
    text = result.stdout + (result.stderr or "")
    assert "extra" in text
    # New requirement: surface the observed value so author can tell unset/ignore/allow apart
    assert "None" in text or "ignore" in text
