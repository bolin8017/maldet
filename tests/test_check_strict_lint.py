"""maldet check fails when stage config_class doesn't set extra='forbid'."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from typer.testing import CliRunner

from maldet.cli import app


@pytest.fixture
def loose_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Lay out a fake detector repo with a Pydantic class that has extra='allow'."""
    pkg = tmp_path / "loose_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "configs.py").write_text(
        textwrap.dedent(
            """\
            from pydantic import BaseModel, ConfigDict

            class LooseConfig(BaseModel):
                model_config = ConfigDict(extra='allow')
                n: int = 1
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest = tmp_path / "maldet.toml"
    manifest.write_text(
        textwrap.dedent(
            """\
            [detector]
            name = "x"
            version = "0.1"
            framework = "sklearn"

            [input]
            binary_format = "elf"

            [output]
            task = "binary_classification"
            classes = ["Malware", "Benign"]
            positive_class = "Malware"

            [resources]

            [lifecycle]

            [artifacts]
            model = { path = "model/", type = "dir" }

            [stages.train]
            config_class = "loose_pkg.configs:LooseConfig"
            params_schema = {}
            """
        )
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def strict_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Same shape as loose_repo but with extra='forbid'."""
    pkg = tmp_path / "strict_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "configs.py").write_text(
        textwrap.dedent(
            """\
            from pydantic import BaseModel, ConfigDict

            class StrictConfig(BaseModel):
                model_config = ConfigDict(extra='forbid')
                n: int = 1
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest = tmp_path / "maldet.toml"
    manifest.write_text(
        textwrap.dedent(
            """\
            [detector]
            name = "x"
            version = "0.1"
            framework = "sklearn"

            [input]
            binary_format = "elf"

            [output]
            task = "binary_classification"
            classes = ["Malware", "Benign"]
            positive_class = "Malware"

            [resources]

            [lifecycle]

            [artifacts]
            model = { path = "model/", type = "dir" }

            [stages.train]
            config_class = "strict_pkg.configs:StrictConfig"
            params_schema = {}
            """
        )
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def dataclass_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A non-BaseModel config_class — should also fail."""
    pkg = tmp_path / "dc_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "configs.py").write_text(
        textwrap.dedent(
            """\
            from dataclasses import dataclass

            @dataclass
            class DataclassConfig:
                n: int = 1
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest = tmp_path / "maldet.toml"
    manifest.write_text(
        textwrap.dedent(
            """\
            [detector]
            name = "x"
            version = "0.1"
            framework = "sklearn"

            [input]
            binary_format = "elf"

            [output]
            task = "binary_classification"
            classes = ["Malware", "Benign"]
            positive_class = "Malware"

            [resources]

            [lifecycle]

            [artifacts]
            model = { path = "model/", type = "dir" }

            [stages.train]
            config_class = "dc_pkg.configs:DataclassConfig"
            params_schema = {}
            """
        )
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_check_rejects_loose_config_class(loose_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["check"])
    assert result.exit_code != 0, result.stdout
    text = (result.stdout or "") + (result.stderr or "")
    assert "extra" in text.lower()
    assert "train" in text.lower()  # offending stage named


def test_check_rejects_dataclass_config_class(dataclass_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["check"])
    assert result.exit_code != 0, result.stdout
    text = (result.stdout or "") + (result.stderr or "")
    assert "BaseModel" in text or "pydantic" in text.lower()
    assert "train" in text.lower()


def test_check_accepts_strict_config_class(strict_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["check"])
    # The strict_repo manifest only fills in config_class for the train stage
    # and doesn't declare reader/extractor/model/etc. — those are None and skipped
    # by the existing _check_symbol loop (it only validates non-None fields).
    # So this should exit 0 with "OK".
    assert result.exit_code == 0, (result.stdout, result.stderr)


@pytest.fixture
def unset_extra_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A BaseModel that simply doesn't declare model_config at all (default extra=ignore)."""
    pkg = tmp_path / "default_pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "configs.py").write_text(
        textwrap.dedent(
            """\
            from pydantic import BaseModel

            class DefaultConfig(BaseModel):
                n: int = 1
            """
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    manifest = tmp_path / "maldet.toml"
    manifest.write_text(
        textwrap.dedent(
            """\
            [detector]
            name = "x"
            version = "0.1"
            framework = "sklearn"

            [input]
            binary_format = "elf"

            [output]
            task = "binary_classification"
            classes = ["Malware", "Benign"]
            positive_class = "Malware"

            [resources]

            [lifecycle]

            [artifacts]
            model = { path = "model/", type = "dir" }

            [stages.train]
            config_class = "default_pkg.configs:DefaultConfig"
            params_schema = {}
            """
        )
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_check_message_distinguishes_unset_extra(unset_extra_repo: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["check"])
    assert result.exit_code != 0
    text = (result.stdout or "") + (result.stderr or "")
    assert "missing" in text.lower()
    # And NOT the "got 'allow'" form, since the user never set 'allow' here
    assert "got 'allow'" not in text
