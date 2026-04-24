"""DetectorManifest parsing and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from maldet.manifest import (
    DetectorManifest,
    ManifestNotFoundError,
    load_manifest,
    search_manifest,
)

FIX = Path(__file__).parent / "fixtures"


def test_load_valid() -> None:
    m = load_manifest(FIX / "valid_manifest.toml")
    assert m.detector.name == "elfrfdet"
    assert m.detector.framework == "sklearn"
    assert m.resources.supports == ["cpu", "gpu1", "gpu2"]
    assert m.lifecycle.stages == ["train", "evaluate", "predict"]
    assert m.stages["train"].trainer == "maldet.trainers.sklearn_trainer:SklearnTrainer"


def test_framework_value_validated(tmp_path: Path) -> None:
    bad = tmp_path / "m.toml"
    bad.write_text(
        (FIX / "valid_manifest.toml")
        .read_text()
        .replace('framework = "sklearn"', 'framework = "tensorflow"')
    )
    with pytest.raises(ValidationError):
        load_manifest(bad)


def test_search_manifest_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    p = tmp_path / "m.toml"
    p.write_text((FIX / "valid_manifest.toml").read_text())
    monkeypatch.setenv("MALDET_MANIFEST", str(p))
    assert search_manifest() == p


def test_search_manifest_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MALDET_MANIFEST", raising=False)
    p = tmp_path / "maldet.toml"
    p.write_text((FIX / "valid_manifest.toml").read_text())
    monkeypatch.chdir(tmp_path)
    assert search_manifest() == p


def test_search_manifest_raises_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MALDET_MANIFEST", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("maldet.manifest._APP_FALLBACK", tmp_path / "__nope__" / "maldet.toml")
    with pytest.raises(ManifestNotFoundError):
        search_manifest()


def test_json_serialization_roundtrip() -> None:
    m = load_manifest(FIX / "valid_manifest.toml")
    j = m.model_dump(mode="json")
    m2 = DetectorManifest.model_validate(j)
    assert m2.detector.name == "elfrfdet"
