"""DetectorManifest parsing and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from maldet.manifest import (
    CompatConfig,
    DetectorManifest,
    ManifestNotFoundError,
    OutputConfig,
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


def test_stage_requires_config_class_and_params_schema() -> None:
    """Phase 11e: each stage in maldet.toml must declare config_class + params_schema."""
    bad = {
        "detector": {"name": "x", "version": "0.1", "framework": "sklearn"},
        "input": {"binary_format": "elf"},
        "output": {"task": "binary_classification"},
        "resources": {},
        "lifecycle": {},
        "artifacts": {"model": {"path": "model/", "type": "dir"}},
        "stages": {"train": {"reader": "m:R"}},  # missing config_class + params_schema
    }
    with pytest.raises(ValidationError):
        DetectorManifest.model_validate(bad)


def test_stage_accepts_config_class_and_params_schema() -> None:
    good = {
        "detector": {"name": "x", "version": "0.1", "framework": "sklearn"},
        "input": {"binary_format": "elf"},
        "output": {
            "task": "binary_classification",
            "classes": ["Malware", "Benign"],
            "positive_class": "Malware",
        },
        "resources": {},
        "lifecycle": {},
        "artifacts": {"model": {"path": "model/", "type": "dir"}},
        "stages": {
            "train": {
                "reader": "m:R",
                "config_class": "elfrfdet.configs:TrainConfig",
                "params_schema": {"type": "object", "properties": {}},
            }
        },
    }
    m = DetectorManifest.model_validate(good)
    assert m.stages["train"].config_class == "elfrfdet.configs:TrainConfig"
    assert m.stages["train"].params_schema == {"type": "object", "properties": {}}


def test_positive_class_required_for_binary() -> None:
    with pytest.raises(ValidationError, match="positive_class is required"):
        OutputConfig(
            task="binary_classification",
            classes=["Benign", "Malware"],
            score_range=(0.0, 1.0),
        )


def test_positive_class_must_be_in_classes() -> None:
    with pytest.raises(ValidationError, match=r"not in output\.classes"):
        OutputConfig(
            task="binary_classification",
            classes=["Benign", "Malware"],
            positive_class="NotARealClass",
            score_range=(0.0, 1.0),
        )


def test_binary_classification_requires_two_classes() -> None:
    with pytest.raises(ValidationError, match="exactly 2 classes"):
        OutputConfig(
            task="binary_classification",
            classes=["A", "B", "C"],
            positive_class="A",
            score_range=(0.0, 1.0),
        )


def test_positive_class_optional_for_multiclass() -> None:
    cfg = OutputConfig(
        task="multiclass_classification",
        classes=["A", "B", "C"],
        score_range=(0.0, 1.0),
    )
    assert cfg.positive_class is None


def test_binary_with_valid_positive_class() -> None:
    cfg = OutputConfig(
        task="binary_classification",
        classes=["Benign", "Malware"],
        positive_class="Malware",
        score_range=(0.0, 1.0),
    )
    assert cfg.positive_class == "Malware"


def test_schema_version_default_is_2() -> None:
    cfg = CompatConfig()
    assert cfg.schema_version == 2


def test_min_maldet_default_is_2_0() -> None:
    cfg = CompatConfig()
    assert cfg.min_maldet == "2.0"
