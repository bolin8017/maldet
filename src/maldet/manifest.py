"""Detector manifest — Pydantic model for ``maldet.toml`` and helpers."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DetectorInfo(_Frozen):
    name: str
    version: str
    framework: Literal["sklearn", "lightning", "sklearn+lightning"]
    description: str = ""


class InputConfig(_Frozen):
    binary_format: Literal["elf", "pe", "apk", "raw_bytes"]
    required_sections: list[str] = Field(default_factory=list)
    dataset_contract: str = "sample_csv"


class OutputConfig(_Frozen):
    task: Literal["binary_classification", "multiclass_classification", "regression", "ranking"]
    classes: list[str] = Field(default_factory=list)
    score_range: tuple[float, float] = (0.0, 1.0)


class ResourcesConfig(_Frozen):
    supports: list[str] = Field(default_factory=lambda: ["cpu"])
    recommended: str = "cpu"
    min_memory_gib: int = 1
    gpu_required: bool = False

    @field_validator("supports")
    @classmethod
    def _valid_profiles(cls, v: list[str]) -> list[str]:
        valid = {"cpu", "gpu1", "gpu2", "gpu4", "gpu8"}
        bad = [x for x in v if x not in valid]
        if bad:
            raise ValueError(f"unsupported resource profiles: {bad}")
        return v


def _default_stages() -> list[Literal["train", "evaluate", "predict"]]:
    return ["train", "evaluate", "predict"]


class LifecycleConfig(_Frozen):
    stages: list[Literal["train", "evaluate", "predict"]] = Field(default_factory=_default_stages)
    supports_serving: bool = False
    supports_hpsweep: bool = True
    supports_distributed: bool | Literal["ddp", "fsdp", "deepspeed"] = False
    supports_multinode: bool = False


class ArtifactSpec(_Frozen):
    path: str
    type: Literal["file", "dir"]


class ArtifactsConfig(_Frozen):
    model: ArtifactSpec
    metrics: ArtifactSpec = ArtifactSpec(path="metrics.json", type="file")
    predictions: ArtifactSpec = ArtifactSpec(path="predictions.csv", type="file")


class CompatConfig(_Frozen):
    min_python: str = "3.12"
    min_maldet: str = "1.0"
    schema_version: int = 1


class StageSpec(_Frozen):
    reader: str | None = None
    extractor: str | None = None
    model: str | None = None
    trainer: str | None = None
    evaluator: str | None = None
    predictor: str | None = None


class DetectorManifest(_Frozen):
    """The full manifest (``maldet.toml`` root)."""

    detector: DetectorInfo
    input: InputConfig
    output: OutputConfig
    resources: ResourcesConfig
    lifecycle: LifecycleConfig
    artifacts: ArtifactsConfig
    compat: CompatConfig = CompatConfig()
    stages: dict[Literal["train", "evaluate", "predict"], StageSpec] = Field(default_factory=dict)


class ManifestNotFoundError(FileNotFoundError):
    """Raised by ``search_manifest`` when no ``maldet.toml`` is discoverable."""


_APP_FALLBACK: Path = Path("/app/maldet.toml")


def load_manifest(path: Path) -> DetectorManifest:
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return DetectorManifest.model_validate(data)


def search_manifest() -> Path:
    """Return the first manifest path found in:
    1. ``$MALDET_MANIFEST`` env var (absolute path)
    2. ``$PWD/maldet.toml``
    3. ``/app/maldet.toml`` (the scaffold Docker WORKDIR)
    """
    env = os.environ.get("MALDET_MANIFEST")
    if env:
        p = Path(env)
        if p.is_file():
            return p

    cwd = Path.cwd() / "maldet.toml"
    if cwd.is_file():
        return cwd

    if _APP_FALLBACK.is_file():
        return _APP_FALLBACK

    raise ManifestNotFoundError(
        "maldet.toml not found. Set MALDET_MANIFEST or run from a directory containing it."
    )
