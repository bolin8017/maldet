"""Runtime-checkable Protocols for maldet's six layers + EventLogger.

Protocols use structural typing — implementations do not need to inherit.
``@runtime_checkable`` enables ``isinstance(obj, Trainer)`` for pipeline-assembly
validation.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from maldet.types import MetricReport, Sample, TrainResult


@runtime_checkable
class SampleReader(Protocol):
    def __iter__(self) -> Iterator[Sample]: ...
    def __len__(self) -> int: ...


@runtime_checkable
class FeatureExtractor(Protocol):
    output_shape: tuple[int, ...] | None
    dtype: str

    def extract(self, sample: Sample) -> np.ndarray: ...


@runtime_checkable
class EventLogger(Protocol):
    def log_metric(self, name: str, value: float, step: int | None = None) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None: ...
    def log_event(self, kind: str, **payload: Any) -> None: ...
    def set_tags(self, tags: dict[str, str]) -> None: ...


@runtime_checkable
class Trainer(Protocol):
    def fit(
        self,
        model: Any,
        train: SampleReader,
        extractor: FeatureExtractor,
        *,
        val: SampleReader | None = None,
        logger: EventLogger,
    ) -> TrainResult: ...
    def save(self, result: TrainResult, out_dir: Path) -> None: ...
    def load(self, model_dir: Path) -> Any: ...


@runtime_checkable
class Evaluator(Protocol):
    def evaluate(
        self,
        model: Any,
        reader: SampleReader,
        extractor: FeatureExtractor,
        *,
        logger: EventLogger,
    ) -> MetricReport: ...


@runtime_checkable
class Predictor(Protocol):
    def predict(
        self,
        model: Any,
        reader: SampleReader,
        extractor: FeatureExtractor,
        *,
        out_path: Path,
        logger: EventLogger,
    ) -> Path: ...
