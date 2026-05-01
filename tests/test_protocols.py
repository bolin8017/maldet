"""Protocol conformance tests (runtime-checkable)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from maldet.protocols import (
    Evaluator,
    EventLogger,
    FeatureExtractor,
    Predictor,
    SampleReader,
    Trainer,
)
from maldet.types import MetricReport, Sample, TrainResult


class GoodReader:
    def __iter__(self) -> Iterator[Sample]:
        yield Sample(sha256="a" * 64, path=Path("/tmp/a"))

    def __len__(self) -> int:
        return 1


class BadReader:
    def __len__(self) -> int:
        return 1


class GoodExtractor:
    output_shape = (256,)
    dtype = "uint8"

    def extract(self, sample: Sample) -> np.ndarray:
        return np.zeros(256, dtype=np.uint8)


class DummyLogger:
    def log_metric(self, name: str, value: float, step: int | None = None) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None: ...
    def log_event(self, kind: str, **payload: Any) -> None: ...
    def set_tags(self, tags: dict[str, str]) -> None: ...


class GoodTrainer:
    def fit(
        self,
        model: Any,
        train: SampleReader,
        extractor: FeatureExtractor,
        *,
        classes: Sequence[str],
        val: SampleReader | None = None,
        logger: EventLogger,
    ) -> TrainResult:
        return TrainResult(model=model)

    def save(self, result: TrainResult, out_dir: Path) -> None: ...
    def load(self, model_dir: Path) -> Any:
        return object()


class GoodEvaluator:
    def evaluate(
        self,
        model: Any,
        reader: SampleReader,
        extractor: FeatureExtractor,
        *,
        logger: EventLogger,
    ) -> MetricReport:
        return MetricReport(
            task="binary_classification", n_samples=0, duration_seconds=0.0, metrics={}
        )


class GoodPredictor:
    def predict(
        self,
        model: Any,
        reader: SampleReader,
        extractor: FeatureExtractor,
        *,
        out_path: Path,
        logger: EventLogger,
    ) -> Path:
        return out_path


def test_sample_reader_protocol_accepts_good() -> None:
    assert isinstance(GoodReader(), SampleReader)


def test_sample_reader_protocol_rejects_missing_iter() -> None:
    assert not isinstance(BadReader(), SampleReader)


def test_feature_extractor_protocol() -> None:
    assert isinstance(GoodExtractor(), FeatureExtractor)


def test_event_logger_protocol() -> None:
    assert isinstance(DummyLogger(), EventLogger)


def test_trainer_protocol() -> None:
    assert isinstance(GoodTrainer(), Trainer)


def test_evaluator_protocol() -> None:
    assert isinstance(GoodEvaluator(), Evaluator)


def test_predictor_protocol() -> None:
    assert isinstance(GoodPredictor(), Predictor)
