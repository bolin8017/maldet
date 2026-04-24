"""BinaryClassification evaluator computes sklearn metrics into MetricReport."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from maldet.evaluators.binary import BinaryClassification
from maldet.types import Sample


class DummyReader:
    def __init__(self, items: list[tuple[str, str]]) -> None:
        self._items = items

    def __iter__(self) -> Iterator[Sample]:
        for sha, label in self._items:
            yield Sample(sha256=sha, path=Path("/tmp") / sha, label=label)

    def __len__(self) -> int:
        return len(self._items)


class DummyExtractor:
    output_shape = (2,)
    dtype = "float32"

    def extract(self, sample: Sample) -> np.ndarray:
        if sample.label == "Malware":
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)


class PerfectModel:
    def predict(self, features: np.ndarray) -> np.ndarray:
        return (features[:, 1] > 0.5).astype(int)


class NoopLogger:
    def log_metric(self, *a: Any, **k: Any) -> None: ...
    def log_params(self, *a: Any, **k: Any) -> None: ...
    def log_artifact(self, *a: Any, **k: Any) -> None: ...
    def log_event(self, *a: Any, **k: Any) -> None: ...
    def set_tags(self, *a: Any, **k: Any) -> None: ...


def test_perfect_classifier() -> None:
    items = [(f"{i:064x}", "Malware" if i % 2 else "Benign") for i in range(10)]
    ev = BinaryClassification(positive_class="Malware", class_names=["Malware", "Benign"])
    report = ev.evaluate(PerfectModel(), DummyReader(items), DummyExtractor(), logger=NoopLogger())
    assert report.task == "binary_classification"
    assert report.n_samples == 10
    assert report.metrics["accuracy"] == 1.0
    assert report.metrics["f1"] == 1.0
    assert report.confusion_matrix["labels"] == ["Malware", "Benign"]
