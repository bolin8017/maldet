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
    """Predicts ``int(features[:, 1] > 0.5)`` — i.e. 1 for the Malware feature
    vector and 0 for the Benign feature vector. Paired with
    ``class_names=["Benign", "Malware"]`` (sklearn-convention alphabetical
    order), this makes Malware encode to index 1 and Benign to index 0, so the
    model is a perfect classifier under the ``classes.index`` encoding."""

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
    ev = BinaryClassification(positive_class="Malware", class_names=["Benign", "Malware"])
    report = ev.evaluate(PerfectModel(), DummyReader(items), DummyExtractor(), logger=NoopLogger())
    assert report.task == "binary_classification"
    assert report.n_samples == 10
    assert report.metrics["accuracy"] == 1.0
    assert report.metrics["f1"] == 1.0
    # CM labels now mirror the constructor's class_names order verbatim
    # (no more [other, positive] swap), so this is just ["Benign", "Malware"].
    assert report.confusion_matrix["labels"] == ["Benign", "Malware"]


class FailingExtractor:
    output_shape = (2,)
    dtype = "float32"

    def __init__(self, fail_shas: set[str]) -> None:
        self._fail = fail_shas

    def extract(self, sample: Sample) -> np.ndarray:
        if sample.sha256 in self._fail:
            raise ValueError(f"simulated extract failure on {sample.sha256}")
        if sample.label == "Malware":
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)


def test_evaluate_skips_samples_when_extractor_raises_value_error() -> None:
    items = [(f"{i:064x}", "Malware" if i % 2 else "Benign") for i in range(10)]
    fail_shas = {items[3][0], items[7][0]}
    ev = BinaryClassification(positive_class="Malware", class_names=["Benign", "Malware"])
    report = ev.evaluate(
        PerfectModel(),
        DummyReader(items),
        FailingExtractor(fail_shas),
        logger=NoopLogger(),
    )
    assert report.n_samples == 8


def test_metrics_correct_when_positive_class_is_index_zero() -> None:
    """classes=['Malware','Benign'], positive_class='Malware' is the inversion-prone
    config — Malware is at index 0, the trainer encodes Malware=0, but the legacy
    evaluator encoded ``1 if ==positive else 0`` (Malware=1). With the encoding fix
    both layers agree (Malware=0) and a perfect classifier scores 1.0 across the
    board.

    Regression test for the encoding bug closed by Tasks 1.5/1.6/1.7 (spec §4.4)."""

    class _Reader:
        def __init__(self, samples: list[Sample]) -> None:
            self._s = samples

        def __iter__(self) -> Iterator[Sample]:
            return iter(self._s)

        def __len__(self) -> int:
            return len(self._s)

    class _Extractor:
        output_shape = (1,)
        dtype = "float32"

        def extract(self, s: Sample) -> np.ndarray:
            # Each sample's feature is its label-as-index under
            # classes=["Malware","Benign"]: Malware=0, Benign=1. The model predicts
            # ``int(round(x[0]))`` so trainer/evaluator round-trip is perfect.
            return np.array([0.0 if s.label == "Malware" else 1.0], dtype=np.float32)

    class _PerfectModel:
        def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
            return np.asarray([int(round(x[0])) for x in X], dtype=np.int64)

    samples = [
        Sample(
            sha256=f"{i:064x}",
            path=Path("/tmp/_x"),
            label="Malware" if i % 2 else "Benign",
        )
        for i in range(10)
    ]
    ev = BinaryClassification(positive_class="Malware", class_names=["Malware", "Benign"])
    report = ev.evaluate(_PerfectModel(), _Reader(samples), _Extractor(), logger=NoopLogger())

    # If the evaluator still hardcoded ``1 if ==positive else 0``, recall would be
    # 0.0 because the model emits 0 for Malware (per its training under
    # classes.index encoding) but the evaluator would expect Malware=1.
    assert report.metrics["accuracy"] == 1.0
    assert report.metrics["recall"] == 1.0
    assert report.metrics["precision"] == 1.0
    assert report.metrics["f1"] == 1.0


def test_metrics_correct_when_positive_class_is_index_one() -> None:
    """classes=['Benign','Malware'], positive_class='Malware' — the alphabetical
    sklearn-convention case. This already worked before the fix; pinning the
    behaviour so neither encoding direction silently regresses."""

    class _Reader:
        def __init__(self, samples: list[Sample]) -> None:
            self._s = samples

        def __iter__(self) -> Iterator[Sample]:
            return iter(self._s)

        def __len__(self) -> int:
            return len(self._s)

    class _Extractor:
        output_shape = (1,)
        dtype = "float32"

        def extract(self, s: Sample) -> np.ndarray:
            # Under classes=["Benign","Malware"]: Benign=0, Malware=1.
            return np.array([1.0 if s.label == "Malware" else 0.0], dtype=np.float32)

    class _PerfectModel:
        def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
            return np.asarray([int(round(x[0])) for x in X], dtype=np.int64)

    samples = [
        Sample(
            sha256=f"{i:064x}",
            path=Path("/tmp/_x"),
            label="Malware" if i % 2 else "Benign",
        )
        for i in range(10)
    ]
    ev = BinaryClassification(positive_class="Malware", class_names=["Benign", "Malware"])
    report = ev.evaluate(_PerfectModel(), _Reader(samples), _Extractor(), logger=NoopLogger())

    assert report.metrics["accuracy"] == 1.0
    assert report.metrics["recall"] == 1.0
    assert report.metrics["precision"] == 1.0
    assert report.metrics["f1"] == 1.0
