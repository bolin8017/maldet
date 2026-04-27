"""Evaluator emits confusion_matrix event after metrics are computed (phase 11e)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from maldet.evaluators.binary import BinaryClassification
from maldet.events.jsonl import JsonlEventLogger
from maldet.types import Sample


class _StubReader:
    def __init__(self, samples: list[Sample]) -> None:
        self._samples = samples

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._samples)


class _StubExtractor:
    output_shape = (2,)
    dtype = "float32"

    def extract(self, sample: Sample) -> np.ndarray:
        if sample.label == "Malware":
            return np.array([0.0, 1.0], dtype=np.float32)
        return np.array([1.0, 0.0], dtype=np.float32)


class _PerfectModel:
    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return (X[:, 1] > 0.5).astype(int)


def _samples() -> list[Sample]:
    return [
        Sample(sha256="a" * 64, path=Path("/tmp/a"), label="Malware"),
        Sample(sha256="b" * 64, path=Path("/tmp/b"), label="Benign"),
        Sample(sha256="c" * 64, path=Path("/tmp/c"), label="Malware"),
    ]


def test_evaluate_emits_confusion_matrix(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(log_path)
    evaluator = BinaryClassification(positive_class="Malware", class_names=["Malware", "Benign"])

    evaluator.evaluate(
        model=_PerfectModel(),
        reader=_StubReader(_samples()),
        extractor=_StubExtractor(),
        logger=logger,
    )

    lines = log_path.read_text().strip().splitlines()
    cm_records = [
        json.loads(line) for line in lines if json.loads(line).get("kind") == "confusion_matrix"
    ]
    assert len(cm_records) == 1
    cm = cm_records[0]
    # Row order matches sklearn labels=[0, 1] = [other, positive] = ["Benign", "Malware"].
    assert cm["labels"] == ["Benign", "Malware"]
    # 1 Benign predicted Benign (TN), 0 Benign predicted Malware (FP),
    # 0 Malware predicted Benign (FN), 2 Malware predicted Malware (TP).
    assert cm["matrix"] == [[1, 0], [0, 2]]
