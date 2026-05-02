"""Evaluator emits per_class event after confusion_matrix (lolday Phase 13b §1.6a)."""

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


def test_evaluate_emits_per_class(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(log_path)
    evaluator = BinaryClassification(positive_class="Malware", class_names=["Benign", "Malware"])

    evaluator.evaluate(
        model=_PerfectModel(),
        reader=_StubReader(_samples()),
        extractor=_StubExtractor(),
        logger=logger,
    )

    lines = log_path.read_text().strip().splitlines()
    pc_records = [json.loads(line) for line in lines if json.loads(line).get("kind") == "per_class"]
    assert len(pc_records) == 1

    pc = pc_records[0]["per_class"]
    assert set(pc.keys()) == {"Malware", "Benign"}

    for class_name in ("Malware", "Benign"):
        entry = pc[class_name]
        assert set(entry.keys()) >= {"precision", "recall", "f1", "support"}

    # _PerfectModel is a perfect classifier: all metrics should be 1.0.
    assert pc["Malware"]["precision"] == 1.0
    assert pc["Malware"]["recall"] == 1.0
    assert pc["Malware"]["f1"] == 1.0
    assert pc["Malware"]["support"] == 2  # two Malware samples

    assert pc["Benign"]["precision"] == 1.0
    assert pc["Benign"]["recall"] == 1.0
    assert pc["Benign"]["f1"] == 1.0
    assert pc["Benign"]["support"] == 1  # one Benign sample
