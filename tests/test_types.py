"""Core dataclass tests."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from maldet.types import MetricReport, Sample, TrainResult


class TestSample:
    def test_fields_frozen(self) -> None:
        s = Sample(sha256="a" * 64, path=Path("/tmp/x"), label="Malware")
        with pytest.raises(dataclasses.FrozenInstanceError):
            s.sha256 = "b" * 64  # type: ignore[misc]

    def test_label_optional_for_predict(self) -> None:
        s = Sample(sha256="a" * 64, path=Path("/tmp/x"))
        assert s.label is None
        assert s.metadata == {}

    def test_metadata_is_per_instance(self) -> None:
        s1 = Sample(sha256="a" * 64, path=Path("/tmp/x"))
        s2 = Sample(sha256="b" * 64, path=Path("/tmp/y"))
        s1.metadata["seen"] = True  # type: ignore[index]
        assert "seen" not in s2.metadata

    def test_sha256_length_validated(self) -> None:
        with pytest.raises(ValueError, match="sha256 must be 64 hex chars"):
            Sample(sha256="abc", path=Path("/tmp/x"))


class TestTrainResult:
    def test_minimal(self) -> None:
        tr = TrainResult(model=object())
        assert tr.best_checkpoint is None
        assert tr.extras == {}


class TestMetricReport:
    def test_roundtrip_dict(self) -> None:
        m = MetricReport(
            task="binary_classification",
            n_samples=100,
            duration_seconds=1.2,
            metrics={"accuracy": 0.95, "f1": 0.94},
            per_class={"Malware": {"precision": 0.9}, "Benign": {"precision": 0.9}},
            confusion_matrix={"labels": ["Benign", "Malware"], "matrix": [[10, 1], [2, 87]]},
            extras={},
        )
        assert m.metrics["accuracy"] == 0.95
        d = m.to_json_dict()
        assert d["schema_version"] == 1
        assert d["metrics"]["accuracy"] == 0.95
