"""SklearnTrainer wraps estimator.fit and joblib.dump."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from maldet.trainers.sklearn_trainer import SklearnTrainer, _materialize
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
    output_shape = (4,)
    dtype = "uint8"

    def extract(self, sample: Sample) -> np.ndarray:
        if sample.label == "Malware":
            return np.array([1, 1, 1, 1], dtype=np.uint8)
        return np.array([0, 0, 0, 0], dtype=np.uint8)


class RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        self.events.append(("metric", {"name": name, "value": value, "step": step}))

    def log_params(self, params: dict[str, Any]) -> None:
        self.events.append(("params", dict(params)))

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        self.events.append(("artifact", {"path": str(path), "artifact_path": artifact_path}))

    def log_event(self, kind: str, **payload: Any) -> None:
        self.events.append((kind, payload))

    def set_tags(self, tags: dict[str, str]) -> None:
        self.events.append(("tags", dict(tags)))


def _train_items() -> list[tuple[str, str]]:
    return [(f"{i:064x}", "Malware" if i % 2 else "Benign") for i in range(20)]


def test_fit_emits_stage_events() -> None:
    logger = RecordingLogger()
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    trainer = SklearnTrainer()
    trainer.fit(model, DummyReader(_train_items()), DummyExtractor(), logger=logger)
    kinds = [e[0] for e in logger.events]
    assert "stage_begin" in kinds
    assert "stage_end" in kinds


def test_save_writes_joblib(tmp_path: Path) -> None:
    logger = RecordingLogger()
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    trainer = SklearnTrainer()
    result = trainer.fit(model, DummyReader(_train_items()), DummyExtractor(), logger=logger)
    out = tmp_path / "model"
    out.mkdir()
    trainer.save(result, out)
    assert (out / "model.joblib").exists()


def test_load_roundtrips(tmp_path: Path) -> None:
    logger = RecordingLogger()
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    trainer = SklearnTrainer()
    result = trainer.fit(model, DummyReader(_train_items()), DummyExtractor(), logger=logger)
    out = tmp_path / "model"
    out.mkdir()
    trainer.save(result, out)
    loaded = trainer.load(out)
    x = np.array([[1, 1, 1, 1]], dtype=np.uint8)
    assert loaded.predict(x).tolist() == [1]


class FailingExtractor:
    """Raises ValueError for samples in ``fail_shas``; returns dummy features otherwise.

    Models the real-world Phase 11 case where ~20% of corpus samples lack a
    ``.text`` section and cause Text256Extractor.extract to raise.
    """

    output_shape = (4,)
    dtype = "uint8"

    def __init__(self, fail_shas: set[str]) -> None:
        self._fail = fail_shas

    def extract(self, sample: Sample) -> np.ndarray:
        if sample.sha256 in self._fail:
            raise ValueError(f"simulated extractor failure on {sample.sha256}")
        return np.array([1, 1, 1, 1], dtype=np.uint8)


def test_materialize_skips_samples_when_extractor_raises_value_error() -> None:
    items = [(f"{i:064x}", "Malware" if i % 2 else "Benign") for i in range(20)]
    fail_shas = {items[3][0], items[7][0]}

    X, y = _materialize(  # noqa: N806
        DummyReader(items), FailingExtractor(fail_shas), require_labels=True
    )

    assert X.shape[0] == 18
    assert y.shape[0] == 18


def test_materialize_raises_when_more_than_half_samples_fail() -> None:
    items = [(f"{i:064x}", "Malware") for i in range(10)]
    fail_shas = {sha for sha, _ in items[:6]}  # 60% failure rate

    with pytest.raises(RuntimeError, match="too many"):
        _materialize(DummyReader(items), FailingExtractor(fail_shas), require_labels=True)


def test_materialize_emits_warning_event_per_skip_when_logger_provided() -> None:
    items = [(f"{i:064x}", "Malware") for i in range(10)]
    bad_sha = items[3][0]
    fail_shas = {bad_sha}
    logger = RecordingLogger()

    _materialize(
        DummyReader(items),
        FailingExtractor(fail_shas),
        require_labels=True,
        logger=logger,
    )

    warnings_emitted = [e for e in logger.events if e[0] == "warning"]
    assert len(warnings_emitted) == 1
    payload = warnings_emitted[0][1]
    assert "message" in payload
    assert bad_sha in payload["message"]
