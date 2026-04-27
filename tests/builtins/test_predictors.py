"""BatchPredictor writes predictions.csv with the required columns."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from maldet.builtins.predictors import BatchPredictor
from maldet.types import Sample


class DummyReader:
    def __init__(self, shas: list[str]) -> None:
        self._shas = shas

    def __iter__(self) -> Iterator[Sample]:
        for sha in self._shas:
            yield Sample(sha256=sha, path=Path("/tmp") / sha)

    def __len__(self) -> int:
        return len(self._shas)


class DummyExtractor:
    output_shape = (4,)
    dtype = "float32"

    def extract(self, sample: Sample) -> np.ndarray:
        return np.arange(4, dtype=np.float32)


class DummyModel:
    classes_: ClassVar[list[int]] = [0, 1]

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.ones(len(X), dtype=np.int64)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return np.stack([np.full(len(X), 0.2), np.full(len(X), 0.8)], axis=1)


class NoopLogger:
    def log_metric(self, *a: Any, **k: Any) -> None: ...
    def log_params(self, *a: Any, **k: Any) -> None: ...
    def log_artifact(self, *a: Any, **k: Any) -> None: ...
    def log_event(self, *a: Any, **k: Any) -> None: ...
    def set_tags(self, *a: Any, **k: Any) -> None: ...


def test_predict_writes_required_columns(tmp_path: Path) -> None:
    predictor = BatchPredictor(class_names=["Benign", "Malware"])
    out = tmp_path / "predictions.csv"
    shas = ["a" * 64, "b" * 64]
    predictor.predict(
        DummyModel(),
        DummyReader(shas),
        DummyExtractor(),
        out_path=out,
        logger=NoopLogger(),
    )
    df = pd.read_csv(out)
    assert list(df.columns)[:3] == ["file_name", "pred_label", "pred_score"]
    assert df["pred_label"].tolist() == ["Malware", "Malware"]
    assert list(df["pred_score"]) == [0.8, 0.8]


def test_predict_handles_no_predict_proba(tmp_path: Path) -> None:
    class ModelWithoutProba:
        classes_: ClassVar[list[int]] = [0, 1]

        def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
            return np.ones(len(X), dtype=np.int64)

    predictor = BatchPredictor(class_names=["Benign", "Malware"])
    out = tmp_path / "predictions.csv"
    predictor.predict(
        ModelWithoutProba(),
        DummyReader(["c" * 64]),
        DummyExtractor(),
        out_path=out,
        logger=NoopLogger(),
    )
    df = pd.read_csv(out)
    assert df["pred_score"].isna().all()


class FailingExtractor:
    output_shape = (4,)
    dtype = "float32"

    def __init__(self, fail_shas: set[str]) -> None:
        self._fail = fail_shas

    def extract(self, sample: Sample) -> np.ndarray:
        if sample.sha256 in self._fail:
            raise ValueError(f"simulated extract failure on {sample.sha256}")
        return np.arange(4, dtype=np.float32)


def test_predict_skips_samples_when_extractor_raises_value_error(tmp_path: Path) -> None:
    shas = [f"{i:064x}" for i in range(10)]
    fail_shas = {shas[3], shas[7]}
    predictor = BatchPredictor(class_names=["Benign", "Malware"])
    out = tmp_path / "predictions.csv"
    predictor.predict(
        DummyModel(),
        DummyReader(shas),
        FailingExtractor(fail_shas),
        out_path=out,
        logger=NoopLogger(),
    )
    # Read file_name as string so the 64-hex-char identifiers don't get
    # auto-coerced to int by pandas' inference.
    df = pd.read_csv(out, dtype={"file_name": str})
    assert len(df) == 8
    assert set(df["file_name"].tolist()) == set(shas) - fail_shas
