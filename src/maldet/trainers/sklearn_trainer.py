"""SklearnTrainer — thin wrapper around sklearn estimator.fit/predict/proba."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from maldet.protocols import EventLogger, FeatureExtractor, SampleReader
from maldet.types import TrainResult

_MODEL_FILENAME = "model.joblib"


def _materialize(
    reader: SampleReader, extractor: FeatureExtractor, require_labels: bool
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for sample in reader:
        xs.append(extractor.extract(sample))
        if require_labels:
            if sample.label is None:
                raise ValueError(
                    "SklearnTrainer: reader yielded an unlabeled sample during fit/val"
                )
            ys.append(1 if sample.label == "Malware" else 0)
    if not xs:
        raise RuntimeError("SklearnTrainer: reader yielded zero samples")
    X = np.stack(xs)  # noqa: N806
    y = np.asarray(ys, dtype=np.int64) if require_labels else np.empty(0, dtype=np.int64)
    return X, y


class SklearnTrainer:
    """Trainer for scikit-learn-compatible estimators (``fit`` + ``predict``)."""

    def fit(
        self,
        model: Any,
        train: SampleReader,
        extractor: FeatureExtractor,
        *,
        val: SampleReader | None = None,
        logger: EventLogger,
    ) -> TrainResult:
        logger.log_event("stage_begin", stage="train")
        if hasattr(model, "get_params"):
            logger.log_params({k: str(v) for k, v in model.get_params().items()})

        X, y = _materialize(train, extractor, require_labels=True)  # noqa: N806
        logger.log_event("data_loaded", n_train=int(X.shape[0]))

        t0 = time.time()
        model.fit(X, y)
        duration = float(time.time() - t0)
        logger.log_metric("train_time_seconds", duration)

        if val is not None:
            Xv, yv = _materialize(val, extractor, require_labels=True)  # noqa: N806
            acc = float(accuracy_score(yv, model.predict(Xv)))
            logger.log_metric("val_accuracy", acc)

        logger.log_event("stage_end", stage="train", status="success")
        return TrainResult(model=model, extras={"train_time_seconds": duration})

    def save(self, result: TrainResult, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.model, out_dir / _MODEL_FILENAME)

    def load(self, model_dir: Path) -> Any:
        return joblib.load(model_dir / _MODEL_FILENAME)
