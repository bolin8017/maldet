"""SklearnTrainer — thin wrapper around sklearn estimator.fit/predict/proba."""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

from maldet.protocols import EventLogger, FeatureExtractor, SampleReader
from maldet.types import TrainResult

_MODEL_FILENAME = "model.joblib"


_SKIP_THRESHOLD = 0.5


def _materialize(
    reader: SampleReader,
    extractor: FeatureExtractor,
    require_labels: bool,
    *,
    classes: Sequence[str] | None = None,
    logger: EventLogger | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize a reader into (X, y) arrays.

    Labels are encoded as ``classes.index(sample.label)``. ``classes`` is
    required when ``require_labels=True`` so that internal int labels match
    the manifest's declared class ordering instead of the historical
    hardcoded ``1 if "Malware" else 0`` mapping.
    """
    if require_labels and not classes:
        raise ValueError(
            "SklearnTrainer: classes is required when require_labels=True; "
            "pass the manifest's output.classes list"
        )
    class_to_idx = {c: i for i, c in enumerate(classes or [])}
    xs: list[np.ndarray] = []
    ys: list[int] = []
    total = 0
    skipped = 0
    for sample in reader:
        total += 1
        try:
            features = extractor.extract(sample)
        except ValueError as e:
            skipped += 1
            if logger is not None:
                logger.log_event(
                    "warning",
                    message=f"feature extractor failed on sample {sample.sha256}: {e}",
                    sample_sha256=sample.sha256,
                )
            continue
        xs.append(features)
        if require_labels:
            if sample.label is None:
                raise ValueError(
                    "SklearnTrainer: reader yielded an unlabeled sample during fit/val"
                )
            if sample.label not in class_to_idx:
                raise ValueError(
                    f"sample.label={sample.label!r} not in manifest classes={list(classes or [])!r}"
                )
            ys.append(class_to_idx[sample.label])
    if not xs:
        raise RuntimeError("SklearnTrainer: reader yielded zero samples")
    if total > 0 and skipped / total > _SKIP_THRESHOLD:
        raise RuntimeError(
            f"SklearnTrainer: too many samples skipped by feature extractor "
            f"({skipped}/{total}); aborting to avoid training on a degenerate dataset"
        )
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
        classes: Sequence[str],
        val: SampleReader | None = None,
        logger: EventLogger,
    ) -> TrainResult:
        logger.log_event("stage_begin", stage="train")
        if hasattr(model, "get_params"):
            logger.log_params({k: str(v) for k, v in model.get_params().items()})

        X, y = _materialize(  # noqa: N806
            train, extractor, require_labels=True, classes=classes, logger=logger
        )
        logger.log_event("data_loaded", n_train=int(X.shape[0]))

        t0 = time.time()
        model.fit(X, y)
        duration = float(time.time() - t0)
        logger.log_metric("train_time_seconds", duration)

        if val is not None:
            Xv, yv = _materialize(  # noqa: N806
                val, extractor, require_labels=True, classes=classes, logger=logger
            )
            acc = float(accuracy_score(yv, model.predict(Xv)))
            logger.log_metric("val_accuracy", acc)

        logger.log_event("stage_end", stage="train", status="success")
        return TrainResult(model=model, extras={"train_time_seconds": duration})

    def save(self, result: TrainResult, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(result.model, out_dir / _MODEL_FILENAME)

    def load(self, model_dir: Path) -> Any:
        return joblib.load(model_dir / _MODEL_FILENAME)
