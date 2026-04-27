"""Binary-classification evaluator."""

from __future__ import annotations

import contextlib
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

from maldet.protocols import EventLogger, FeatureExtractor, SampleReader
from maldet.types import MetricReport


class BinaryClassification:
    """Binary classification metrics using ``sklearn.metrics``.

    Runs ``model.predict`` once over the whole reader. Optionally calls
    ``predict_proba`` if available to compute ROC-AUC.
    """

    def __init__(self, positive_class: str, class_names: Sequence[str]) -> None:
        if positive_class not in class_names:
            raise ValueError(
                f"positive_class {positive_class!r} not in class_names {list(class_names)!r}"
            )
        self._positive = positive_class
        self._classes = list(class_names)
        self._pos_idx = self._classes.index(positive_class)

    def evaluate(
        self,
        model: Any,
        reader: SampleReader,
        extractor: FeatureExtractor,
        *,
        logger: EventLogger,
    ) -> MetricReport:
        t0 = time.time()
        shas: list[str] = []
        ys: list[int] = []
        mats: list[np.ndarray] = []
        total = 0
        skipped = 0
        for sample in reader:
            if sample.label is None:
                raise ValueError("BinaryClassification.evaluate requires labeled samples")
            total += 1
            try:
                features_one = extractor.extract(sample)
            except ValueError as e:
                skipped += 1
                logger.log_event(
                    "warning",
                    message=(f"feature extractor failed on sample {sample.sha256}: {e}"),
                    sample_sha256=sample.sha256,
                )
                continue
            shas.append(sample.sha256)
            ys.append(1 if sample.label == self._positive else 0)
            mats.append(features_one)
        if not mats:
            raise RuntimeError("BinaryClassification.evaluate: zero usable samples")
        if total > 0 and skipped / total > 0.5:
            raise RuntimeError(
                f"BinaryClassification.evaluate: too many samples skipped by feature "
                f"extractor ({skipped}/{total})"
            )
        features = np.stack(mats)
        y = np.asarray(ys)

        y_pred = np.asarray(model.predict(features))
        metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
        }

        proba = getattr(model, "predict_proba", None)
        if callable(proba):
            probs = np.asarray(proba(features))[:, 1]
            with contextlib.suppress(ValueError):
                metrics["roc_auc"] = float(roc_auc_score(y, probs))

        cm = confusion_matrix(y, y_pred, labels=[0, 1]).tolist()
        p_per, r_per, f_per, s_per = precision_recall_fscore_support(
            y, y_pred, labels=[0, 1], zero_division=0
        )
        # Note on label order: index 0 = not-positive (first "other" class), index 1 = positive class.
        # So the confusion_matrix.labels returns [positive_class, other_class] to match model classes_.
        # The per_class dict keys are the actual class name strings.
        other = next(c for c in self._classes if c != self._positive)
        per_class = {
            self._positive: {
                "precision": float(p_per[1]),
                "recall": float(r_per[1]),
                "f1": float(f_per[1]),
                "support": int(s_per[1]),
            },
            other: {
                "precision": float(p_per[0]),
                "recall": float(r_per[0]),
                "f1": float(f_per[0]),
                "support": int(s_per[0]),
            },
        }

        report = MetricReport(
            task="binary_classification",
            n_samples=len(y),
            duration_seconds=float(time.time() - t0),
            metrics=metrics,
            per_class=per_class,
            confusion_matrix={"labels": [self._positive, other], "matrix": cm},
        )
        for k, v in metrics.items():
            logger.log_metric(k, v)
        return report
