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

    Labels are encoded via ``classes.index(sample.label)`` so that the
    evaluator's internal ``y`` matches the trainer's encoding for the same
    ``sample.label`` regardless of class ordering. ``pos_label`` is then
    threaded through sklearn metrics as the positive class index, so flipping
    ``classes`` order does not flip metric values.
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
        class_to_idx = {c: i for i, c in enumerate(self._classes)}
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
            if sample.label not in class_to_idx:
                raise ValueError(
                    f"sample.label={sample.label!r} not in manifest classes={list(self._classes)!r}"
                )
            shas.append(sample.sha256)
            ys.append(class_to_idx[sample.label])
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
            "precision": float(
                precision_score(y, y_pred, pos_label=self._pos_idx, zero_division=0)
            ),
            "recall": float(recall_score(y, y_pred, pos_label=self._pos_idx, zero_division=0)),
            "f1": float(f1_score(y, y_pred, pos_label=self._pos_idx, zero_division=0)),
        }

        proba = getattr(model, "predict_proba", None)
        if callable(proba):
            probs = np.asarray(proba(features))[:, self._pos_idx]
            with contextlib.suppress(ValueError):
                metrics["roc_auc"] = float(roc_auc_score((y == self._pos_idx).astype(int), probs))

        labels_idx = list(range(len(self._classes)))
        cm = confusion_matrix(y, y_pred, labels=labels_idx).tolist()
        cm_payload = {"labels": list(self._classes), "matrix": cm}

        p_per, r_per, f_per, s_per = precision_recall_fscore_support(
            y, y_pred, labels=labels_idx, zero_division=0
        )
        # Iterate in self._classes order so per-class dict keys match the
        # confusion-matrix label order. Dict insertion order is deterministic
        # (Python 3.7+), so consumers can rely on this iteration order.
        per_class: dict[str, dict[str, float | int]] = {
            self._classes[i]: {
                "precision": float(p_per[i]),
                "recall": float(r_per[i]),
                "f1": float(f_per[i]),
                "support": int(s_per[i]),
            }
            for i in range(len(self._classes))
        }

        report = MetricReport(
            task="binary_classification",
            n_samples=len(y),
            duration_seconds=float(time.time() - t0),
            metrics=metrics,
            per_class=per_class,
            confusion_matrix=cm_payload,
        )
        for k, v in metrics.items():
            logger.log_metric(k, v)
        logger.log_event(
            "confusion_matrix",
            labels=cm_payload["labels"],
            matrix=cm_payload["matrix"],
        )
        logger.log_event("per_class", per_class=per_class)
        return report
