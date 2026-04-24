"""Built-in predictor: batch prediction over a SampleReader."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from maldet.protocols import EventLogger, FeatureExtractor, SampleReader


class BatchPredictor:
    """Iterate samples, extract features, call ``model.predict`` in one batch.

    Writes a CSV with the required columns ``file_name, pred_label, pred_score``.
    Extra columns are added as ``pred_prob_<class>`` when ``predict_proba`` is
    available.
    """

    def __init__(self, class_names: Sequence[str]) -> None:
        self._class_names = list(class_names)

    def predict(
        self,
        model: Any,
        reader: SampleReader,
        extractor: FeatureExtractor,
        *,
        out_path: Path,
        logger: EventLogger,
    ) -> Path:
        shas: list[str] = []
        mats: list[np.ndarray] = []
        for sample in reader:
            shas.append(sample.sha256)
            mats.append(extractor.extract(sample))
        if not mats:
            raise RuntimeError("BatchPredictor: no samples yielded from reader")
        feat_matrix = np.stack(mats)

        preds = np.asarray(model.predict(feat_matrix))
        pred_label = [
            self._class_names[int(p)] if p < len(self._class_names) else str(int(p)) for p in preds
        ]

        pred_proba = getattr(model, "predict_proba", None)
        pred_score: list[float | None]
        prob_cols: dict[str, list[float]] = {}
        if callable(pred_proba):
            probs = np.asarray(pred_proba(feat_matrix))
            pred_score = [float(probs[i, int(preds[i])]) for i in range(len(preds))]
            for ci, cname in enumerate(self._class_names):
                prob_cols[f"pred_prob_{cname}"] = probs[:, ci].tolist()
        else:
            pred_score = [None for _ in preds]

        df = pd.DataFrame(
            {
                "file_name": shas,
                "pred_label": pred_label,
                "pred_score": pred_score,
                **prob_cols,
            }
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        logger.log_event("artifact_written", path=str(out_path), size_bytes=out_path.stat().st_size)
        return out_path
