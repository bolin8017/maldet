"""SklearnTrainer.save writes MLflow Models layout; load roundtrips via mlflow.sklearn."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from maldet.trainers.sklearn_trainer import SklearnTrainer
from maldet.types import Sample


class _DummyReader:
    def __init__(self, n: int) -> None:
        self._n = n

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self._n):
            yield Sample(
                sha256=f"{i:064x}",
                path=Path("/tmp") / f"{i}",
                label="Malware" if i % 2 else "Benign",
            )

    def __len__(self) -> int:
        return self._n


class _DummyExtractor:
    output_shape = (4,)
    dtype = "uint8"

    def extract(self, sample: Sample) -> np.ndarray:
        return np.array([1, 1, 1, 1] if sample.label == "Malware" else [0, 0, 0, 0], dtype=np.uint8)


class _RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def log_metric(self, name, value, step=None):  # type: ignore[no-untyped-def]
        self.events.append(("metric", {"name": name, "value": value}))

    def log_params(self, params):  # type: ignore[no-untyped-def]
        self.events.append(("params", dict(params)))

    def log_artifact(self, path, artifact_path=None):  # type: ignore[no-untyped-def]
        self.events.append(("artifact", {"path": str(path), "artifact_path": artifact_path}))

    def log_event(self, kind, **payload):  # type: ignore[no-untyped-def]
        self.events.append((kind, dict(payload)))

    def set_tags(self, tags):  # type: ignore[no-untyped-def]
        self.events.append(("tags", dict(tags)))

    def log_model(self, **kwargs):  # type: ignore[no-untyped-def]
        self.events.append(("model", kwargs))

    def close(self):  # type: ignore[no-untyped-def]
        pass


def _trained(tmp_path: Path) -> tuple[SklearnTrainer, Any, np.ndarray]:
    logger = _RecordingLogger()
    model = RandomForestClassifier(n_estimators=5, random_state=0)
    trainer = SklearnTrainer()
    result = trainer.fit(
        model,
        _DummyReader(20),
        _DummyExtractor(),
        classes=["Benign", "Malware"],
        logger=logger,
    )
    sample_x = np.stack(
        [
            np.array([1, 1, 1, 1], dtype=np.uint8),
            np.array([0, 0, 0, 0], dtype=np.uint8),
        ]
    )
    return trainer, result, sample_x


def test_save_writes_mlflow_models_layout(tmp_path: Path) -> None:
    trainer, result, sample_x = _trained(tmp_path)
    logger = _RecordingLogger()
    out = tmp_path / "model"
    trainer.save(result, out, logger=logger, signature_input_sample=sample_x)
    assert (out / "MLmodel").exists()
    # python_env.yaml is mlflow >= 2.0; older versions only had conda.yaml
    assert (out / "python_env.yaml").exists() or (out / "conda.yaml").exists()
    # Model artifact (mlflow.sklearn picks .pkl by default; could be model.pkl)
    pickle_files = list(out.glob("model.pkl")) + list(out.glob("*.pkl"))
    assert pickle_files, f"no .pkl file under {out}; ls={list(out.iterdir())}"


def test_save_logs_model_artifact_to_logger(tmp_path: Path) -> None:
    trainer, result, sample_x = _trained(tmp_path)
    logger = _RecordingLogger()
    out = tmp_path / "model"
    trainer.save(result, out, logger=logger, signature_input_sample=sample_x)
    artifact_events = [e for e in logger.events if e[0] == "artifact"]
    assert any(e[1]["artifact_path"] == "model" for e in artifact_events)


def test_load_via_mlflow_sklearn_roundtrips(tmp_path: Path) -> None:
    trainer, result, sample_x = _trained(tmp_path)
    logger = _RecordingLogger()
    out = tmp_path / "model"
    trainer.save(result, out, logger=logger, signature_input_sample=sample_x)

    loaded = trainer.load(out)
    pred_loaded = loaded.predict(sample_x)
    pred_original = result.model.predict(sample_x)
    np.testing.assert_array_equal(pred_loaded, pred_original)


def test_save_includes_signature_when_sample_provided(tmp_path: Path) -> None:
    trainer, result, sample_x = _trained(tmp_path)
    logger = _RecordingLogger()
    out = tmp_path / "model"
    trainer.save(result, out, logger=logger, signature_input_sample=sample_x)
    mlmodel_text = (out / "MLmodel").read_text()
    assert "signature:" in mlmodel_text
