"""MlflowEventLogger delegates to mlflow SDK (mocked)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from maldet.events.mlflow_logger import MlflowEventLogger


def test_log_metric_calls_mlflow() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    logger.log_metric("loss", 0.3, step=1)
    mlflow.log_metric.assert_called_once_with("loss", 0.3, step=1)


def test_log_params_calls_mlflow() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    logger.log_params({"lr": 1e-3, "bs": 32})
    mlflow.log_params.assert_called_once_with({"lr": 1e-3, "bs": 32})


def test_log_artifact_calls_mlflow(tmp_path: Path) -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    p = tmp_path / "f"
    p.write_text("x")
    logger.log_artifact(p, artifact_path="model")
    mlflow.log_artifact.assert_called_once_with(str(p), artifact_path="model")


def test_log_artifact_dir_uses_log_artifacts(tmp_path: Path) -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    d = tmp_path / "model"
    d.mkdir()
    logger.log_artifact(d, artifact_path="model")
    mlflow.log_artifacts.assert_called_once_with(str(d), artifact_path="model")


def test_log_event_is_a_mlflow_tag() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    logger.log_event("stage_begin", stage="train", config_hash="abc")
    mlflow.set_tag.assert_any_call("maldet.stage_begin.stage", "train")


def test_set_tags_calls_mlflow() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    logger.set_tags({"gpu_count": "2"})
    mlflow.set_tags.assert_called_once_with({"gpu_count": "2"})


def test_no_available_mlflow_noops_silently() -> None:
    # When mlflow is None (not installed AND not passed), methods are no-ops.
    logger = MlflowEventLogger(mlflow=None)
    # Force internal state: simulate "mlflow not importable"
    logger._mlflow = None  # type: ignore[attr-defined]
    logger.log_metric("x", 1.0)  # must not raise
    logger.log_params({"a": 1})
    logger.log_event("stage_begin", stage="x")
