"""log_model dispatches to mlflow.<flavor>.log_model."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from maldet.events.mlflow_logger import MlflowEventLogger


def test_log_model_sklearn_dispatches_to_mlflow_sklearn() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    model = object()
    logger.log_model(model=model, flavor="sklearn", artifact_path="model")
    mlflow.sklearn.log_model.assert_called_once_with(
        model,
        artifact_path="model",
        signature=None,
        input_example=None,
        pip_requirements=None,
    )


def test_log_model_pytorch_dispatches_to_mlflow_pytorch() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    model = object()
    logger.log_model(
        model=model,
        flavor="pytorch",
        artifact_path="model",
        signature="sig",
        input_example="ex",
        pip_requirements=["torch==2.5"],
    )
    mlflow.pytorch.log_model.assert_called_once_with(
        model,
        artifact_path="model",
        signature="sig",
        input_example="ex",
        pip_requirements=["torch==2.5"],
    )


def test_log_model_pyfunc_dispatches_with_python_model_kw() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    model = object()
    logger.log_model(model=model, flavor="pyfunc", artifact_path="model")
    mlflow.pyfunc.log_model.assert_called_once_with(
        python_model=model,
        artifact_path="model",
        signature=None,
        input_example=None,
        pip_requirements=None,
    )


def test_log_model_unknown_flavor_raises() -> None:
    logger = MlflowEventLogger(mlflow=MagicMock())
    with pytest.raises(ValueError, match="unknown mlflow flavor"):
        logger.log_model(model=object(), flavor="tensorflow", artifact_path="model")


def test_log_model_noops_when_mlflow_unavailable() -> None:
    logger = MlflowEventLogger(mlflow=None)
    logger._mlflow = None
    logger.log_model(model=object(), flavor="sklearn")  # must not raise
