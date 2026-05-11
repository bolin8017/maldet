"""StageRunner emits mlflow.log_input() for dataset lineage in train/evaluate/predict."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytest.importorskip("mlflow")


def test_train_branch_calls_log_input_with_training_context(tmp_path: Path) -> None:
    """We mock mlflow.log_input + mlflow.data.from_pandas to assert call site."""
    import pandas as pd

    from maldet.runner import _log_dataset_input

    train_csv = tmp_path / "train.csv"
    pd.DataFrame({"file_name": ["a", "b"], "label": ["Benign", "Malware"]}).to_csv(
        train_csv, index=False
    )

    class _Cfg:
        def get(self, k: str):  # type: ignore[no-untyped-def]
            return {"lolday": {"train_dataset_id": "abc-123"}}.get(k)

    with (
        patch("mlflow.log_input") as mock_log_input,
        patch("mlflow.data.from_pandas") as mock_from_pandas,
        patch("mlflow.active_run", return_value=object()),
    ):
        _log_dataset_input(_Cfg(), "train", train_csv)
        mock_from_pandas.assert_called_once()
        mock_log_input.assert_called_once()
        kwargs = mock_log_input.call_args.kwargs
        assert kwargs.get("context") == "training"


def test_evaluate_branch_uses_evaluation_context(tmp_path: Path) -> None:
    import pandas as pd

    from maldet.runner import _log_dataset_input

    csv = tmp_path / "test.csv"
    pd.DataFrame({"file_name": ["a"], "label": ["Benign"]}).to_csv(csv, index=False)

    class _Cfg:
        def get(self, k: str):  # type: ignore[no-untyped-def]
            return {"lolday": {"test_dataset_id": "xyz"}}.get(k)

    with (
        patch("mlflow.log_input") as mock_log_input,
        patch("mlflow.data.from_pandas"),
        patch("mlflow.active_run", return_value=object()),
    ):
        _log_dataset_input(_Cfg(), "evaluate", csv)
        assert mock_log_input.call_args.kwargs.get("context") == "evaluation"


def test_log_input_noop_when_no_active_run(tmp_path: Path) -> None:
    from maldet.runner import _log_dataset_input

    with patch("mlflow.active_run", return_value=None), patch("mlflow.log_input") as mock_log_input:
        _log_dataset_input(None, "train", tmp_path / "no.csv")
        mock_log_input.assert_not_called()


def test_log_input_noop_when_csv_missing(tmp_path: Path) -> None:
    from maldet.runner import _log_dataset_input

    with (
        patch("mlflow.active_run", return_value=object()),
        patch("mlflow.log_input") as mock_log_input,
    ):
        _log_dataset_input(None, "train", tmp_path / "nope.csv")
        mock_log_input.assert_not_called()
