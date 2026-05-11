"""MlflowEventLogger routes EventKind payloads to the right MLflow API.

Spec §5.2 — confusion_matrix / per_class are structured artifacts not tags;
warnings/errors are buffered + flushed on close().
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

from maldet.events.mlflow_logger import MlflowEventLogger


def test_confusion_matrix_writes_dict_artifact_not_stringified_tag() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    logger.log_event(
        "confusion_matrix",
        labels=["Benign", "Malware"],
        matrix=[[90, 0], [1, 77]],
    )
    mlflow.log_dict.assert_called_once_with(
        {"labels": ["Benign", "Malware"], "matrix": [[90, 0], [1, 77]]},
        "confusion_matrix.json",
    )
    # critically — the old stringified tag pattern must NOT happen
    mlflow.set_tag.assert_not_called()


def test_per_class_writes_dict_and_per_class_metrics() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    per_class = {
        "Benign": {"precision": 0.989, "recall": 1.0, "f1": 0.994, "support": 90},
        "Malware": {"precision": 1.0, "recall": 0.987, "f1": 0.994, "support": 78},
    }
    logger.log_event("per_class", per_class=per_class)
    mlflow.log_dict.assert_called_once_with(per_class, "per_class_metrics.json")
    calls = [c.args for c in mlflow.log_metric.call_args_list]
    assert ("per_class/Benign/precision", 0.989) in calls
    assert ("per_class/Malware/f1", 0.994) in calls
    assert ("per_class/Benign/support", 90.0) in calls


def test_data_loaded_emits_metric_not_tag() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    logger.log_event("data_loaded", n_train=645)
    mlflow.log_metric.assert_called_with("maldet/n_train", 645.0)


def test_warning_is_buffered_not_set_as_tag() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    logger.log_event("warning", message="bad sample 1", sample_sha256="aaa")
    logger.log_event("warning", message="bad sample 2", sample_sha256="bbb")
    # No tag overwrites
    mlflow.set_tag.assert_not_called()
    # Both stored in buffer
    assert len(logger._warning_buf) == 2
    assert logger._warning_buf[0]["sample_sha256"] == "aaa"
    assert logger._warning_buf[1]["sample_sha256"] == "bbb"


def test_close_flushes_warnings_to_log_text() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    logger.log_event("warning", message="m1", sample_sha256="a")
    logger.log_event("warning", message="m2", sample_sha256="b")
    logger.close()
    # warnings.jsonl uploaded as JSONL string
    args = mlflow.log_text.call_args
    text, name = args.args
    assert name == "warnings.jsonl"
    lines = [json.loads(line) for line in text.splitlines() if line]
    assert len(lines) == 2
    # also a count metric
    mlflow.log_metric.assert_any_call("maldet/warnings_total", 2.0)


def test_close_with_no_warnings_does_not_call_log_text() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    logger.close()
    mlflow.log_text.assert_not_called()


def test_stage_begin_writes_stage_tag_and_timestamp() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    logger.log_event("stage_begin", stage="train")
    calls = [c.args for c in mlflow.set_tag.call_args_list]
    assert ("maldet.stage", "train") in calls
    keys_set = {c.args[0] for c in mlflow.set_tag.call_args_list}
    assert "maldet.stage_begin_ts" in keys_set


def test_stage_end_writes_status_tag() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    logger.log_event("stage_end", stage="train", status="success")
    calls = [c.args for c in mlflow.set_tag.call_args_list]
    assert ("maldet.status", "success") in calls
    assert ("maldet.stage_end", "train") in calls


def test_artifact_written_logs_metric_for_size_and_tag_for_path() -> None:
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    logger.log_event("artifact_written", path="/mnt/output/predictions.csv", size_bytes=14541)
    mlflow.set_tag.assert_any_call("maldet.artifact.predictions.csv", "/mnt/output/predictions.csv")
    mlflow.log_metric.assert_any_call("maldet/artifact_bytes/predictions.csv", 14541.0)


def test_unknown_kind_falls_back_to_scoped_tags() -> None:
    """Forward compat: unknown event kind shouldn't crash; payload still gets recorded as tags."""
    mlflow = MagicMock()
    logger = MlflowEventLogger(mlflow=mlflow)
    logger.log_event("my_future_event", foo="bar", n=42)
    mlflow.set_tag.assert_any_call("maldet.my_future_event.foo", "bar")
    mlflow.set_tag.assert_any_call("maldet.my_future_event.n", "42")
