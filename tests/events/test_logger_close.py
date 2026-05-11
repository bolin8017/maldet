"""close() is a graceful no-op on jsonl / stdout sinks; only mlflow uses it."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from maldet.events.jsonl import JsonlEventLogger
from maldet.events.logger import CompositeEventLogger
from maldet.events.stdout import StdoutEventLogger


def test_jsonl_close_is_noop(tmp_path: Path) -> None:
    logger = JsonlEventLogger(tmp_path / "events.jsonl")
    logger.close()  # must not raise
    logger.close()  # idempotent


def test_stdout_close_is_noop(capsys) -> None:
    logger = StdoutEventLogger()
    logger.close()
    logger.close()


def test_jsonl_log_model_writes_line(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(path)
    logger.log_model(model=object(), flavor="sklearn", artifact_path="model")
    text = path.read_text()
    assert '"kind": "model_logged"' in text
    assert '"flavor": "sklearn"' in text
    assert '"artifact_path": "model"' in text


def test_stdout_log_model_prints_line(capsys) -> None:
    logger = StdoutEventLogger()
    logger.log_model(model=object(), flavor="pytorch", artifact_path="model")
    out = capsys.readouterr().out
    assert "model_logged" in out
    assert "pytorch" in out


def test_composite_log_model_fans_out() -> None:
    a, b = MagicMock(), MagicMock()
    composite = CompositeEventLogger([a, b])
    composite.log_model(model=object(), flavor="sklearn", artifact_path="model")
    a.log_model.assert_called_once()
    b.log_model.assert_called_once()


def test_composite_close_fans_out_and_isolates_failure() -> None:
    a = MagicMock()
    a.close.side_effect = RuntimeError("boom")
    b = MagicMock()
    composite = CompositeEventLogger([a, b])
    composite.close()  # must not raise
    a.close.assert_called_once()
    b.close.assert_called_once()
