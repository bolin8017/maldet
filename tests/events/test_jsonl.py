"""JsonlEventLogger writes one NDJSON line per event, fsync per write."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from maldet.events.jsonl import JsonlEventLogger


def test_log_event_appends_line(tmp_path: Path) -> None:
    out = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(out)
    logger.log_event("stage_begin", stage="train", config_hash="abc")
    logger.log_event("stage_end", stage="train", status="success")
    lines = out.read_text().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["kind"] == "stage_begin"
    assert first["stage"] == "train"
    assert "ts" in first
    datetime.fromisoformat(first["ts"].replace("Z", "+00:00"))


def test_log_metric_is_a_metric_event(tmp_path: Path) -> None:
    out = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(out)
    logger.log_metric("train_loss", 0.34, step=1)
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["kind"] == "metric"
    assert rec["name"] == "train_loss"
    assert rec["value"] == 0.34
    assert rec["step"] == 1


def test_log_params_fanout_one_event_per_param(tmp_path: Path) -> None:
    out = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(out)
    logger.log_params({"lr": 1e-3, "batch_size": 32})
    lines = out.read_text().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["kind"] == "params"
    assert rec["params"] == {"lr": 1e-3, "batch_size": 32}


def test_file_is_created_if_missing(tmp_path: Path) -> None:
    out = tmp_path / "nested" / "events.jsonl"
    logger = JsonlEventLogger(out)
    logger.log_event("stage_begin", stage="train")
    assert out.exists()


def test_set_tags_becomes_tags_event(tmp_path: Path) -> None:
    out = tmp_path / "events.jsonl"
    logger = JsonlEventLogger(out)
    logger.set_tags({"gpu_count": "2"})
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["kind"] == "tags"
    assert rec["tags"] == {"gpu_count": "2"}
