"""StdoutEventLogger prints `maldet.event: {json}` lines."""

from __future__ import annotations

import json

import pytest

from maldet.events.stdout import StdoutEventLogger


def test_log_event_prints_json_line(capsys: pytest.CaptureFixture[str]) -> None:
    logger = StdoutEventLogger()
    logger.log_event("stage_begin", stage="train")
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("maldet.event: ")
    rec = json.loads(out[0].removeprefix("maldet.event: "))
    assert rec["kind"] == "stage_begin"
    assert rec["stage"] == "train"
