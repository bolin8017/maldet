"""Event-kind enum and payload validation."""

from __future__ import annotations

import pytest

from maldet.events.kinds import ALL_EVENT_KINDS, EventKind, validate_payload


def test_all_kinds_present() -> None:
    expected = {
        "stage_begin",
        "stage_end",
        "data_loaded",
        "epoch_begin",
        "epoch_end",
        "metric",
        "artifact_written",
        "checkpoint_saved",
        "warning",
        "error",
        "confusion_matrix",
        "per_class",
    }
    assert {k.value for k in EventKind} == expected
    assert set(ALL_EVENT_KINDS) == expected


def test_metric_requires_name_and_value() -> None:
    with pytest.raises(ValueError, match="metric requires 'name' and 'value'"):
        validate_payload(EventKind.METRIC, {})


def test_stage_end_requires_status() -> None:
    with pytest.raises(ValueError, match="stage_end requires 'status'"):
        validate_payload(EventKind.STAGE_END, {"stage": "train"})


def test_metric_payload_ok() -> None:
    validate_payload(EventKind.METRIC, {"name": "loss", "value": 0.3, "step": 1})


def test_warning_requires_message() -> None:
    with pytest.raises(ValueError, match="warning requires 'message'"):
        validate_payload(EventKind.WARNING, {"context": "x"})
