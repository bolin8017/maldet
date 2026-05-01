"""per_class event kind — emitted by binary evaluators alongside confusion_matrix."""

from __future__ import annotations

import pytest

from maldet.events.kinds import EventKind, validate_payload


def test_per_class_kind_in_enum() -> None:
    assert EventKind.PER_CLASS.value == "per_class"


def test_per_class_validates_with_per_class_field() -> None:
    validate_payload(
        EventKind.PER_CLASS,
        {
            "per_class": {
                "Malware": {"precision": 0.9, "recall": 0.95, "f1": 0.92, "support": 100},
                "Benign": {"precision": 0.97, "recall": 0.93, "f1": 0.95, "support": 100},
            }
        },
    )


def test_per_class_rejects_missing_per_class() -> None:
    with pytest.raises(ValueError, match="per_class"):
        validate_payload(EventKind.PER_CLASS, {})
