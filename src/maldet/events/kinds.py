"""Event kinds + payload validators for the maldet event stream."""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class EventKind(StrEnum):
    STAGE_BEGIN = "stage_begin"
    STAGE_END = "stage_end"
    DATA_LOADED = "data_loaded"
    EPOCH_BEGIN = "epoch_begin"
    EPOCH_END = "epoch_end"
    METRIC = "metric"
    ARTIFACT_WRITTEN = "artifact_written"
    CHECKPOINT_SAVED = "checkpoint_saved"
    WARNING = "warning"
    ERROR = "error"
    CONFUSION_MATRIX = "confusion_matrix"
    PER_CLASS = "per_class"


ALL_EVENT_KINDS: tuple[str, ...] = tuple(k.value for k in EventKind)


_REQUIRED_FIELDS: dict[EventKind, tuple[str, ...]] = {
    EventKind.STAGE_BEGIN: ("stage",),
    EventKind.STAGE_END: ("stage", "status"),
    EventKind.DATA_LOADED: (),
    EventKind.EPOCH_BEGIN: ("epoch",),
    EventKind.EPOCH_END: ("epoch",),
    EventKind.METRIC: ("name", "value"),
    EventKind.ARTIFACT_WRITTEN: ("path",),
    EventKind.CHECKPOINT_SAVED: ("path",),
    EventKind.WARNING: ("message",),
    EventKind.ERROR: ("message",),
    EventKind.CONFUSION_MATRIX: ("labels", "matrix"),
    EventKind.PER_CLASS: ("per_class",),
}


def validate_payload(kind: EventKind, payload: dict[str, Any]) -> None:
    """Raise ``ValueError`` if required fields for ``kind`` are missing."""
    required = _REQUIRED_FIELDS[kind]
    missing = [f for f in required if f not in payload]
    if missing:
        joined = " and ".join(f"'{m}'" for m in missing)
        raise ValueError(f"{kind.value} requires {joined}")
