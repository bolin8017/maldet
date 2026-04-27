"""confusion_matrix event kind — emitted by evaluators after MetricReport."""

import pytest

from maldet.events.kinds import EventKind, validate_payload


def test_confusion_matrix_kind_in_enum():
    assert EventKind.CONFUSION_MATRIX.value == "confusion_matrix"


def test_confusion_matrix_validates_with_labels_and_matrix():
    validate_payload(EventKind.CONFUSION_MATRIX, {"labels": ["a", "b"], "matrix": [[1, 0], [0, 1]]})


def test_confusion_matrix_rejects_missing_labels():
    with pytest.raises(ValueError, match="labels"):
        validate_payload(EventKind.CONFUSION_MATRIX, {"matrix": [[1, 0], [0, 1]]})


def test_confusion_matrix_rejects_missing_matrix():
    with pytest.raises(ValueError, match="matrix"):
        validate_payload(EventKind.CONFUSION_MATRIX, {"labels": ["a", "b"]})
