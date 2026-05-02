"""LightningTrainer label-encoding tests (Task 1.6).

Calls ``_materialize_tensor`` directly so the tests don't pull in the full
Lightning training loop, but skips the whole module when ``lightning`` is not
installed in the active environment (CI installs it; some local dev shells
don't).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Skip the whole module if lightning isn't installed — the trainer itself
# imports lightning at module load, so the encoding helpers can't be tested
# in isolation without it.
pytest.importorskip("lightning.pytorch")

from maldet.trainers.lightning_trainer import _materialize_tensor
from maldet.types import Sample


class _Reader:
    def __init__(self, samples: list[Sample]) -> None:
        self._s = samples

    def __iter__(self) -> Iterator[Sample]:
        return iter(self._s)

    def __len__(self) -> int:
        return len(self._s)


class _Extractor:
    output_shape = (1,)
    dtype = "float32"

    def extract(self, sample: Sample) -> np.ndarray:
        return np.array([1.0], dtype=np.float32)


class _Logger:
    def log_metric(self, *a: Any, **kw: Any) -> None: ...
    def log_params(self, *a: Any, **kw: Any) -> None: ...
    def log_artifact(self, *a: Any, **kw: Any) -> None: ...
    def log_event(self, *a: Any, **kw: Any) -> None: ...
    def set_tags(self, *a: Any, **kw: Any) -> None: ...


def _samples(*labels: str) -> list[Sample]:
    return [
        Sample(sha256=f"{i:064x}", path=Path("/tmp/_x"), label=lbl) for i, lbl in enumerate(labels)
    ]


def test_lightning_encoding_alphabetical() -> None:
    """classes=['Benign', 'Malware'] → Benign=0, Malware=1."""
    samples = _samples("Benign", "Malware", "Benign")
    _x, y = _materialize_tensor(
        _Reader(samples),
        _Extractor(),
        classes=["Benign", "Malware"],
        logger=_Logger(),
    )
    assert y.tolist() == [0, 1, 0]


def test_lightning_encoding_positive_first() -> None:
    """classes=['Malware', 'Benign'] → Malware=0, Benign=1 (positive-first)."""
    samples = _samples("Benign", "Malware", "Benign")
    _x, y = _materialize_tensor(
        _Reader(samples),
        _Extractor(),
        classes=["Malware", "Benign"],
        logger=_Logger(),
    )
    assert y.tolist() == [1, 0, 1]


def test_lightning_encoding_unknown_label_raises() -> None:
    samples = _samples("Benign", "Outlier")
    with pytest.raises(ValueError, match="not in manifest classes"):
        _materialize_tensor(
            _Reader(samples),
            _Extractor(),
            classes=["Benign", "Malware"],
            logger=_Logger(),
        )


def test_lightning_materialize_requires_classes() -> None:
    """Passing an empty classes list is a ValueError (caller bug)."""
    samples = _samples("Malware")
    with pytest.raises(ValueError, match="non-empty sequence"):
        _materialize_tensor(
            _Reader(samples),
            _Extractor(),
            classes=[],
            logger=_Logger(),
        )
