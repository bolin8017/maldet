"""Core dataclasses shared across maldet layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Sample:
    """A single binary sample.

    ``label`` is ``None`` during ``predict``. ``metadata`` is a per-instance mutable
    dict — frozen=True forbids reassigning the field, not mutating the contained dict.
    """

    sha256: str
    path: Path
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.sha256) != 64 or not all(c in "0123456789abcdef" for c in self.sha256.lower()):
            raise ValueError(f"sha256 must be 64 hex chars (got {self.sha256!r})")


@dataclass
class TrainResult:
    """Return value of Trainer.fit().

    ``model`` is the trained estimator or LightningModule. ``best_checkpoint`` is set
    by Lightning when a ModelCheckpoint callback ran. ``extras`` is a free-form dict
    that round-trips into events.
    """

    model: Any
    best_checkpoint: Path | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricReport:
    """Return value of Evaluator.evaluate()."""

    task: str
    n_samples: int
    duration_seconds: float
    metrics: dict[str, float]
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: dict[str, Any] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Serialize into the wire shape of metrics.json (schema_version=1)."""
        return {
            "schema_version": 1,
            "task": self.task,
            "n_samples": self.n_samples,
            "duration_seconds": self.duration_seconds,
            "metrics": dict(self.metrics),
            "per_class": dict(self.per_class),
            "confusion_matrix": dict(self.confusion_matrix),
            "extras": dict(self.extras),
        }
