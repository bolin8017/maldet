"""CompositeEventLogger — fans out to N loggers, isolating failures."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from maldet.protocols import EventLogger

_log = logging.getLogger(__name__)


class CompositeEventLogger:
    """Forwards every call to every wrapped logger.

    A delegate raising is caught and logged at WARNING; other delegates still run.
    This protects the training loop from a broken MLflow / filesystem from killing
    the detector run.
    """

    def __init__(self, delegates: Sequence[EventLogger]) -> None:
        self._delegates = list(delegates)

    def _fanout(self, method: str, *args: Any, **kwargs: Any) -> None:
        for d in self._delegates:
            try:
                getattr(d, method)(*args, **kwargs)
            except Exception as exc:  # isolation is the point
                _log.warning(
                    "event_delegate_failed",
                    extra={"method": method, "delegate": type(d).__name__, "error": str(exc)},
                )

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        self._fanout("log_metric", name, value, step)

    def log_params(self, params: dict[str, Any]) -> None:
        self._fanout("log_params", params)

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        self._fanout("log_artifact", path, artifact_path)

    def log_event(self, kind: str, **payload: Any) -> None:
        self._fanout("log_event", kind, **payload)

    def set_tags(self, tags: dict[str, str]) -> None:
        self._fanout("set_tags", tags)
