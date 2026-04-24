"""Stdout event logger — prefixed JSON line per event."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_PREFIX = "maldet.event: "


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class StdoutEventLogger:
    """Writes ``maldet.event: {json}\\n`` lines to stdout."""

    def _write(self, record: dict[str, Any]) -> None:
        record = {"ts": _now(), **record}
        sys.stdout.write(_PREFIX + json.dumps(record, default=str) + "\n")
        sys.stdout.flush()

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        rec: dict[str, Any] = {"kind": "metric", "name": name, "value": value}
        if step is not None:
            rec["step"] = step
        self._write(rec)

    def log_params(self, params: dict[str, Any]) -> None:
        self._write({"kind": "params", "params": dict(params)})

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        self._write({"kind": "artifact_written", "path": str(path), "artifact_path": artifact_path})

    def log_event(self, kind: str, **payload: Any) -> None:
        self._write({"kind": kind, **payload})

    def set_tags(self, tags: dict[str, str]) -> None:
        self._write({"kind": "tags", "tags": dict(tags)})
