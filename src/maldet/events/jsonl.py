"""JSONL event logger — append-only, fsync per event."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class JsonlEventLogger:
    """Writes one NDJSON line per event to ``path``.

    Each write is followed by ``os.fsync`` so a pod kill does not lose events in the
    page cache. Parent directory is created if missing.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.touch(exist_ok=True)

    def _write(self, record: dict[str, Any]) -> None:
        record = {"ts": _now(), **record}
        line = json.dumps(record, default=str)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        record: dict[str, Any] = {"kind": "metric", "name": name, "value": value}
        if step is not None:
            record["step"] = step
        self._write(record)

    def log_params(self, params: dict[str, Any]) -> None:
        self._write({"kind": "params", "params": dict(params)})

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        self._write(
            {
                "kind": "artifact_written",
                "path": str(path),
                "artifact_path": artifact_path,
            }
        )

    def log_event(self, kind: str, **payload: Any) -> None:
        self._write({"kind": kind, **payload})

    def set_tags(self, tags: dict[str, str]) -> None:
        self._write({"kind": "tags", "tags": dict(tags)})
