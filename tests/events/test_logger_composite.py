"""CompositeEventLogger fans out to every delegate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from maldet.events.logger import CompositeEventLogger


class RecordingLogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        self.calls.append(("log_metric", (name, value, step), {}))

    def log_params(self, params: dict[str, Any]) -> None:
        self.calls.append(("log_params", (), {"params": params}))

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        self.calls.append(("log_artifact", (path, artifact_path), {}))

    def log_event(self, kind: str, **payload: Any) -> None:
        self.calls.append(("log_event", (kind,), payload))

    def set_tags(self, tags: dict[str, str]) -> None:
        self.calls.append(("set_tags", (), {"tags": tags}))


def test_fanout_all_methods() -> None:
    a = RecordingLogger()
    b = RecordingLogger()
    comp = CompositeEventLogger([a, b])

    comp.log_metric("loss", 0.1, step=1)
    comp.log_params({"lr": 1e-3})
    comp.log_artifact(Path("/tmp/x"), artifact_path="model")
    comp.log_event("metric", name="loss", value=0.1)
    comp.set_tags({"gpu_count": "2"})

    for rec in (a, b):
        assert [c[0] for c in rec.calls] == [
            "log_metric",
            "log_params",
            "log_artifact",
            "log_event",
            "set_tags",
        ]


def test_one_delegate_failing_does_not_stop_others() -> None:
    class Boom:
        def log_metric(self, *a: Any, **k: Any) -> None:
            raise RuntimeError("boom")

        log_params = log_artifact = log_event = set_tags = log_metric

    a = Boom()
    b = RecordingLogger()
    comp = CompositeEventLogger([a, b])
    comp.log_metric("loss", 0.1)  # must not raise
    assert b.calls[0][0] == "log_metric"
