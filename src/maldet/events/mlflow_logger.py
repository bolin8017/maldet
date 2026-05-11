"""MLflow-backed event logger with kind-aware routing.

Spec §5.2 (docs/superpowers/specs/2026-05-11-mlflow-data-model-redesign-design.md) —
structured payloads (``confusion_matrix``, ``per_class``) become ``log_dict``
artifacts; line-stream events (``warning``, ``error``) are buffered in-memory
and flushed as ``*.jsonl`` artifacts on ``close()``; scalar fields become
metrics or tags depending on shape. MLflow is a soft dependency — install
``maldet[mlflow]`` to enable.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any


def _try_import_mlflow() -> Any:
    try:
        import mlflow

        return mlflow
    except ImportError:
        return None


class MlflowEventLogger:
    def __init__(self, mlflow: Any = None) -> None:
        self._mlflow = mlflow if mlflow is not None else _try_import_mlflow()
        self._warning_buf: list[dict[str, Any]] = []
        self._error_buf: list[dict[str, Any]] = []

    def _available(self) -> bool:
        return self._mlflow is not None

    # ---------- scalar / param / artifact passthrough ----------

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        if not self._available():
            return
        self._mlflow.log_metric(name, value, step=step)

    def log_params(self, params: dict[str, Any]) -> None:
        if not self._available():
            return
        self._mlflow.log_params(dict(params))

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        if not self._available():
            return
        if path.is_dir():
            self._mlflow.log_artifacts(str(path), artifact_path=artifact_path)
        else:
            self._mlflow.log_artifact(str(path), artifact_path=artifact_path)

    def set_tags(self, tags: dict[str, str]) -> None:
        if not self._available():
            return
        self._mlflow.set_tags(dict(tags))

    # ---------- kind-aware event routing ----------

    def log_event(self, kind: str, **payload: Any) -> None:
        if not self._available() or kind == "metric":
            return
        handler = _EVENT_HANDLERS.get(kind, _handle_generic_tag)
        handler(self._mlflow, kind, payload, self)

    # ---------- model logging ----------

    def log_model(
        self,
        model: Any,
        flavor: str,
        artifact_path: str = "model",
        signature: Any = None,
        input_example: Any = None,
        pip_requirements: list[str] | None = None,
    ) -> None:
        if not self._available():
            return
        if flavor == "sklearn":
            self._mlflow.sklearn.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                pip_requirements=pip_requirements,
            )
        elif flavor == "pytorch":
            self._mlflow.pytorch.log_model(
                model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                pip_requirements=pip_requirements,
            )
        elif flavor == "pyfunc":
            self._mlflow.pyfunc.log_model(
                python_model=model,
                artifact_path=artifact_path,
                signature=signature,
                input_example=input_example,
                pip_requirements=pip_requirements,
            )
        else:
            raise ValueError(f"unknown mlflow flavor: {flavor!r}")

    # ---------- lifecycle ----------

    def close(self) -> None:
        """Flush buffered line-stream events to MLflow as JSONL artifacts."""
        if not self._available():
            return
        if self._warning_buf:
            self._mlflow.log_text(
                "\n".join(json.dumps(w, default=str) for w in self._warning_buf),
                "warnings.jsonl",
            )
            self._mlflow.log_metric("maldet/warnings_total", float(len(self._warning_buf)))
        if self._error_buf:
            self._mlflow.log_text(
                "\n".join(json.dumps(e, default=str) for e in self._error_buf),
                "errors.jsonl",
            )
            self._mlflow.log_metric("maldet/errors_total", float(len(self._error_buf)))


# ---------- event handlers (module-level for testability) ----------


def _handle_stage_begin(
    mlflow: Any, kind: str, payload: dict[str, Any], logger: MlflowEventLogger
) -> None:
    if "stage" in payload:
        mlflow.set_tag("maldet.stage", str(payload["stage"]))
    mlflow.set_tag("maldet.stage_begin_ts", str(time.time()))


def _handle_stage_end(
    mlflow: Any, kind: str, payload: dict[str, Any], logger: MlflowEventLogger
) -> None:
    if "stage" in payload:
        mlflow.set_tag("maldet.stage_end", str(payload["stage"]))
    if "status" in payload:
        mlflow.set_tag("maldet.status", str(payload["status"]))


def _handle_data_loaded(
    mlflow: Any, kind: str, payload: dict[str, Any], logger: MlflowEventLogger
) -> None:
    for k, v in payload.items():
        try:
            mlflow.log_metric(f"maldet/{k}", float(v))
        except (TypeError, ValueError):
            mlflow.set_tag(f"maldet.data.{k}", str(v))


def _handle_warning(
    mlflow: Any, kind: str, payload: dict[str, Any], logger: MlflowEventLogger
) -> None:
    logger._warning_buf.append({"ts": time.time(), **payload})


def _handle_error(
    mlflow: Any, kind: str, payload: dict[str, Any], logger: MlflowEventLogger
) -> None:
    logger._error_buf.append({"ts": time.time(), **payload})


def _handle_confusion_matrix(
    mlflow: Any, kind: str, payload: dict[str, Any], logger: MlflowEventLogger
) -> None:
    mlflow.log_dict(
        {"labels": payload["labels"], "matrix": payload["matrix"]},
        "confusion_matrix.json",
    )


def _handle_per_class(
    mlflow: Any, kind: str, payload: dict[str, Any], logger: MlflowEventLogger
) -> None:
    per_class = payload["per_class"]
    mlflow.log_dict(per_class, "per_class_metrics.json")
    for cls, metrics in per_class.items():
        if not isinstance(metrics, dict):
            continue
        for name, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"per_class/{cls}/{name}", float(v))


def _handle_artifact_written(
    mlflow: Any, kind: str, payload: dict[str, Any], logger: MlflowEventLogger
) -> None:
    path = payload.get("path", "")
    name = Path(path).name if path else "unknown"
    if path:
        mlflow.set_tag(f"maldet.artifact.{name}", str(path))
    if "size_bytes" in payload:
        with contextlib.suppress(TypeError, ValueError):
            mlflow.log_metric(f"maldet/artifact_bytes/{name}", float(payload["size_bytes"]))


def _handle_checkpoint_saved(
    mlflow: Any, kind: str, payload: dict[str, Any], logger: MlflowEventLogger
) -> None:
    _handle_artifact_written(mlflow, kind, payload, logger)


def _handle_generic_tag(
    mlflow: Any, kind: str, payload: dict[str, Any], logger: MlflowEventLogger
) -> None:
    """Fallback for forward compat — scalar fields become scoped tags."""
    for k, v in payload.items():
        if isinstance(v, (str, int, float, bool)):
            mlflow.set_tag(f"maldet.{kind}.{k}", str(v))


_EVENT_HANDLERS: dict[str, Callable[[Any, str, dict[str, Any], MlflowEventLogger], None]] = {
    "stage_begin": _handle_stage_begin,
    "stage_end": _handle_stage_end,
    "data_loaded": _handle_data_loaded,
    "warning": _handle_warning,
    "error": _handle_error,
    "confusion_matrix": _handle_confusion_matrix,
    "per_class": _handle_per_class,
    "artifact_written": _handle_artifact_written,
    "checkpoint_saved": _handle_checkpoint_saved,
    "epoch_begin": _handle_generic_tag,
    "epoch_end": _handle_generic_tag,
}
