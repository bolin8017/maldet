"""MLflow-backed event logger.

Keeps the ``mlflow`` import optional — if the caller passes ``mlflow=None`` and
``mlflow`` is not importable, ``log_*`` methods silently no-op. This makes MLflow
a soft dependency (install ``maldet[mlflow]`` to enable).
"""

from __future__ import annotations

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

    def _available(self) -> bool:
        return self._mlflow is not None

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

    def log_event(self, kind: str, **payload: Any) -> None:
        """Flatten event payload into tags named ``maldet.<kind>.<field>``.

        Metric events are NOT forwarded here — the caller calls ``log_metric`` for
        those.
        """
        if not self._available() or kind == "metric":
            return
        for k, v in payload.items():
            self._mlflow.set_tag(f"maldet.{kind}.{k}", str(v))

    def set_tags(self, tags: dict[str, str]) -> None:
        if not self._available():
            return
        self._mlflow.set_tags(dict(tags))
