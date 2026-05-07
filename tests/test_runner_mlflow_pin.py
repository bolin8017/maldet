"""StageRunner._pinned_mlflow_run pins the active MLflow run.

Without this guard, top-level ``mlflow.*`` API calls auto-create a fresh
untagged run on first invocation, producing a "learned-stag-591"-style
orphan that runs alongside the platform's properly-tagged run. The pin
resumes the platform-injected ``cfg.mlflow.run_id`` and exports the
``MLFLOW_*`` env vars so PyTorch Lightning DDP subprocess workers inherit
them.
"""

from __future__ import annotations

import os
from typing import Any

import mlflow
import pytest
from omegaconf import OmegaConf

from maldet.runner import StageRunner


def _platform_run(tracking_uri: str) -> tuple[str, str]:
    """Create a registered run inside an experiment, return (exp_id, run_id)."""
    mlflow.set_tracking_uri(tracking_uri)
    exp_id = mlflow.create_experiment("platform/elf-cnn/v4.0.0")
    run = mlflow.MlflowClient(tracking_uri).create_run(experiment_id=exp_id)
    return exp_id, run.info.run_id


@pytest.fixture
def reset_mlflow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip MLFLOW_* env + end any leaked active run from prior tests so each
    test starts from a clean global mlflow state."""
    for var in ("MLFLOW_TRACKING_URI", "MLFLOW_EXPERIMENT_ID", "MLFLOW_RUN_ID"):
        monkeypatch.delenv(var, raising=False)
    while mlflow.active_run() is not None:
        mlflow.end_run()


def test_pin_resumes_platform_run_and_sets_env(tmp_path: Any, reset_mlflow_env: None) -> None:
    tracking_uri = f"file://{tmp_path}"
    exp_id, run_id = _platform_run(tracking_uri)

    cfg = OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": tracking_uri,
                "run_id": run_id,
                "experiment_id": exp_id,
            }
        }
    )

    captured: dict[str, str | None] = {}
    with StageRunner._pinned_mlflow_run(cfg):
        active = mlflow.active_run()
        captured["active_run_id"] = active.info.run_id if active else None
        captured["env_run_id"] = os.environ.get("MLFLOW_RUN_ID")
        captured["env_exp_id"] = os.environ.get("MLFLOW_EXPERIMENT_ID")
        captured["env_tracking"] = os.environ.get("MLFLOW_TRACKING_URI")

    assert captured["active_run_id"] == run_id
    assert captured["env_run_id"] == run_id
    assert captured["env_exp_id"] == exp_id
    assert captured["env_tracking"] == tracking_uri
    # On exit, the run must be ended so a stale active run doesn't leak into
    # the next stage / process.
    assert mlflow.active_run() is None


def test_pin_is_noop_when_cfg_has_no_mlflow_section(reset_mlflow_env: None) -> None:
    """Legacy YAML / offline dev runs must keep working — no platform run is
    set, no env is touched, no exception."""
    cfg = OmegaConf.create({"stage": "train", "paths": {}})

    with StageRunner._pinned_mlflow_run(cfg):
        assert mlflow.active_run() is None
        assert "MLFLOW_RUN_ID" not in os.environ


def test_pin_log_metric_inside_block_targets_pinned_run(
    tmp_path: Any, reset_mlflow_env: None
) -> None:
    """Top-level ``mlflow.log_metric`` inside the pinned block must write to
    the platform run, not a freshly-auto-created one."""
    tracking_uri = f"file://{tmp_path}"
    exp_id, run_id = _platform_run(tracking_uri)

    cfg = OmegaConf.create(
        {
            "mlflow": {
                "tracking_uri": tracking_uri,
                "run_id": run_id,
                "experiment_id": exp_id,
            }
        }
    )

    with StageRunner._pinned_mlflow_run(cfg):
        mlflow.log_metric("smoke", 0.42)

    client = mlflow.MlflowClient(tracking_uri)
    assert client.get_run(run_id).data.metrics.get("smoke") == 0.42

    # No second auto-created run should exist in the experiment.
    runs = client.search_runs(experiment_ids=[exp_id])
    assert len(runs) == 1, [r.info.run_id for r in runs]
