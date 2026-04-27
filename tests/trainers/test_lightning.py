"""LightningTrainer trains a minimal LightningModule on a tiny CPU dataset."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pytest
import torch
from torch import nn

from maldet.trainers.lightning_trainer import LightningTrainer, _materialize_tensor
from maldet.types import Sample


class DummyReader:
    def __init__(self, items: list[tuple[str, str]]) -> None:
        self._items = items

    def __iter__(self) -> Iterator[Sample]:
        for sha, label in self._items:
            yield Sample(sha256=sha, path=Path("/tmp") / sha, label=label)

    def __len__(self) -> int:
        return len(self._items)


class DummyExtractor:
    output_shape = (4,)
    dtype = "float32"

    def extract(self, sample: Sample) -> np.ndarray:
        if sample.label == "Malware":
            return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


class TinyMLP(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.net(x)
        loss = self.loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=1e-2)


class RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        self.events.append(("metric", {"name": name, "value": value, "step": step}))

    def log_params(self, params: dict[str, Any]) -> None:
        self.events.append(("params", dict(params)))

    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None:
        self.events.append(("artifact", {"path": str(path), "artifact_path": artifact_path}))

    def log_event(self, kind: str, **payload: Any) -> None:
        self.events.append((kind, payload))

    def set_tags(self, tags: dict[str, str]) -> None:
        self.events.append(("tags", dict(tags)))


@pytest.mark.parametrize("max_epochs", [1])
def test_fit_runs_on_cpu(max_epochs: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MALDET_GPU_COUNT", "0")
    logger = RecordingLogger()
    trainer = LightningTrainer(max_epochs=max_epochs, default_root_dir=str(tmp_path))
    items = [(f"{i:064x}", "Malware" if i % 2 else "Benign") for i in range(16)]
    result = trainer.fit(TinyMLP(), DummyReader(items), DummyExtractor(), logger=logger)
    kinds = [e[0] for e in logger.events]
    assert "stage_begin" in kinds
    assert "stage_end" in kinds
    assert any(e[0] == "metric" and e[1]["name"] == "train_loss" for e in logger.events)
    assert result.model is not None


def test_save_load_roundtrip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MALDET_GPU_COUNT", "0")
    logger = RecordingLogger()
    trainer = LightningTrainer(max_epochs=1, default_root_dir=str(tmp_path))
    items = [(f"{i:064x}", "Malware" if i % 2 else "Benign") for i in range(8)]
    result = trainer.fit(TinyMLP(), DummyReader(items), DummyExtractor(), logger=logger)
    out = tmp_path / "model"
    out.mkdir()
    trainer.save(result, out)
    fresh = TinyMLP()
    loaded = trainer.load(out, model_factory=lambda: fresh)
    x = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    assert loaded(x).shape == (1, 2)


class FailingExtractor:
    """Raises ValueError for samples in ``fail_shas``; returns dummy features otherwise.

    Mirrors the real-world Phase 11 case where ELF samples lacking ``.text`` cause
    Text256Extractor.extract to raise ValueError.
    """

    output_shape = (4,)
    dtype = "float32"

    def __init__(self, fail_shas: set[str]) -> None:
        self._fail = fail_shas

    def extract(self, sample: Sample) -> np.ndarray:
        if sample.sha256 in self._fail:
            raise ValueError(f"simulated extractor failure on {sample.sha256}")
        return np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)


def test_materialize_tensor_skips_samples_when_extractor_raises_value_error() -> None:
    items = [(f"{i:064x}", "Malware" if i % 2 else "Benign") for i in range(20)]
    fail_shas = {items[3][0], items[7][0]}

    x_t, y_t = _materialize_tensor(DummyReader(items), FailingExtractor(fail_shas))

    assert x_t.shape[0] == 18
    assert y_t.shape[0] == 18


def test_materialize_tensor_raises_when_more_than_half_samples_fail() -> None:
    items = [(f"{i:064x}", "Malware") for i in range(10)]
    fail_shas = {sha for sha, _ in items[:6]}

    with pytest.raises(RuntimeError, match="too many"):
        _materialize_tensor(DummyReader(items), FailingExtractor(fail_shas))


def test_materialize_tensor_emits_warning_event_per_skip_when_logger_provided() -> None:
    items = [(f"{i:064x}", "Malware") for i in range(10)]
    bad_sha = items[3][0]
    fail_shas = {bad_sha}
    logger = RecordingLogger()

    _materialize_tensor(DummyReader(items), FailingExtractor(fail_shas), logger=logger)

    warnings_emitted = [e for e in logger.events if e[0] == "warning"]
    assert len(warnings_emitted) == 1
    payload = warnings_emitted[0][1]
    assert "message" in payload
    assert bad_sha in payload["message"]
