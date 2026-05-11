"""LightningTrainer.save writes MLflow Models layout; load roundtrips via mlflow.pytorch."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("lightning", reason="LightningTrainer requires the [lightning] extra")
pytest.importorskip("torch", reason="LightningTrainer requires the [lightning] extra")

import lightning.pytorch as pl
import torch
from torch import nn

from maldet.trainers.lightning_trainer import LightningTrainer
from maldet.types import Sample


class _MinimalCNN(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(num_embeddings=256, embedding_dim=4)
        self.fc = nn.Linear(4, 2)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.fc(self.embed(x).mean(dim=1))

    def training_step(self, batch, batch_idx):  # type: ignore[no-untyped-def]
        x, y = batch
        loss = self.loss(self.forward(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):  # type: ignore[no-untyped-def]
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class _DummyReader:
    def __init__(self, n: int) -> None:
        self._n = n

    def __iter__(self) -> Iterator[Sample]:
        for i in range(self._n):
            yield Sample(
                sha256=f"{i:064x}",
                path=Path("/tmp") / f"{i}",
                label="Malware" if i % 2 else "Benign",
            )

    def __len__(self) -> int:
        return self._n


class _DummyExtractor:
    output_shape = (4,)
    dtype = "uint8"

    def extract(self, sample: Sample) -> np.ndarray:
        return np.array([1, 1, 1, 1] if sample.label == "Malware" else [0, 0, 0, 0], dtype=np.uint8)


class _RecordingLogger:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def log_metric(self, name, value, step=None):  # type: ignore[no-untyped-def]
        pass

    def log_params(self, params):  # type: ignore[no-untyped-def]
        pass

    def log_artifact(self, path, artifact_path=None):  # type: ignore[no-untyped-def]
        self.events.append(("artifact", {"path": str(path), "artifact_path": artifact_path}))

    def log_event(self, kind, **payload):  # type: ignore[no-untyped-def]
        pass

    def set_tags(self, tags):  # type: ignore[no-untyped-def]
        pass

    def log_model(self, **kwargs):  # type: ignore[no-untyped-def]
        pass

    def close(self):  # type: ignore[no-untyped-def]
        pass


def test_lightning_save_writes_mlflow_models_layout(tmp_path: Path) -> None:
    logger = _RecordingLogger()
    model = _MinimalCNN()
    trainer = LightningTrainer(max_epochs=1, batch_size=4, default_root_dir=str(tmp_path / "lt"))
    result = trainer.fit(
        model,
        _DummyReader(8),
        _DummyExtractor(),
        classes=["Benign", "Malware"],
        logger=logger,
    )
    out = tmp_path / "model"
    sample_in = torch.zeros(2, 4, dtype=torch.long)
    trainer.save(result, out, logger=logger, signature_input_sample=sample_in)
    assert (out / "MLmodel").exists()


def test_lightning_load_via_mlflow_pytorch(tmp_path: Path) -> None:
    logger = _RecordingLogger()
    model = _MinimalCNN()
    trainer = LightningTrainer(max_epochs=1, batch_size=4, default_root_dir=str(tmp_path / "lt"))
    result = trainer.fit(
        model,
        _DummyReader(8),
        _DummyExtractor(),
        classes=["Benign", "Malware"],
        logger=logger,
    )
    out = tmp_path / "model"
    sample_in = torch.zeros(2, 4, dtype=torch.long)
    trainer.save(result, out, logger=logger, signature_input_sample=sample_in)

    loaded = trainer.load(out)
    pred = loaded(sample_in)
    assert pred.shape == (2, 2)
