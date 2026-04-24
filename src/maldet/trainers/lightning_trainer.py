"""Lightning-based Trainer for deep-learning detectors.

Reads the platform-injected env vars ``MALDET_GPU_COUNT`` and
``MALDET_DISTRIBUTED_STRATEGY`` to pick ``accelerator``, ``devices``, and
``strategy`` for ``lightning.Trainer``.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from maldet.protocols import EventLogger, FeatureExtractor, SampleReader
from maldet.types import TrainResult


def _accelerator_and_devices() -> tuple[str, int]:
    count = int(os.environ.get("MALDET_GPU_COUNT", "0"))
    if count <= 0:
        # Hide GPUs from Lightning/CUDA so isolate_rng does not attempt
        # cuda.get_rng_state_all() even when a GPU is physically present.
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        return "cpu", 1
    if not torch.cuda.is_available():
        return "cpu", 1
    return "gpu", count


def _strategy_from_env(device_count: int) -> str:
    strat = os.environ.get("MALDET_DISTRIBUTED_STRATEGY", "").lower()
    if strat in {"ddp", "fsdp", "deepspeed"}:
        return strat
    return "auto" if device_count <= 1 else "ddp"


def _materialize_tensor(
    reader: SampleReader, extractor: FeatureExtractor
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for sample in reader:
        xs.append(extractor.extract(sample))
        if sample.label is None:
            raise ValueError("LightningTrainer: unlabeled sample encountered during fit")
        ys.append(1 if sample.label == "Malware" else 0)
    if not xs:
        raise RuntimeError("LightningTrainer: reader yielded zero samples")
    X = np.stack(xs)  # noqa: N806
    if str(X.dtype) == "uint8":
        x_t = torch.from_numpy(X.astype(np.int64))
    else:
        x_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.tensor(ys, dtype=torch.int64)
    return x_t, y_t


class MaldetLightningLogger(pl.loggers.Logger):
    """Adapter from Lightning's Logger API onto maldet EventLogger."""

    def __init__(self, delegate: EventLogger) -> None:
        super().__init__()
        self._delegate = delegate

    @property
    def name(self) -> str:
        return "maldet"

    @property
    def version(self) -> str:
        return "1"

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        for k, v in metrics.items():
            try:
                self._delegate.log_metric(k, float(v), step=step)
            except (TypeError, ValueError):
                continue

    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        try:
            d = dict(params) if isinstance(params, dict) else getattr(params, "__dict__", {})
        except Exception:
            d = {}
        if d:
            self._delegate.log_params({k: str(v) for k, v in d.items()})


class MaldetProgressCallback(Callback):
    """Emits ``epoch_begin`` / ``epoch_end`` events through the EventLogger."""

    def __init__(self, delegate: EventLogger) -> None:
        self._delegate = delegate
        self._epoch_started: float | None = None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._epoch_started = time.time()
        self._delegate.log_event("epoch_begin", epoch=int(trainer.current_epoch))

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        dur = time.time() - (self._epoch_started or time.time())
        self._delegate.log_event("epoch_end", epoch=int(trainer.current_epoch), duration_s=dur)


class LightningTrainer:
    """PyTorch Lightning-based Trainer."""

    def __init__(
        self,
        *,
        max_epochs: int = 10,
        batch_size: int = 32,
        monitor: str = "train_loss",
        patience: int = 5,
        save_top_k: int = 1,
        default_root_dir: str | None = None,
    ) -> None:
        self._max_epochs = max_epochs
        self._batch_size = batch_size
        self._monitor = monitor
        self._patience = patience
        self._save_top_k = save_top_k
        self._default_root_dir = default_root_dir

    def fit(
        self,
        model: pl.LightningModule,
        train: SampleReader,
        extractor: FeatureExtractor,
        *,
        val: SampleReader | None = None,
        logger: EventLogger,
    ) -> TrainResult:
        logger.log_event("stage_begin", stage="train")
        acc, dev = _accelerator_and_devices()
        strategy = _strategy_from_env(dev)

        X, y = _materialize_tensor(train, extractor)  # noqa: N806
        logger.log_event("data_loaded", n_train=int(X.shape[0]))
        train_dl = DataLoader(TensorDataset(X, y), batch_size=self._batch_size, shuffle=True)

        val_dl: DataLoader[tuple[torch.Tensor, ...]] | None = None
        if val is not None:
            Xv, yv = _materialize_tensor(val, extractor)  # noqa: N806
            val_dl = DataLoader(TensorDataset(Xv, yv), batch_size=self._batch_size)

        ckpt_dir = Path(self._default_root_dir or ".") / "checkpoints"
        callbacks: list[Callback] = [
            ModelCheckpoint(
                dirpath=str(ckpt_dir), save_top_k=self._save_top_k, monitor=self._monitor
            ),
            MaldetProgressCallback(logger),
        ]
        if val_dl is not None:
            callbacks.append(EarlyStopping(monitor=self._monitor, patience=self._patience))

        pl_logger = MaldetLightningLogger(logger)
        pl_trainer = pl.Trainer(
            max_epochs=self._max_epochs,
            accelerator=acc,
            devices=dev,
            strategy=strategy,
            logger=pl_logger,
            callbacks=callbacks,
            enable_progress_bar=False,
            enable_model_summary=False,
            default_root_dir=self._default_root_dir,
            log_every_n_steps=1,
        )
        pl_trainer.fit(model, train_dl, val_dl)

        best = None
        for cb in callbacks:
            if isinstance(cb, ModelCheckpoint) and cb.best_model_path:
                best = Path(cb.best_model_path)
                break

        logger.log_event("stage_end", stage="train", status="success")
        return TrainResult(model=model, best_checkpoint=best)

    def save(self, result: TrainResult, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / "model.ckpt"
        if result.best_checkpoint is not None and result.best_checkpoint.exists():
            shutil.copy2(result.best_checkpoint, target)
            return
        module = result.model
        torch.save({"state_dict": module.state_dict()}, target)

    def load(
        self, model_dir: Path, *, model_factory: Callable[[], pl.LightningModule] | None = None
    ) -> pl.LightningModule:
        ckpt = model_dir / "model.ckpt"
        if model_factory is None:
            raise ValueError("LightningTrainer.load requires model_factory to rebuild the module")
        module = model_factory()
        state = torch.load(ckpt, map_location="cpu")
        if "state_dict" in state:
            module.load_state_dict(state["state_dict"])
        else:
            module.load_state_dict(state)
        return module
