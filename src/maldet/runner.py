"""StageRunner — loads manifest, composes layers via manifest symbols, drives stage."""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import inspect
import json
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from maldet.events.jsonl import JsonlEventLogger
from maldet.events.logger import CompositeEventLogger
from maldet.events.mlflow_logger import MlflowEventLogger
from maldet.events.stdout import StdoutEventLogger
from maldet.manifest import DetectorManifest, load_manifest, search_manifest

_log = logging.getLogger(__name__)

# Hydra meta-fields that callers (esp. lolday's params guard) reject as RCE
# vectors — drop them silently from cfg.model kwargs so legacy YAML configs that
# still carry a stale `_target_` don't pass it through to the model factory.
_HYDRA_META_KEYS = ("_target_", "_partial_", "_args_", "_recursive_", "_convert_")


def _load_symbol(dotted: str) -> Any:
    """Resolve ``module.sub:attribute`` into the attribute object."""
    if ":" not in dotted:
        raise ValueError(f"expected 'module:attribute', got {dotted!r}")
    mod_name, attr = dotted.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, attr)


def _require(value: str | None, what: str) -> str:
    if value is None:
        raise ValueError(f"stage missing required symbol: {what}")
    return value


def _model_kwargs(cfg: DictConfig) -> dict[str, Any]:
    """Extract cfg.model as a plain kwargs dict, dropping Hydra meta-fields."""
    raw = cfg.get("model")
    if raw is None:
        return {}
    container = OmegaConf.to_container(raw, resolve=True)
    if not isinstance(container, dict):
        raise ValueError(f"cfg.model must be a mapping, got {type(container).__name__}")
    result: dict[str, Any] = {}
    for k, v in container.items():
        if isinstance(k, str) and k not in _HYDRA_META_KEYS:
            result[k] = v
    return result


def _load_with_optional_factory(trainer: Any, source_model: Path, factory: Any) -> Any:
    """Call ``trainer.load(source_model)``, threading ``model_factory`` if accepted.

    LightningTrainer.load needs ``model_factory`` to rebuild the LightningModule
    around the saved state dict, while SklearnTrainer.load doesn't take any
    kwargs (the joblib payload carries the class). Inspecting the signature
    keeps the runner compatible with both without baking in trainer-specific
    branching.
    """
    if factory is None:
        return trainer.load(source_model)
    try:
        params = inspect.signature(trainer.load).parameters
    except (TypeError, ValueError):
        return trainer.load(source_model)
    if "model_factory" in params:
        return trainer.load(source_model, model_factory=factory)
    return trainer.load(source_model)


def _log_dataset_input(cfg: DictConfig | None, stage: str, csv_path: Path) -> None:
    """Emit ``mlflow.log_input`` for the dataset consumed by ``stage``.

    No-ops when mlflow isn't importable, there is no active run, or the CSV
    can't be loaded. ``cfg.lolday.{train,test,predict}_dataset_id`` may provide
    a stable platform-side ID; falls back to ``"unknown"`` otherwise.
    """
    try:
        import mlflow
        import mlflow.data
        import pandas as pd
    except ImportError:
        return
    if mlflow.active_run() is None:
        return
    if not csv_path.exists():
        return
    try:
        df = pd.read_csv(csv_path)
    except Exception as exc:
        _log.warning("dataset_input read failed: %s", exc)
        return

    lolday_meta: dict[str, Any] = {}
    if cfg is not None and hasattr(cfg, "get"):
        raw = cfg.get("lolday")
        if raw is not None:
            try:
                lolday_meta = dict(raw) if not isinstance(raw, dict) else raw
            except Exception:
                lolday_meta = {}

    key_map = {
        "train": "train_dataset_id",
        "evaluate": "test_dataset_id",
        "predict": "predict_dataset_id",
    }
    ds_id = lolday_meta.get(key_map.get(stage, ""), "unknown")
    digest = hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()[:16]
    try:
        # mlflow.data.from_pandas is the canonical mlflow >= 2.4 entry; the
        # mlflow type stubs don't surface it on the package, so resolve at
        # runtime via getattr to bypass mypy attr-defined.
        from_pandas = getattr(mlflow.data, "from_pandas", None)
        if from_pandas is None:
            return
        ds = from_pandas(
            df=df,
            source=str(csv_path),
            name=f"{stage}_{ds_id}",
            digest=digest,
        )
        context_map = {
            "train": "training",
            "evaluate": "evaluation",
            "predict": "prediction",
        }
        mlflow.log_input(ds, context=context_map.get(stage, stage))
    except Exception as exc:
        # log_input is best-effort lineage; don't bring down the stage if it fails.
        _log.warning("mlflow.log_input failed: %s", exc)


def _first_sample_features(reader: Any, extractor: Any, *, n: int = 5) -> np.ndarray | None:
    """Pull the first ``n`` successfully-extracted feature vectors as a 2-D array.

    Used by the train branch of ``_run_stage`` to build an ``input_example``
    for MLflow Models signature inference. Re-iterating the reader after
    ``trainer.fit`` consumed it is acceptable: readers in maldet are designed
    to be replayed (file-backed iterators), and only a handful of samples are
    needed.
    """
    out: list[np.ndarray] = []
    for sample in reader:
        try:
            out.append(extractor.extract(sample))
        except Exception:
            continue
        if len(out) >= n:
            break
    if not out:
        return None
    return np.stack(out)


class StageRunner:
    """Orchestrates a single stage (train / evaluate / predict)."""

    def __init__(self, manifest: DetectorManifest | None = None) -> None:
        if manifest is None:
            manifest = load_manifest(search_manifest())
        self._manifest = manifest

    def run(self, *, stage: str, config_path: Path) -> None:
        if stage not in self._manifest.lifecycle.stages:
            raise ValueError(f"stage {stage!r} not declared in manifest.lifecycle.stages")
        stage_spec = self._manifest.stages.get(stage)
        if stage_spec is None:
            raise ValueError(f"no stages.{stage} block in maldet.toml")

        cfg = OmegaConf.load(config_path)
        assert isinstance(cfg, DictConfig)

        with self._pinned_mlflow_run(cfg):
            self._run_stage(stage, stage_spec, cfg)

    def _run_stage(self, stage: str, stage_spec: Any, cfg: DictConfig) -> None:
        output_dir = Path(str(cfg.paths.output_dir))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write manifest.json alongside artifacts for provenance (spec §2B).
        (output_dir / "manifest.json").write_text(
            json.dumps(self._manifest.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )

        logger = CompositeEventLogger(
            [
                JsonlEventLogger(output_dir / "events.jsonl"),
                StdoutEventLogger(),
                MlflowEventLogger(),
            ]
        )
        try:
            self._dispatch_stage(stage, stage_spec, cfg, logger, output_dir)
        finally:
            # Flush buffered events (warnings.jsonl / errors.jsonl) before the
            # pinned mlflow run is ended by _pinned_mlflow_run.
            with contextlib.suppress(Exception):
                logger.close()

    def _dispatch_stage(
        self,
        stage: str,
        stage_spec: Any,
        cfg: DictConfig,
        logger: CompositeEventLogger,
        output_dir: Path,
    ) -> None:
        reader_cls = _load_symbol(_require(stage_spec.reader, "reader"))
        extractor_cls = _load_symbol(_require(stage_spec.extractor, "extractor"))

        if stage == "train":
            train_csv = Path(str(cfg.data.train_csv))
            _log_dataset_input(cfg, "train", train_csv)
            samples_root = Path(str(cfg.paths.samples_root))
            reader = reader_cls(csv=train_csv, samples_root=samples_root)
            extractor = extractor_cls()

            model_factory = _load_symbol(_require(stage_spec.model, "model"))
            model = model_factory(**_model_kwargs(cfg))
            trainer_cls = _load_symbol(_require(stage_spec.trainer, "trainer"))
            trainer = trainer_cls()
            result = trainer.fit(
                model,
                reader,
                extractor,
                classes=self._manifest.output.classes,
                logger=logger,
            )
            # Re-iterate the reader for a tiny signature sample. Readers in
            # maldet are file-backed and replayable, so this is cheap and
            # avoids leaking sample arrays out of trainer.fit's contract.
            sig_reader = reader_cls(csv=train_csv, samples_root=samples_root)
            sig_sample = _first_sample_features(sig_reader, extractor, n=5)
            trainer.save(
                result,
                output_dir / "model",
                logger=logger,
                signature_input_sample=sig_sample,
            )
            return

        if stage == "evaluate":
            source_model = Path(str(cfg.paths.source_model))
            train_spec = self._manifest.stages.get("train")
            trainer_symbol = stage_spec.trainer or (train_spec.trainer if train_spec else None)
            trainer = _load_symbol(_require(trainer_symbol, "trainer"))()
            model_symbol = stage_spec.model or (train_spec.model if train_spec else None)
            factory = _load_symbol(model_symbol) if model_symbol else None
            model = _load_with_optional_factory(trainer, source_model, factory)
            test_csv = Path(str(cfg.data.test_csv))
            _log_dataset_input(cfg, "evaluate", test_csv)
            samples_root = Path(str(cfg.paths.samples_root))
            reader = reader_cls(csv=test_csv, samples_root=samples_root)
            extractor = extractor_cls()
            evaluator_cls = _load_symbol(_require(stage_spec.evaluator, "evaluator"))
            evaluator = evaluator_cls(
                positive_class=self._manifest.output.positive_class,
                class_names=self._manifest.output.classes,
            )
            report = evaluator.evaluate(model, reader, extractor, logger=logger)
            metrics_path = output_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(report.to_json_dict(), indent=2, default=str), encoding="utf-8"
            )
            logger.log_artifact(metrics_path)
            return

        if stage == "predict":
            source_model = Path(str(cfg.paths.source_model))
            train_spec = self._manifest.stages.get("train")
            trainer_symbol = stage_spec.trainer or (train_spec.trainer if train_spec else None)
            trainer = _load_symbol(_require(trainer_symbol, "trainer"))()
            model_symbol = stage_spec.model or (train_spec.model if train_spec else None)
            factory = _load_symbol(model_symbol) if model_symbol else None
            model = _load_with_optional_factory(trainer, source_model, factory)
            predict_csv = Path(str(cfg.data.predict_csv))
            _log_dataset_input(cfg, "predict", predict_csv)
            samples_root = Path(str(cfg.paths.samples_root))
            reader = reader_cls(csv=predict_csv, samples_root=samples_root)
            extractor = extractor_cls()
            predictor_cls = _load_symbol(_require(stage_spec.predictor, "predictor"))
            predictor = predictor_cls(class_names=self._manifest.output.classes)
            predictions_path = output_dir / "predictions.csv"
            predictor.predict(model, reader, extractor, out_path=predictions_path, logger=logger)
            logger.log_artifact(predictions_path)
            return

        raise ValueError(f"unhandled stage: {stage}")

    @staticmethod
    @contextlib.contextmanager
    def _pinned_mlflow_run(cfg: DictConfig) -> Iterator[None]:
        """Pin the active MLflow run to the platform-injected ``cfg.mlflow.run_id``.

        Lolday creates the MLflow run before the stage container starts and
        writes ``(tracking_uri, run_id, experiment_id)`` into ``cfg.mlflow``.
        Without an explicit pin, top-level ``mlflow.*`` calls — including those
        emitted from PyTorch Lightning DDP subprocess workers and the model
        artifact upload at stage end — race and auto-create a fresh untagged
        run instead of resuming the platform's. We've observed this leak as a
        second "learned-stag-591"-style auto-named run appearing alongside the
        platform's tagged run for every multi-GPU train.

        Behaviour:

        - If ``cfg.mlflow`` is absent (offline / dev YAML), no-op.
        - If ``mlflow`` is not importable (``maldet`` installed without the
          ``mlflow`` extra), no-op.
        - Otherwise: set tracking URI + experiment + run via both the SDK and
          ``MLFLOW_*`` env vars. The env vars cover DDP subprocess workers
          spawned by Lightning, which inherit the parent's environment.
        """
        if cfg is None:
            yield
            return
        mlflow_cfg = cfg.get("mlflow") if isinstance(cfg, DictConfig) else None
        if mlflow_cfg is None:
            yield
            return
        try:
            import mlflow
        except ImportError:
            yield
            return
        tracking_uri = mlflow_cfg.get("tracking_uri")
        experiment_id = mlflow_cfg.get("experiment_id")
        run_id = mlflow_cfg.get("run_id")
        if tracking_uri:
            tracking_uri = str(tracking_uri)
            mlflow.set_tracking_uri(tracking_uri)
            os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
        if experiment_id:
            os.environ["MLFLOW_EXPERIMENT_ID"] = str(experiment_id)
        if not run_id:
            yield
            return
        run_id = str(run_id)
        os.environ["MLFLOW_RUN_ID"] = run_id
        mlflow.start_run(run_id=run_id)
        try:
            yield
        finally:
            if mlflow.active_run() is not None:
                mlflow.end_run()
