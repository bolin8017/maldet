"""StageRunner — loads manifest, composes layers via manifest symbols, drives stage."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from maldet.events.jsonl import JsonlEventLogger
from maldet.events.logger import CompositeEventLogger
from maldet.events.mlflow_logger import MlflowEventLogger
from maldet.events.stdout import StdoutEventLogger
from maldet.manifest import DetectorManifest, load_manifest, search_manifest

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

        reader_cls = _load_symbol(_require(stage_spec.reader, "reader"))
        extractor_cls = _load_symbol(_require(stage_spec.extractor, "extractor"))

        if stage == "train":
            train_csv = Path(str(cfg.data.train_csv))
            samples_root = Path(str(cfg.paths.samples_root))
            reader = reader_cls(csv=train_csv, samples_root=samples_root)
            extractor = extractor_cls()

            model_factory = _load_symbol(_require(stage_spec.model, "model"))
            model = model_factory(**_model_kwargs(cfg))
            trainer_cls = _load_symbol(_require(stage_spec.trainer, "trainer"))
            trainer = trainer_cls()
            result = trainer.fit(model, reader, extractor, logger=logger)
            trainer.save(result, output_dir / "model")
            # Upload the model artifact to MLflow under "model/" so downstream
            # evaluate/predict stages (and the lolday model-fetcher init container)
            # can fetch ``runs:/<run_id>/model``.
            logger.log_artifact(output_dir / "model", artifact_path="model")
            return

        if stage == "evaluate":
            source_model = Path(str(cfg.paths.source_model))
            train_spec = self._manifest.stages.get("train")
            trainer_symbol = stage_spec.trainer or (train_spec.trainer if train_spec else None)
            trainer = _load_symbol(_require(trainer_symbol, "trainer"))()
            model = trainer.load(source_model)
            test_csv = Path(str(cfg.data.test_csv))
            samples_root = Path(str(cfg.paths.samples_root))
            reader = reader_cls(csv=test_csv, samples_root=samples_root)
            extractor = extractor_cls()
            evaluator_cls = _load_symbol(_require(stage_spec.evaluator, "evaluator"))
            # Convention: for binary_classification, classes[0] is the positive class.
            evaluator = evaluator_cls(
                positive_class=self._manifest.output.classes[0],
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
            model = trainer.load(source_model)
            predict_csv = Path(str(cfg.data.predict_csv))
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
