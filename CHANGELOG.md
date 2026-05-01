# Changelog

All notable changes to `maldet` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [1.2.0] — 2026-05-01

### Added

- `EventKind.PER_CLASS` (`"per_class"`) — new event kind. `BinaryClassification.evaluate` now emits one immediately after `confusion_matrix`, with payload `{per_class: dict[str, dict[str, float | int]]}`. Each entry maps a class name to `{precision, recall, f1, support}`. The lolday reconciler's `_project_summary_metrics` reads this to populate the `<PerClassMetrics>` card on evaluate-job detail pages (lolday Phase 13b §1.6a).

## [1.1.0] — 2026-04-27

### BREAKING

- `manifest.stages.{stage}` now requires both `config_class` (import path to a Pydantic `BaseModel` subclass) and `params_schema` (JSON Schema). Manifests built with maldet ≤ 1.0 will be rejected by `DetectorManifest.model_validate`. Detector authors must rebuild against this version.
- `BinaryClassification.evaluate` now emits the `confusion_matrix` event with labels in row-aligned order `[other_class, positive_class]` (matches sklearn `labels=[0, 1]`). The previous `[positive, other]` ordering was a latent bug — see commit `69841ec`. Downstream consumers reading `report.confusion_matrix["labels"]` may need to swap orientation.

### Added

- `maldet introspect-schema --config-class module.sub:Class [--out FILE]` — auto-derives JSON Schema from a stage's Pydantic config class via `cls.model_json_schema(mode="serialization")`. Used by `maldet build` to populate `manifest.stages.{stage}.params_schema` automatically.
- `EventKind.CONFUSION_MATRIX` (`"confusion_matrix"`) — new event kind. `BinaryClassification.evaluate` emits one per evaluation, with payload `{labels: list[str], matrix: list[list[int]]}`.
- `maldet check` strict lint: every `stage.config_class` must be a `pydantic.BaseModel` subclass with `model_config = ConfigDict(extra="forbid")`. Errors include the offending stage name.
- Scaffold templates (`rf`, `cnn`) emit `src/{name}/configs.py` with Pydantic `TrainConfig` / `EvaluateConfig` / `PredictConfig` (`extra="forbid"`), so a fresh scaffold passes `maldet check` end-to-end.

### Fixed

- `BinaryClassification.evaluate` confusion matrix labels-order alignment (see BREAKING above).
- `maldet introspect-schema` and `maldet check` produce friendly errors for import/attribute failures and unset `extra` (instead of bare tracebacks or misleading messages).

### Migration

For detector authors:
1. Define one Pydantic `BaseModel` per stage (`extra="forbid"`).
2. In `maldet.toml` each `[stages.{stage}]` block, set `config_class = "package.configs:MyConfig"` and `params_schema = {}` (placeholder; auto-populated by `maldet build`).
3. Update CI / build pipeline to call `maldet build` (which invokes `maldet introspect-schema` per stage) — if your build is the lolday backend pipeline, no detector-side change needed beyond the manifest + configs.
4. Run `maldet check` to verify.

## [1.0.8] — 2026-04-27

### Fixed

- `StageRunner` now threads `model_factory` (resolved from manifest `stages.train.model`) into `trainer.load(...)` for evaluate/predict when the trainer's signature accepts it. `LightningTrainer.load` requires the factory to rebuild the LightningModule around the saved state dict; `SklearnTrainer.load` doesn't take kwargs. Without this, CNN evaluate/predict failed with `ValueError: LightningTrainer.load requires model_factory to rebuild the module`.

## [1.0.7] — 2026-04-27

### Fixed

- `BinaryClassification.evaluate` and `BatchPredictor.predict` now skip samples whose extractor raises `ValueError` (with a `warning` event emitted), matching the behavior of `SklearnTrainer._materialize` / `LightningTrainer._materialize_tensor`. Same threshold (>50% skip → `RuntimeError`). Phase 11d E2E surfaced this — train ran clean (skip already in place) but evaluate/predict crashed on the first ELF sample missing `.text`.

## [1.0.6] — 2026-04-27

### Fixed

- `StageRunner` now uploads each stage's primary artifact to MLflow via `logger.log_artifact`: train uploads the model directory under `model/`, evaluate uploads `metrics.json`, predict uploads `predictions.csv`. Previously the runner wrote artifacts to the local output dir but never invoked the MLflow side of the `CompositeEventLogger` for them. Phase 11d E2E exposed this — evaluate/predict's model-fetcher init container failed with "Failed to download artifacts from path 'model'" because the trained model only ever existed on the (ephemeral) train pod's emptyDir.

## [1.0.5] — 2026-04-27

### Fixed

- `LightningTrainer.fit` now falls back to `tempfile.gettempdir()` (rather than cwd `.`) when `default_root_dir` is unset — Lightning writes its checkpoint subdir and internal log dir relative to that root, and cwd `/app` is mounted read-only under lolday's `readOnlyRootFilesystem` security context. Previously the trainer crashed at `save_checkpoint` with `OSError [Errno 30] Read-only file system: '/app/checkpoints'` after a successful training loop.

## [1.0.4] — 2026-04-27

### Changed

- `lightning` extra: cap `torch>=2.2,<2.7` (previously `torch>=2.2`). PyPI default wheels for torch 2.7+ are built against CUDA 12.8 and refuse to `_cuda_init` under NVIDIA driver line 560.35.03 (which tops out at CUDA 12.6). torch 2.6 wheels target CUDA 12.4, which is forward-compatible on the 12.6 driver. Loosen the upper bound once deployment drivers are upgraded past 565.

## [1.0.3] — 2026-04-27

### Changed

- `StageRunner` now loads the model class via `_load_symbol(stage_spec.model)` from the manifest, consistent with how `reader`, `extractor`, and `trainer` are loaded. `cfg.model` is treated as plain kwargs for the factory; Hydra meta-fields (`_target_`, `_partial_`, `_args_`, `_recursive_`, `_convert_`) are silently stripped if present, so legacy YAML still works. Removes the `hydra.utils.instantiate` dependency from the train branch.

### Why

- The previous `hydra_instantiate(cfg.model)` path required users to pass `_target_` in YAML, which conflicts with platforms that forbid Hydra meta-overrides on security grounds (e.g. lolday's params guard rejects `_target_` to prevent arbitrary remote code execution). Phase 11d E2E surfaced this — train jobs failed with `ConfigAttributeError: Missing key model` because the platform's rendered YAML had no `_target_` and the user couldn't supply one. After this change the manifest is the single source of truth for what factory is used; users only configure kwargs.

## [1.0.2] — 2026-04-27

### Fixed
- CI/publish workflows: unpin `astral-sh/setup-uv@v3` from `0.4.*` so the newer lockfile format (editable dynamic packages omit the `version` field) parses correctly. v1.0.1 tagged but its publish run failed at `uv sync` for this reason; v1.0.2 supersedes it. No code changes since 1.0.1.

## [1.0.1] — 2026-04-27

### Fixed
- `SklearnTrainer._materialize` and `LightningTrainer._materialize_tensor` now skip samples whose feature extractor raises `ValueError`, emitting a `warning` event when an `EventLogger` is supplied. They abort with `RuntimeError` only when the skip rate exceeds 50% or zero usable samples remain. Previously the first bad sample killed the whole train run — discovered during Phase 11d E2E where ELF samples lacking a `.text` section crashed `Text256Extractor.extract`.
- `templates/{rf,cnn}/Dockerfile.j2` now `COPY README.md` so scaffolded detectors that declare `readme = "README.md"` in `pyproject.toml` no longer hit hatchling "Readme file does not exist" at wheel build.

## [1.0.0] — 2026-04-24

### Added
- Core framework: `Sample`, `TrainResult`, `MetricReport` dataclasses; runtime-checkable Protocols for the six layers + `EventLogger`.
- Event stream: `CompositeEventLogger`, `JsonlEventLogger`, `StdoutEventLogger`, `MlflowEventLogger` (optional dep).
- Manifest: `DetectorManifest` Pydantic model + discovery via `MALDET_MANIFEST` env, `./maldet.toml`, `/app/maldet.toml` fallbacks.
- Built-ins: `SampleCsvReader`, `BatchPredictor`, `BinaryClassification`.
- Trainers: `SklearnTrainer` (joblib save/load), `LightningTrainer` (DDP-ready via `MALDET_GPU_COUNT` / `MALDET_DISTRIBUTED_STRATEGY` env; `ModelCheckpoint` + `EarlyStopping` callbacks; DataParallel retired in favor of DDP).
- `StageRunner` orchestrating manifest + Hydra YAML → component instantiation → stage execution; writes `manifest.json` alongside `model/`, `events.jsonl`, `metrics.json`, `predictions.csv`.
- CLI: `maldet run train|evaluate|predict`, `maldet describe` (json/toml), `maldet check`, `maldet scaffold --template rf|cnn`.
- mkdocs-material documentation site deployed to https://bolin8017.github.io/maldet/.
- GitHub Actions CI (ruff + mypy + pytest with 80% coverage gate) and `v*`-triggered publish workflow (uv publish + `UV_PUBLISH_TOKEN` secret).
- 57 unit tests + 2 integration tests (sklearn + Lightning CPU end-to-end).
