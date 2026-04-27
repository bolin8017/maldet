# Changelog

All notable changes to `maldet` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
