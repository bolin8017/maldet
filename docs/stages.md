# Stages & Manifest

## Stages

maldet supports three stages. Each stage is driven by `maldet run <stage>
--config <path>`.

### train

Loads `data.train_csv`, materializes features, calls `Trainer.fit()`, and
writes artifacts to `paths.output_dir`:

- `model/` — serialized estimator or Lightning checkpoint directory
- `events.jsonl` — full event stream
- `manifest.json` — snapshot of `maldet.toml` for provenance

### evaluate

Loads `paths.source_model`, runs the model over `data.test_csv`, calls
`Evaluator.evaluate()`, and writes `metrics.json`.

### predict

Loads `paths.source_model`, runs the model over `data.predict_csv`, calls
`Predictor.predict()`, and writes `predictions.csv`.

## maldet.toml fields

`maldet.toml` is the detector manifest. It must be present in the working
directory or at `$MALDET_MANIFEST`.

### `[detector]`

| Field | Type | Required |
|-------|------|----------|
| `name` | string | yes |
| `version` | string | yes |
| `framework` | `sklearn` \| `lightning` \| `sklearn+lightning` | yes |
| `description` | string | no |

### `[input]`

| Field | Type | Default |
|-------|------|---------|
| `binary_format` | `elf` \| `pe` \| `apk` \| `raw_bytes` | required |
| `required_sections` | list[str] | `[]` |
| `dataset_contract` | string | `"sample_csv"` |

### `[output]`

| Field | Type | Default |
|-------|------|---------|
| `task` | `binary_classification` \| `multiclass_classification` \| ... | required |
| `classes` | list[str] | `[]` |
| `score_range` | [float, float] | `[0.0, 1.0]` |

### `[resources]`

| Field | Type | Default |
|-------|------|---------|
| `supports` | list of `cpu`/`gpu1`/`gpu2`/`gpu4`/`gpu8` | `["cpu"]` |
| `recommended` | string | `"cpu"` |
| `min_memory_gib` | int | `1` |
| `gpu_required` | bool | `false` |

### `[lifecycle]`

| Field | Type | Default |
|-------|------|---------|
| `stages` | list | `["train", "evaluate", "predict"]` |
| `supports_serving` | bool | `false` |
| `supports_hpsweep` | bool | `true` |
| `supports_distributed` | bool \| `"ddp"` \| `"fsdp"` | `false` |
| `supports_multinode` | bool | `false` |

### `[artifacts]`

Defines the expected output paths for `model`, `metrics`, and `predictions`.

### `[compat]`

| Field | Default |
|-------|---------|
| `min_python` | `"3.12"` |
| `min_maldet` | `"1.0"` |
| `schema_version` | `1` |

### `[stages.<name>]`

Each stage block declares dotted-import paths (`module:Class`) for the
layer symbols to instantiate:

```toml
[stages.train]
reader    = "mydet.readers:MyReader"
extractor = "mydet.features:MyFeatures"
model     = "mydet.models:make_rf"
trainer   = "maldet.trainers.sklearn_trainer:SklearnTrainer"

[stages.evaluate]
evaluator = "maldet.evaluators.binary:BinaryClassificationEvaluator"

[stages.predict]
predictor = "maldet.builtins.predictors:BatchPredictor"
```
