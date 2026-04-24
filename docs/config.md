# Config (Hydra)

maldet uses [Hydra](https://hydra.cc) for configuration. Each stage reads a
single YAML file passed via `--config`.

## Required top-level keys

```yaml
defaults: [_self_]
stage: train          # train | evaluate | predict

paths:
  config_dir: .
  output_dir: ./output
  samples_root: ./samples
  source_model: ./output/model

data:
  train_csv: ./train.csv
  test_csv: ./test.csv
  predict_csv: ./predict.csv

model:
  _target_: mydet.models.make_rf
  n_estimators: 100
```

`model._target_` is resolved by `hydra.utils.instantiate`. Any extra keys
under `model:` become keyword arguments to the factory function.

## Path injection

`paths.output_dir`, `paths.samples_root`, and `paths.source_model` are
injected by the `StageRunner` before each stage. Override on the command
line:

```
maldet run train --config config.yaml \
  paths.output_dir=/mnt/runs/001 \
  paths.samples_root=/data/elf
```

## CLI overrides

Any config key can be overridden with dot-notation:

```
maldet run train --config config.yaml model.n_estimators=200
```

## Multirun sweeps

Hydra multirun is triggered with `--multirun` (or `-m`):

```
maldet run train --config config.yaml --multirun \
  model.n_estimators=50,100,200
```

Each combination runs in its own subdirectory under `output_dir`.
Combine with `hydra/launcher=joblib` for local parallelism.
