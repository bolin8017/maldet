# Quickstart

## Install

```
pip install maldet[lightning,mlflow]
```

## Scaffold a detector

```
maldet scaffold --template rf --name mydet --out ./mydet
cd ./mydet
pip install -e .
maldet check
maldet describe
```

## Write your features

Open `src/mydet/features.py`. Replace the ELF parse with your feature logic.
The class must satisfy the `FeatureExtractor` protocol:

```python
class MyFeatures:
    output_shape = (128,)
    dtype = "float32"
    def extract(self, sample): ...
```

## Train locally

Write a `config.yaml`:

```yaml
defaults: [_self_]
stage: train
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

Run:

```
maldet run train --config config.yaml
```

Artifacts land under `./output/` (`model/`, `events.jsonl`, `metrics.json`, `manifest.json`).
