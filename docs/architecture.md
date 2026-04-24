# Architecture

maldet is a composition framework. Each detector is a pipeline assembled from
six layers; each layer has a single responsibility and a well-defined interface
(a `typing.Protocol`).

## The six layers

1. **SampleReader** — produces an iterable of `Sample(sha256, path, label, metadata)`. The builtin `SampleCsvReader` reads the `sample_csv` contract: `file_name[,label]` columns, sample at `samples_root/<sha[:2]>/<sha>`.
2. **FeatureExtractor** — transforms one `Sample` into an ndarray or tensor. Stateless, pure, cacheable.
3. **Model** — the estimator or `nn.Module`. Contains no I/O, no training logic.
4. **Trainer** — runs `fit()` and owns serialization. Two engines: `SklearnTrainer` (joblib) and `LightningTrainer` (ckpt + DDP-ready).
5. **Evaluator** — computes metrics into a `MetricReport`. Builtin `BinaryClassification` uses `sklearn.metrics`.
6. **Predictor** — batch predictions into the standardized `predictions.csv` shape. Online mode is deferred.

## Control plane

The `maldet` CLI is the only entry point. It reads `maldet.toml` (manifest) and a Hydra YAML config, instantiates the declared layers, and drives a single stage.

See [Protocols](protocols.md) and [Stages & Manifest](stages.md) for the interface details.
