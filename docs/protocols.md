# Protocols

All six layers plus the event logger are defined as `@runtime_checkable`
`typing.Protocol` classes in `maldet.protocols`. Implementations do not
need to inherit — structural typing is used throughout.

`@runtime_checkable` enables `isinstance(obj, Trainer)` checks during
pipeline assembly, so the CLI and `StageRunner` can validate components
at startup rather than at first use.

## SampleReader

```python
class SampleReader(Protocol):
    def __iter__(self) -> Iterator[Sample]: ...
    def __len__(self) -> int: ...
```

Produces `Sample` objects. `__len__` must return a fast estimate (for
progress reporting); it does not have to force full iteration.

## FeatureExtractor

```python
class FeatureExtractor(Protocol):
    output_shape: tuple[int, ...] | None
    dtype: str
    def extract(self, sample: Sample) -> np.ndarray: ...
```

Stateless and pure. `output_shape` is `None` when the shape is dynamic
(variable-length sequences). `dtype` is a numpy dtype string, e.g. `"float32"`.

## EventLogger

```python
class EventLogger(Protocol):
    def log_metric(self, name: str, value: float, step: int | None = None) -> None: ...
    def log_params(self, params: dict[str, Any]) -> None: ...
    def log_artifact(self, path: Path, artifact_path: str | None = None) -> None: ...
    def log_event(self, kind: str, **payload: Any) -> None: ...
    def set_tags(self, tags: dict[str, str]) -> None: ...
```

Passed into every layer. The `CompositeEventLogger` fans out to JSONL,
stdout, and MLflow simultaneously.

## Trainer

```python
class Trainer(Protocol):
    def fit(self, model, train, extractor, *, val=None, logger) -> TrainResult: ...
    def save(self, result: TrainResult, out_dir: Path) -> None: ...
    def load(self, model_dir: Path) -> Any: ...
```

Owns the full training loop plus serialization and deserialization.

## Evaluator

```python
class Evaluator(Protocol):
    def evaluate(self, model, reader, extractor, *, logger) -> MetricReport: ...
```

Returns a `MetricReport`; does not write files (that is the runner's job).

## Predictor

```python
class Predictor(Protocol):
    def predict(self, model, reader, extractor, *, out_path: Path, logger) -> Path: ...
```

Writes `predictions.csv` to `out_path` and returns the path.

See [API Reference](reference.md) for full signatures.
