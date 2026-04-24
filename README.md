# maldet

Plug-and-play framework for building and shipping malware detectors. Six composable layers (reader / feature / model / trainer / evaluator / predictor) connected by runtime-checkable Protocols, a framework-owned CLI (`maldet run|describe|check|scaffold`), Hydra-based configuration with CLI overrides and multirun sweeps, and unified support for scikit-learn and PyTorch Lightning. Built to run on the lolday malware-detection platform and standalone.

## Install

```
pip install maldet             # core
pip install maldet[lightning]  # add PyTorch Lightning engine
pip install maldet[mlflow]     # add MLflow experiment tracking
pip install maldet[all]        # everything
```

## Quickstart

See [docs](https://bolin8017.github.io/maldet/quickstart/).

## License

MIT.
