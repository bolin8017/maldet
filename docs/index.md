# maldet

Plug-and-play framework for building malware detectors in Python.

Six composable layers — **Reader**, **Extractor**, **Model**, **Trainer**,
**Evaluator**, **Predictor** — connected by runtime-checkable Protocols.
Classical ML (scikit-learn) and deep learning (PyTorch Lightning) are
first-class, side-by-side. A single framework-owned CLI — `maldet run`,
`describe`, `check`, `scaffold` — is the only thing platforms need to invoke.

## Install

```
pip install maldet[lightning,mlflow]
```

## Get started

See [Quickstart](quickstart.md).
