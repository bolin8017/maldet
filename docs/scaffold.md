# Scaffolding

`maldet scaffold` generates a complete detector repository from a Jinja2
template. The output is a pip-installable Python package that satisfies the
maldet protocol contracts out of the box.

## Usage

```
maldet scaffold --template <rf|cnn> --name <detector_name> --out <directory>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--template` | `rf` | Template to use: `rf` or `cnn` |
| `--name` | _(required)_ | Package and detector name |
| `--out` | `.` | Parent directory for the generated tree |

## Templates

**`rf`** — Random Forest baseline using `SklearnTrainer`. Generates:
a `pyproject.toml`, `maldet.toml`, `src/<name>/features.py`,
`src/<name>/models.py`, `config.yaml`, and `Dockerfile`.

**`cnn`** — PyTorch Lightning CNN using `LightningTrainer`. Generates the
same structure with a `LightningModule` stub and Lightning-specific config.

## After scaffolding

```
cd <out>
pip install -e .
maldet check      # validates maldet.toml
maldet describe   # prints manifest summary
```

Edit `src/<name>/features.py` to implement your feature extraction logic.
