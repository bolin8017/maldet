"""Scaffold → install → train → evaluate → predict end-to-end (sklearn path)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from maldet.cli import app

runner = CliRunner()
pytestmark = pytest.mark.integration


def test_e2e_sklearn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    det_dir = tmp_path / "mydet"
    r = runner.invoke(
        app, ["scaffold", "--template", "rf", "--name", "mydet", "--out", str(det_dir)]
    )
    assert r.exit_code == 0

    # Patch features.py with a trivial extractor to avoid real ELF requirement.
    (det_dir / "src" / "mydet" / "features.py").write_text(
        "import numpy as np\n"
        "from maldet.types import Sample\n"
        "class Text256Extractor:\n"
        "    output_shape = (4,)\n"
        "    dtype = 'float32'\n"
        "    def extract(self, sample):\n"
        "        return np.ones(4, dtype=np.float32) if sample.label == 'Malware' else np.zeros(4, dtype=np.float32)\n"
    )

    # Install the scaffolded detector into this venv.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", str(det_dir)])

    # Layout fake samples + CSVs
    samples = tmp_path / "samples"
    samples.mkdir()
    shas = [f"{i:064x}" for i in range(20)]
    for sha in shas:
        (samples / sha[:2]).mkdir(parents=True, exist_ok=True)
        (samples / sha[:2] / sha).write_bytes(b"x")
    train = tmp_path / "train.csv"
    train.write_text(
        "file_name,label\n"
        + "\n".join(f"{s},{'Malware' if i % 2 else 'Benign'}" for i, s in enumerate(shas))
        + "\n"
    )
    test = tmp_path / "test.csv"
    test.write_text(train.read_text())
    predict = tmp_path / "predict.csv"
    predict.write_text("file_name\n" + "\n".join(shas) + "\n")

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        f"""
defaults:
  - _self_
stage: train
paths:
  config_dir: {tmp_path}
  output_dir: {tmp_path}/output
  samples_root: {samples}
  source_model: {tmp_path}/output/model
data:
  train_csv: {train}
  test_csv: {test}
  predict_csv: {predict}
model:
  n_estimators: 5
  random_state: 0
"""
    )

    monkeypatch.setenv("MALDET_MANIFEST", str(det_dir / "maldet.toml"))

    # Train
    r = runner.invoke(app, ["run", "train", "--config", str(cfg)])
    assert r.exit_code == 0, r.stdout
    assert (tmp_path / "output" / "model" / "model.joblib").exists()
    events_raw = (tmp_path / "output" / "events.jsonl").read_text()
    assert '"stage_end"' in events_raw

    # Evaluate
    cfg.write_text(cfg.read_text().replace("stage: train", "stage: evaluate"))
    r = runner.invoke(app, ["run", "evaluate", "--config", str(cfg)])
    assert r.exit_code == 0, r.stdout
    metrics_text = (tmp_path / "output" / "metrics.json").read_text()
    assert "binary_classification" in metrics_text

    # Predict
    cfg.write_text(cfg.read_text().replace("stage: evaluate", "stage: predict"))
    r = runner.invoke(app, ["run", "predict", "--config", str(cfg)])
    assert r.exit_code == 0, r.stdout
    assert (tmp_path / "output" / "predictions.csv").exists()


# Cleanup: uninstall the scaffolded package after the test (leave-no-trace).
def teardown_module() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "mydet"],
        check=False,
        capture_output=True,
    )
