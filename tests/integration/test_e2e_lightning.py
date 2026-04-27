"""End-to-end Lightning training (CPU, 1 epoch, trivial MLP)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from maldet.cli import app

runner = CliRunner()
pytestmark = pytest.mark.integration


def test_e2e_lightning_cpu(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MALDET_GPU_COUNT", "0")

    det_dir = tmp_path / "mynn"
    r = runner.invoke(
        app, ["scaffold", "--template", "cnn", "--name", "mynn", "--out", str(det_dir)]
    )
    assert r.exit_code == 0

    # Trivial features extractor (avoid real ELF)
    (det_dir / "src" / "mynn" / "features.py").write_text(
        "import numpy as np\n"
        "from maldet.types import Sample\n"
        "class Text256Extractor:\n"
        "    output_shape = (4,)\n"
        "    dtype = 'float32'\n"
        "    def extract(self, sample):\n"
        "        return np.ones(4, dtype=np.float32) if sample.label == 'Malware' else np.zeros(4, dtype=np.float32)\n"
    )

    # Replace CNN with trivial MLP
    (det_dir / "src" / "mynn" / "models.py").write_text(
        "import lightning.pytorch as pl\n"
        "import torch\n"
        "from torch import nn\n\n"
        "class TinyMLP(pl.LightningModule):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.net = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))\n"
        "        self.loss = nn.CrossEntropyLoss()\n"
        "    def forward(self, x):\n"
        "        return self.net(x.float())\n"
        "    def training_step(self, batch, batch_idx):\n"
        "        x, y = batch\n"
        "        logits = self(x)\n"
        "        loss = self.loss(logits, y)\n"
        "        self.log('train_loss', loss)\n"
        "        return loss\n"
        "    def configure_optimizers(self):\n"
        "        return torch.optim.Adam(self.parameters(), lr=1e-2)\n\n"
        "def make_cnn(**kwargs):\n"
        "    return TinyMLP()\n"
    )

    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", str(det_dir)])

    samples = tmp_path / "samples"
    samples.mkdir()
    shas = [f"{i:064x}" for i in range(16)]
    for sha in shas:
        (samples / sha[:2]).mkdir(parents=True, exist_ok=True)
        (samples / sha[:2] / sha).write_bytes(b"x")
    train = tmp_path / "train.csv"
    train.write_text(
        "file_name,label\n"
        + "\n".join(f"{s},{'Malware' if i % 2 else 'Benign'}" for i, s in enumerate(shas))
        + "\n"
    )

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
  test_csv: {train}
  predict_csv: {train}
"""
    )
    monkeypatch.setenv("MALDET_MANIFEST", str(det_dir / "maldet.toml"))
    r = runner.invoke(app, ["run", "train", "--config", str(cfg)])
    assert r.exit_code == 0, r.stdout
    assert (tmp_path / "output" / "model" / "model.ckpt").exists()


def teardown_module() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "mynn"],
        check=False,
        capture_output=True,
    )
