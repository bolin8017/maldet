"""StageRunner end-to-end orchestration (sklearn path)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from maldet.runner import StageRunner

FIX = Path(__file__).parent / "fixtures"


def _write_config(tmp_path: Path, stage: str, paths: dict[str, str]) -> Path:
    cfg = {
        "defaults": ["_self_"],
        "stage": stage,
        "paths": paths,
        "data": {
            "train_csv": str(tmp_path / "train.csv"),
            "test_csv": str(tmp_path / "test.csv"),
            "predict_csv": str(tmp_path / "predict.csv"),
        },
        "model": {
            "_target_": "sklearn.ensemble.RandomForestClassifier",
            "n_estimators": 5,
            "random_state": 0,
        },
    }
    p = tmp_path / "config.yaml"
    p.write_text(OmegaConf.to_yaml(OmegaConf.create(cfg)))
    return p


def _write_fake_detector(tmp_path: Path) -> None:
    (tmp_path / "fakedet.py").write_text(
        "import numpy as np\n"
        "class Extr:\n"
        "    output_shape = (2,)\n"
        "    dtype = 'float32'\n"
        "    def extract(self, sample):\n"
        "        return np.array([1.0, 0.0] if sample.label == 'Benign' else [0.0, 1.0], dtype=np.float32)\n"
    )


def _write_csvs_and_samples(tmp_path: Path) -> dict[str, str]:
    samples = tmp_path / "samples"
    samples.mkdir()
    shas = [f"{i:064x}" for i in range(10)]
    for sha in shas:
        (samples / sha[:2]).mkdir(parents=True, exist_ok=True)
        (samples / sha[:2] / sha).write_bytes(b"x")
    train = tmp_path / "train.csv"
    train.write_text(
        "file_name,label\n"
        + "\n".join(f"{sha},{'Malware' if i % 2 else 'Benign'}" for i, sha in enumerate(shas))
        + "\n"
    )
    test = tmp_path / "test.csv"
    test.write_text(train.read_text())
    predict = tmp_path / "predict.csv"
    predict.write_text("file_name\n" + "\n".join(shas) + "\n")
    return {
        "config_dir": str(tmp_path),
        "output_dir": str(tmp_path / "output"),
        "samples_root": str(samples),
        "source_model": str(tmp_path / "output" / "model"),
    }


def test_runner_train_sklearn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(tmp_path))
    _write_fake_detector(tmp_path)
    paths = _write_csvs_and_samples(tmp_path)
    cfg_path = _write_config(tmp_path, "train", paths)

    manifest = tmp_path / "maldet.toml"
    manifest.write_text(
        (FIX / "valid_manifest.toml")
        .read_text()
        .replace(
            'extractor = "maldet.builtins.readers:SampleCsvReader"',
            'extractor = "fakedet:Extr"',
        )
    )
    monkeypatch.setenv("MALDET_MANIFEST", str(manifest))

    runner = StageRunner()
    runner.run(stage="train", config_path=cfg_path)

    out_model = Path(paths["output_dir"]) / "model" / "model.joblib"
    assert out_model.exists()

    events_file = Path(paths["output_dir"]) / "events.jsonl"
    assert events_file.exists()
    events = [json.loads(line) for line in events_file.read_text().splitlines()]
    kinds = [e["kind"] for e in events]
    assert "stage_begin" in kinds
    assert "stage_end" in kinds

    # manifest.json should also be present for provenance
    assert (Path(paths["output_dir"]) / "manifest.json").exists()
