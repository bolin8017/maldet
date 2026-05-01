"""StageRunner end-to-end orchestration (sklearn path)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from maldet.evaluators.binary import BinaryClassification
from maldet.runner import StageRunner, _model_kwargs
from maldet.trainers.sklearn_trainer import SklearnTrainer

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


def test_model_kwargs_drops_legacy_hydra_meta_fields() -> None:
    """Legacy YAML with `_target_` (or other Hydra meta-fields) still works —
    runner now uses the manifest's stages.train.model symbol as the source of
    truth, and silently drops Hydra meta-fields from cfg.model so they don't
    get passed to the factory as stray kwargs."""
    cfg = OmegaConf.create(
        {
            "model": {
                "_target_": "some.legacy.path",
                "_convert_": "partial",
                "n_estimators": 5,
                "random_state": 0,
            }
        }
    )
    assert _model_kwargs(cfg) == {"n_estimators": 5, "random_state": 0}


def test_model_kwargs_handles_missing_section() -> None:
    """When cfg has no model section, factory gets called with no kwargs."""
    cfg = OmegaConf.create({"stage": "train"})
    assert _model_kwargs(cfg) == {}


def test_train_passes_classes_from_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner must pass manifest.output.classes as classes= kwarg to trainer.fit."""
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

    captured: dict = {}
    real_fit = SklearnTrainer.fit

    def spy_fit(self, model, train, extractor, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return real_fit(self, model, train, extractor, **kwargs)

    monkeypatch.setattr(SklearnTrainer, "fit", spy_fit)

    StageRunner().run(stage="train", config_path=cfg_path)

    # Fixture's manifest declares classes = ["Malware", "Benign"] — runner must
    # forward it verbatim (NOT silently reorder, NOT derive from positive_class).
    assert captured["classes"] == ["Malware", "Benign"]


def test_evaluate_passes_explicit_positive_class_from_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Runner must pass manifest.output.positive_class to evaluator constructor,
    not derive from classes[0]. The fixture sets classes=["Malware", "Benign"]
    with positive_class="Malware" — we deliberately override to positive_class=
    "Benign" so the runner code paths that incorrectly use classes[0] would
    leak the wrong value through to the evaluator."""
    monkeypatch.syspath_prepend(str(tmp_path))
    _write_fake_detector(tmp_path)
    paths = _write_csvs_and_samples(tmp_path)

    # First train (positive_class doesn't matter for train).
    cfg_path = _write_config(tmp_path, "train", paths)
    manifest = tmp_path / "maldet.toml"
    # The evaluate stage in the fixture lacks an extractor — runner requires one,
    # so inject ``extractor = "fakedet:Extr"`` after the [stages.evaluate] header.
    # Also flip positive_class to "Benign" (≠ classes[0]) so the spied evaluator
    # call asserts the runner passed the explicit field, not classes[0].
    manifest_text = (
        (FIX / "valid_manifest.toml")
        .read_text()
        .replace(
            'extractor = "maldet.builtins.readers:SampleCsvReader"',
            'extractor = "fakedet:Extr"',
        )
        .replace(
            'positive_class = "Malware"',
            'positive_class = "Benign"',
        )
        .replace(
            "[stages.evaluate]\n" 'reader = "maldet.builtins.readers:SampleCsvReader"\n',
            "[stages.evaluate]\n"
            'reader = "maldet.builtins.readers:SampleCsvReader"\n'
            'extractor = "fakedet:Extr"\n',
        )
    )
    manifest.write_text(manifest_text)
    monkeypatch.setenv("MALDET_MANIFEST", str(manifest))

    StageRunner().run(stage="train", config_path=cfg_path)

    # Now spy on BinaryClassification.__init__ and run evaluate.
    captured: dict = {}
    real_init = BinaryClassification.__init__

    def spy_init(self, **kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return real_init(self, **kwargs)

    monkeypatch.setattr(BinaryClassification, "__init__", spy_init)

    eval_cfg_path = _write_config(tmp_path, "evaluate", paths)
    StageRunner().run(stage="evaluate", config_path=eval_cfg_path)

    # The runner must have forwarded the explicit positive_class field, NOT classes[0].
    assert captured["positive_class"] == "Benign"
    assert captured["class_names"] == ["Malware", "Benign"]
