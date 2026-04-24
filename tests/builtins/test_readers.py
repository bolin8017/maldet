"""SampleCsvReader reads platform-provided sample CSVs."""

from __future__ import annotations

from pathlib import Path

import pytest

from maldet.builtins.readers import SampleCsvReader


def _write_samples(root: Path, shas: list[str]) -> None:
    for sha in shas:
        d = root / sha[:2]
        d.mkdir(parents=True, exist_ok=True)
        (d / sha).write_bytes(b"\x7fELF")


def test_train_csv_with_labels(tmp_path: Path) -> None:
    shas = ["a" * 64, "b" * 64]
    _write_samples(tmp_path / "samples", shas)
    csv = tmp_path / "train.csv"
    csv.write_text("file_name,label\n" + f"{shas[0]},Malware\n{shas[1]},Benign\n")
    reader = SampleCsvReader(csv=csv, samples_root=tmp_path / "samples")
    assert len(reader) == 2
    records = list(reader)
    assert records[0].sha256 == shas[0]
    assert records[0].label == "Malware"
    assert records[0].path == tmp_path / "samples" / shas[0][:2] / shas[0]


def test_predict_csv_without_label(tmp_path: Path) -> None:
    sha = "c" * 64
    _write_samples(tmp_path / "samples", [sha])
    csv = tmp_path / "predict.csv"
    csv.write_text("file_name\n" + sha + "\n")
    reader = SampleCsvReader(csv=csv, samples_root=tmp_path / "samples")
    records = list(reader)
    assert records[0].label is None


def test_missing_sample_file_raises_if_strict(tmp_path: Path) -> None:
    csv = tmp_path / "train.csv"
    csv.write_text("file_name,label\n" + "d" * 64 + ",Malware\n")
    reader = SampleCsvReader(csv=csv, samples_root=tmp_path / "samples", strict=True)
    with pytest.raises(FileNotFoundError):
        list(reader)


def test_missing_sample_skipped_by_default(tmp_path: Path) -> None:
    sha_good = "e" * 64
    _write_samples(tmp_path / "samples", [sha_good])
    csv = tmp_path / "train.csv"
    csv.write_text("file_name,label\n" + f"{sha_good},Malware\n" + "f" * 64 + ",Benign\n")
    reader = SampleCsvReader(csv=csv, samples_root=tmp_path / "samples", strict=False)
    records = list(reader)
    assert len(records) == 1
    assert records[0].sha256 == sha_good


def test_missing_file_name_column_raises(tmp_path: Path) -> None:
    csv = tmp_path / "train.csv"
    csv.write_text("sha,label\n" + "a" * 64 + ",Malware\n")
    reader = SampleCsvReader(csv=csv, samples_root=tmp_path)
    with pytest.raises(ValueError, match="file_name"):
        list(reader)
