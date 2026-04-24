"""Built-in sample readers."""

from __future__ import annotations

import csv as _csv
from collections.abc import Iterator
from pathlib import Path

from maldet.types import Sample


class SampleCsvReader:
    """Reads a `sample_csv` contract CSV: columns ``file_name[,label]``.

    Resolves each sample path under ``samples_root/<sha[:2]>/<sha>``.

    When ``strict=False`` (default), missing sample files are skipped with no
    error (lolday frequently produces CSVs that reference samples not yet
    present; platform guarantees the SHA is valid, not that the byte stream
    is).
    """

    output_shape = None
    dtype = "bytes"

    def __init__(self, csv: Path, samples_root: Path, *, strict: bool = False) -> None:
        self._csv = csv
        self._root = samples_root
        self._strict = strict
        self._count: int | None = None

    def _resolve(self, sha: str) -> Path:
        return self._root / sha[:2] / sha

    def __iter__(self) -> Iterator[Sample]:
        count = 0
        with self._csv.open("r", encoding="utf-8", newline="") as f:
            reader = _csv.DictReader(f)
            if "file_name" not in (reader.fieldnames or []):
                raise ValueError(f"{self._csv}: CSV missing required 'file_name' column")
            for row in reader:
                sha = row["file_name"].strip()
                label = (row.get("label") or "").strip() or None
                path = self._resolve(sha)
                if not path.exists():
                    if self._strict:
                        raise FileNotFoundError(f"sample not found: {path}")
                    continue
                count += 1
                yield Sample(sha256=sha, path=path, label=label)
        self._count = count

    def __len__(self) -> int:
        if self._count is not None:
            return self._count
        with self._csv.open("r", encoding="utf-8", newline="") as f:
            self._count = sum(1 for _ in _csv.DictReader(f))
        return self._count
