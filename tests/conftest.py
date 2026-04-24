"""Shared pytest fixtures for maldet tests."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Iterator[Path]:
    """A disposable project-like directory for tests that write files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "conf").mkdir()
    return tmp_path
