"""Shared pytest fixtures for maldet tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """A disposable project-like directory for tests that write files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "conf").mkdir()
    return tmp_path
