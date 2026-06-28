"""Shared pytest fixtures for Phase 0 tests."""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def tmp_duckdb_path(tmp_path: Path) -> Path:
    """A temporary DuckDB file path that is cleaned up after the test."""
    return tmp_path / "test.duckdb"


@pytest.fixture()
def tmp_sqlite_path(tmp_path: Path) -> Path:
    """A temporary SQLite file path that is cleaned up after the test."""
    return tmp_path / "test.db"
