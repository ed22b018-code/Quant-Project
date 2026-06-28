"""Tests for the live data source factory.

Verifies that get_live_source() returns the correct adapter based on the
config.data.live_data_source key, and that swapping the key from 'dhan' to
'upstox' is the only change needed to switch live sources.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.data.base import DataSource
from src.data.dhan_adapter import DhanDataSource
from src.data.live_source_factory import get_live_source
from src.data.live_upstox import UpstoxLiveSource
from src.db.schema import DUCKDB_SCHEMA_SQL
from src.db.duckdb_client import DuckDBClient


# ── Helpers ────────────────────────────────────────────────────────────────────

def _config(source: str, db_path: str = ":memory:") -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(
        live_data_source=source,
        duckdb_path=db_path,
    ))


@pytest.fixture()
def db(tmp_duckdb_path: Path) -> Path:
    with DuckDBClient(tmp_duckdb_path) as conn:
        conn.execute_schema(DUCKDB_SCHEMA_SQL)
    return tmp_duckdb_path


# ── Core routing tests ─────────────────────────────────────────────────────────

def test_factory_returns_dhan_for_dhan_config(db):
    cfg = _config("dhan", str(db))
    src = get_live_source(cfg)
    assert isinstance(src, DhanDataSource)


def test_factory_returns_upstox_for_upstox_config(db):
    cfg = _config("upstox", str(db))
    src = get_live_source(cfg)
    assert isinstance(src, UpstoxLiveSource)


def test_dhan_source_is_datasource_subclass(db):
    cfg = _config("dhan", str(db))
    src = get_live_source(cfg)
    assert isinstance(src, DataSource)


def test_upstox_source_is_datasource_subclass(db):
    cfg = _config("upstox", str(db))
    src = get_live_source(cfg)
    assert isinstance(src, DataSource)


def test_unknown_source_name_raises_value_error(db):
    cfg = _config("zerodha", str(db))
    with pytest.raises(ValueError, match="zerodha"):
        get_live_source(cfg)


def test_factory_importable_without_either_sdk():
    """Importing live_source_factory must not trigger SDK imports."""
    import importlib
    import sys
    # Remove cached module to force re-import
    mods_to_remove = [k for k in sys.modules if "live_source_factory" in k]
    for m in mods_to_remove:
        del sys.modules[m]
    # Should import without dhanhq or upstox_client installed
    mod = importlib.import_module("src.data.live_source_factory")
    assert hasattr(mod, "get_live_source")


def test_db_path_override_used(tmp_duckdb_path: Path):
    """db_path kwarg overrides config.data.duckdb_path."""
    with DuckDBClient(tmp_duckdb_path) as conn:
        conn.execute_schema(DUCKDB_SCHEMA_SQL)
    cfg = _config("dhan", db_path=":memory:")
    src = get_live_source(cfg, db_path=str(tmp_duckdb_path))
    assert isinstance(src, DhanDataSource)
    # Verify it's using our db_path (DhanDataSource stores _db_path)
    assert str(tmp_duckdb_path) in src._db_path


def test_config_live_data_source_default_dhan(db):
    """If live_data_source attr is missing, factory defaults to 'dhan'."""
    cfg = SimpleNamespace(data=SimpleNamespace(duckdb_path=str(db)))
    src = get_live_source(cfg)
    assert isinstance(src, DhanDataSource)
