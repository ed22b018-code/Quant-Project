"""Tests for DuckDB and SQLite schema creation."""
from __future__ import annotations

from pathlib import Path

from src.db.duckdb_client import DuckDBClient
from src.db.schema import DUCKDB_SCHEMA_SQL, SQLITE_SCHEMA_SQL
from src.db.sqlite_client import SQLiteClient

EXPECTED_DUCKDB_TABLES = {"ohlcv_daily", "ohlcv_minute", "ticks", "news", "option_chain"}
EXPECTED_SQLITE_TABLES = {"runs", "predictions", "trades", "signals", "positions", "state"}


def test_duckdb_creates_all_tables(tmp_duckdb_path: Path) -> None:
    with DuckDBClient(tmp_duckdb_path) as db:
        db.execute_schema(DUCKDB_SCHEMA_SQL)
        tables = db.list_tables()
    assert EXPECTED_DUCKDB_TABLES.issubset(tables)


def test_sqlite_creates_all_tables(tmp_sqlite_path: Path) -> None:
    with SQLiteClient(tmp_sqlite_path) as db:
        db.execute_schema(SQLITE_SCHEMA_SQL)
        tables = db.list_tables()
    assert EXPECTED_SQLITE_TABLES.issubset(tables)


def test_duckdb_schema_idempotent(tmp_duckdb_path: Path) -> None:
    """Running schema SQL twice must not raise (IF NOT EXISTS guards)."""
    with DuckDBClient(tmp_duckdb_path) as db:
        db.execute_schema(DUCKDB_SCHEMA_SQL)
        db.execute_schema(DUCKDB_SCHEMA_SQL)


def test_sqlite_schema_idempotent(tmp_sqlite_path: Path) -> None:
    with SQLiteClient(tmp_sqlite_path) as db:
        db.execute_schema(SQLITE_SCHEMA_SQL)
        db.execute_schema(SQLITE_SCHEMA_SQL)


def test_duckdb_in_memory_schema() -> None:
    """Schema works in an in-memory database (used by some tests)."""
    with DuckDBClient(":memory:") as db:
        db.execute_schema(DUCKDB_SCHEMA_SQL)
        assert EXPECTED_DUCKDB_TABLES.issubset(db.list_tables())


def test_duckdb_ohlcv_daily_insert_and_query(tmp_duckdb_path: Path) -> None:
    """Smoke-test that the ohlcv_daily schema accepts real data."""
    import datetime
    with DuckDBClient(tmp_duckdb_path) as db:
        db.execute_schema(DUCKDB_SCHEMA_SQL)
        db.execute(
            """
            INSERT INTO ohlcv_daily
              (symbol, date, open, high, low, close, volume, ingested_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("RELIANCE", datetime.date(2024, 1, 2), 2400.0, 2420.0, 2390.0, 2410.0,
             1_000_000, datetime.datetime.now(datetime.timezone.utc)),
        )
        df = db.query_df("SELECT * FROM ohlcv_daily")
    assert len(df) == 1
    assert df.iloc[0]["symbol"] == "RELIANCE"
