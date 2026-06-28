"""DuckDB connection manager and query helpers."""
from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any

import duckdb
import pandas as pd


class DuckDBClient:
    """Thin context-manager wrapper around a DuckDB connection.

    Usage:
        with DuckDBClient(db_path) as db:
            db.execute_schema(DUCKDB_SCHEMA_SQL)
            df = db.query_df("SELECT * FROM ohlcv_daily WHERE symbol = ?", ("RELIANCE",))
    """

    def __init__(self, db_path: Path | str = ":memory:") -> None:
        self.db_path = str(db_path)
        self._conn: duckdb.DuckDBPyConnection | None = None

    def __enter__(self) -> "DuckDBClient":
        self._conn = duckdb.connect(self.db_path)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            raise RuntimeError("DuckDBClient must be used inside a 'with' block")
        return self._conn

    def execute_schema(self, sql: str) -> None:
        """Execute a multi-statement DDL string, splitting on ';'.

        Comment lines (-- ...) are stripped before splitting so that a semicolon
        inside a comment does not produce a spurious empty statement.
        """
        stripped = "\n".join(
            line for line in sql.splitlines() if not line.strip().startswith("--")
        )
        for stmt in stripped.split(";"):
            stmt = stmt.strip()
            if stmt:
                self.conn.execute(stmt)

    def execute(
        self, sql: str, params: tuple[Any, ...] | None = None
    ) -> duckdb.DuckDBPyRelation:
        if params:
            return self.conn.execute(sql, list(params))
        return self.conn.execute(sql)

    def query_df(self, sql: str, params: tuple[Any, ...] | None = None) -> pd.DataFrame:
        """Execute a SELECT and return a pandas DataFrame."""
        return self.execute(sql, params).df()

    def list_tables(self) -> set[str]:
        """Return the set of table names in the connected database."""
        rows = self.conn.execute("SHOW TABLES").fetchall()
        return {row[0] for row in rows}
