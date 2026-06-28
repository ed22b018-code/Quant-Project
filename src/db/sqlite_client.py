"""SQLite connection helper — state persistence and run logs."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from types import TracebackType
from typing import Any


class SQLiteClient:
    """Thin context-manager wrapper around sqlite3.

    Usage:
        with SQLiteClient(db_path) as db:
            db.execute_schema(SQLITE_SCHEMA_SQL)
            db.execute("INSERT INTO state VALUES (?, ?, ?)", ("key", "val", ts))
    """

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> "SQLiteClient":
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")   # safe concurrent writes
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
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("SQLiteClient must be used inside a 'with' block")
        return self._conn

    def execute_schema(self, sql: str) -> None:
        """Execute a multi-statement schema SQL string (split on ';')."""
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                self.conn.execute(stmt)
        self.conn.commit()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def list_tables(self) -> set[str]:
        """Return the set of user-created table names."""
        rows = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        return {row[0] for row in rows}
