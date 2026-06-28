"""Experiment run tracker — SQLite-backed, captures everything needed to reproduce a run.

Each run record stores:
  config_hash          SHA-256 of the full config JSON — detects silent config drift
  config_json          Full config snapshot — independent of config.toml on disk
  git_commit           Commit hash at run time — links result to exact code version
  data_source          Name of the DataSource adapter used
  data_source_limitations  Human-readable caveat string from the adapter
  data_granularity     'daily' | 'minute' | 'tick' | 'none'
                       REQUIRED: every report must disclose whether backtest used
                       daily or intraday data (DESIGN.md rule — no silent daily→intraday conflation)
  random_seed          Fixed seed for full reproducibility
  date_range_start/end Training / simulation date range
  universe_size/hash   How many symbols and which set
  metrics_json         Arbitrarily structured result metrics
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class RunStatus:
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"


class ExperimentTracker:
    """Write and query the 'runs' SQLite table."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id                   TEXT     PRIMARY KEY,
                    created_at               TEXT     NOT NULL,
                    completed_at             TEXT,
                    status                   TEXT     NOT NULL DEFAULT 'running',
                    config_hash              TEXT     NOT NULL,
                    config_json              TEXT     NOT NULL,
                    git_commit               TEXT,
                    data_source              TEXT     NOT NULL,
                    data_source_limitations  TEXT     NOT NULL,
                    data_granularity         TEXT     NOT NULL,
                    random_seed              INTEGER  NOT NULL,
                    date_range_start         TEXT,
                    date_range_end           TEXT,
                    universe_size            INTEGER,
                    universe_hash            TEXT,
                    metrics_json             TEXT,
                    error                    TEXT,
                    notes                    TEXT
                )
            """)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Public API ────────────────────────────────────────────────────────────

    def start_run(
        self,
        *,
        config_hash: str,
        config_json: dict[str, Any],
        git_commit: str | None,
        data_source: str,
        data_source_limitations: str,
        data_granularity: str,
        random_seed: int,
        date_range_start: str | None,
        date_range_end: str | None,
        universe_size: int,
        universe_hash: str,
        notes: str | None = None,
    ) -> str:
        """Insert a new run record with status=running. Returns the run_id."""
        run_id = str(uuid.uuid4())
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO runs (
                    run_id, created_at, status,
                    config_hash, config_json, git_commit,
                    data_source, data_source_limitations, data_granularity,
                    random_seed, date_range_start, date_range_end,
                    universe_size, universe_hash, notes
                ) VALUES (?,?,?, ?,?,?, ?,?,?, ?,?,?, ?,?,?)
                """,
                (
                    run_id, self._now(), RunStatus.RUNNING,
                    config_hash, json.dumps(config_json), git_commit,
                    data_source, data_source_limitations, data_granularity,
                    random_seed, date_range_start, date_range_end,
                    universe_size, universe_hash, notes,
                ),
            )
        return run_id

    def complete_run(self, run_id: str, metrics: dict[str, Any]) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status=?, completed_at=?, metrics_json=? WHERE run_id=?",
                (RunStatus.COMPLETED, self._now(), json.dumps(metrics), run_id),
            )

    def fail_run(self, run_id: str, error: str) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE runs SET status=?, completed_at=?, error=? WHERE run_id=?",
                (RunStatus.FAILED, self._now(), error, run_id),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        if result.get("config_json"):
            result["config_json"] = json.loads(result["config_json"])
        metrics_raw = result.pop("metrics_json", None)
        result["metrics"] = json.loads(metrics_raw) if metrics_raw else None
        return result

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
