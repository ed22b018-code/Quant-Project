"""Tests for the experiment tracker run log."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.experiment.tracker import ExperimentTracker, RunStatus

# ── Helpers ───────────────────────────────────────────────────────────────────

def _minimal_run(tracker: ExperimentTracker, **overrides) -> str:
    defaults = dict(
        config_hash="abc123",
        config_json={"project": {"name": "test"}},
        git_commit="deadbeef",
        data_source="stub_historical",
        data_source_limitations="Phase 0 stub",
        data_granularity="none",
        random_seed=42,
        date_range_start=None,
        date_range_end=None,
        universe_size=80,
        universe_hash="universe_hash_xyz",
    )
    defaults.update(overrides)
    return tracker.start_run(**defaults)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_start_run_returns_uuid(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "runs.db")
    run_id = _minimal_run(tracker)
    assert len(run_id) == 36   # UUID4 string length


def test_started_run_has_running_status(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "runs.db")
    run_id = _minimal_run(tracker)
    run = tracker.get_run(run_id)
    assert run is not None
    assert run["status"] == RunStatus.RUNNING


def test_get_run_fields(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "runs.db")
    run_id = _minimal_run(tracker)
    run = tracker.get_run(run_id)
    assert run["run_id"] == run_id
    assert run["random_seed"] == 42
    assert run["data_source"] == "stub_historical"
    assert run["data_granularity"] == "none"
    assert run["universe_size"] == 80
    assert run["metrics"] is None


def test_complete_run_updates_status(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "runs.db")
    run_id = _minimal_run(tracker)
    tracker.complete_run(run_id, metrics={"sharpe": 1.42, "n_trades": 37})
    run = tracker.get_run(run_id)
    assert run["status"] == RunStatus.COMPLETED
    assert run["completed_at"] is not None


def test_complete_run_stores_metrics(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "runs.db")
    run_id = _minimal_run(tracker)
    tracker.complete_run(run_id, metrics={"sharpe": 1.42, "n_trades": 37})
    run = tracker.get_run(run_id)
    assert run["metrics"]["sharpe"] == pytest.approx(1.42)
    assert run["metrics"]["n_trades"] == 37


def test_fail_run_updates_status(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "runs.db")
    run_id = _minimal_run(tracker)
    tracker.fail_run(run_id, error="KeyError: 'close'")
    run = tracker.get_run(run_id)
    assert run["status"] == RunStatus.FAILED
    assert run["error"] == "KeyError: 'close'"


def test_get_nonexistent_run_returns_none(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "runs.db")
    assert tracker.get_run("no-such-id") is None


def test_list_runs_empty_initially(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "runs.db")
    assert tracker.list_runs() == []


def test_list_runs_returns_all(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "runs.db")
    for _ in range(3):
        _minimal_run(tracker)
    assert len(tracker.list_runs()) == 3


def test_config_json_round_trips(tmp_path: Path) -> None:
    tracker = ExperimentTracker(tmp_path / "runs.db")
    payload = {"project": {"name": "quant-fin"}, "trading": {"token_pool": 500_000}}
    run_id = _minimal_run(tracker, config_json=payload)
    run = tracker.get_run(run_id)
    assert run["config_json"] == payload


def test_tracker_creates_db_dir(tmp_path: Path) -> None:
    """ExperimentTracker must create the parent directory if it doesn't exist."""
    nested = tmp_path / "a" / "b" / "runs.db"
    tracker = ExperimentTracker(nested)
    run_id = _minimal_run(tracker)
    assert tracker.get_run(run_id) is not None


def test_data_granularity_is_stored(tmp_path: Path) -> None:
    """Every run must record data_granularity — backtest reports depend on this."""
    tracker = ExperimentTracker(tmp_path / "runs.db")
    run_id = _minimal_run(tracker, data_granularity="daily")
    run = tracker.get_run(run_id)
    assert run["data_granularity"] == "daily"
