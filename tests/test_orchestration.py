"""Tests for orchestration infrastructure: preflight, test_fetch, Orchestrator.

All tests use injectable mocks — no real API calls, no disk I/O beyond tmp dirs.

Coverage
--------
  preflight  : env var checks, disk space, DB writable, connection check
  test_fetch : column validation, days>5 guard, pass/partial/fail status
  orchestrator: stage ordering, checkpointing, resume, start_from, GPU detection
  RUNBOOK    : file exists
"""
from __future__ import annotations

import json
import os
import types
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest

# ── helpers ────────────────────────────────────────────────────────────────────

def _good_ohlcv(n: int = 5) -> pd.DataFrame:
    """Return a minimal DataFrame with all expected columns."""
    return pd.DataFrame({
        "ts": pd.date_range("2024-01-02", periods=n, freq="1min", tz="UTC"),
        "open": [100.0] * n,
        "high": [101.0] * n,
        "low":  [99.0] * n,
        "close":[100.5] * n,
        "volume": [1000] * n,
    })


def _mock_downloader_ok() -> MagicMock:
    dl = MagicMock()
    dl._download_chunk.return_value = _good_ohlcv()
    dl.run.return_value = None
    dl.run_fo.return_value = None
    dl.compute_fo_metrics.return_value = 0
    dl.coverage_report.return_value = "coverage OK"
    return dl


# ── preflight tests ────────────────────────────────────────────────────────────

class TestPreflight:
    def _env(self, **overrides) -> dict:
        base = {
            "DHAN_CLIENT_ID": "client123",
            "DHAN_ACCESS_TOKEN": "token456",
            "DHAN_ADJUSTMENT_STATUS": "adjusted",
        }
        base.update(overrides)
        return base

    def _run(self, tmp_path, env_overrides=None, **kwargs):
        from src.orchestration.preflight import run_preflight

        env = self._env(**(env_overrides or {}))
        old = {k: os.environ.get(k) for k in env}
        for k, v in env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        try:
            return run_preflight(
                str(tmp_path / "market.duckdb"),
                _disk_usage_fn=lambda p: types.SimpleNamespace(
                    free=30 * 1024 ** 3, used=0, total=30 * 1024 ** 3
                ),
                verbose=False,
                **kwargs,
            )
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_preflight_passes_with_all_good(self, tmp_path):
        from src.orchestration.preflight import run_preflight
        report = self._run(tmp_path)
        assert report.ok

    def test_preflight_fails_missing_client_id(self, tmp_path):
        from src.orchestration.preflight import PreflightError
        with pytest.raises(PreflightError, match="DHAN_CLIENT_ID"):
            self._run(tmp_path, env_overrides={"DHAN_CLIENT_ID": ""})

    def test_preflight_fails_missing_access_token(self, tmp_path):
        from src.orchestration.preflight import PreflightError
        with pytest.raises(PreflightError, match="DHAN_ACCESS_TOKEN"):
            self._run(tmp_path, env_overrides={"DHAN_ACCESS_TOKEN": ""})

    def test_preflight_fails_bad_adjustment_status(self, tmp_path):
        from src.orchestration.preflight import PreflightError
        with pytest.raises(PreflightError, match="DHAN_ADJUSTMENT_STATUS"):
            self._run(tmp_path, env_overrides={"DHAN_ADJUSTMENT_STATUS": "bad_value"})

    def test_preflight_adjustment_status_unadjusted_ok(self, tmp_path):
        report = self._run(tmp_path, env_overrides={"DHAN_ADJUSTMENT_STATUS": "unadjusted"})
        assert report.ok

    def test_preflight_fails_low_disk(self, tmp_path):
        from src.orchestration.preflight import PreflightError, run_preflight
        import types, os

        env = self._env()
        old = {k: os.environ.get(k) for k in env}
        for k, v in env.items():
            os.environ[k] = v
        try:
            with pytest.raises(PreflightError, match="Disk space"):
                run_preflight(
                    str(tmp_path / "market.duckdb"),
                    _disk_usage_fn=lambda p: types.SimpleNamespace(
                        free=5 * 1024 ** 3, used=0, total=5 * 1024 ** 3
                    ),
                    verbose=False,
                )
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_preflight_connection_check_pass(self, tmp_path):
        report = self._run(tmp_path, _connection_test_fn=lambda: True)
        assert report.ok
        names = [c.name for c in report.checks]
        assert "Dhan API reachable" in names

    def test_preflight_connection_check_fail(self, tmp_path):
        from src.orchestration.preflight import PreflightError
        with pytest.raises(PreflightError, match="Dhan API reachable"):
            self._run(tmp_path, _connection_test_fn=lambda: False)

    def test_preflight_report_has_five_base_checks(self, tmp_path):
        report = self._run(tmp_path)
        assert len(report.checks) == 5

    def test_preflight_error_lists_all_failed_names(self, tmp_path):
        from src.orchestration.preflight import PreflightError
        with pytest.raises(PreflightError) as exc_info:
            self._run(
                tmp_path,
                env_overrides={
                    "DHAN_CLIENT_ID": "",
                    "DHAN_ACCESS_TOKEN": "",
                },
            )
        msg = str(exc_info.value)
        assert "DHAN_CLIENT_ID" in msg
        assert "DHAN_ACCESS_TOKEN" in msg


# ── test_fetch tests ───────────────────────────────────────────────────────────

class TestTestFetch:
    def test_days_over_5_raises_value_error(self):
        from scripts.test_fetch import test_fetch
        with pytest.raises(ValueError, match="SMALL probe"):
            test_fetch(days=6, _mock_fetch=lambda *a, **kw: _good_ohlcv(), _quiet=True)

    def test_pass_with_good_columns(self):
        from scripts.test_fetch import test_fetch

        def mock_fetch(symbol, tf, start, end):
            return _good_ohlcv()

        result = test_fetch(_mock_fetch=mock_fetch, _quiet=True)
        assert result["status"] == "pass"
        assert result["missing_cols"] == []
        assert result["rows"] == 5
        assert result["fix_location"] is None

    def test_partial_with_missing_columns(self):
        from scripts.test_fetch import test_fetch

        def mock_fetch(symbol, tf, start, end):
            df = _good_ohlcv()
            return df.drop(columns=["volume"])  # remove one expected column

        result = test_fetch(_mock_fetch=mock_fetch, _quiet=True)
        assert result["status"] == "partial"
        assert "volume" in result["missing_cols"]
        assert result["fix_location"] is not None
        assert "_parse_ohlcv_response" in result["fix_location"]

    def test_fail_with_empty_dataframe(self):
        from scripts.test_fetch import test_fetch

        def mock_fetch(symbol, tf, start, end):
            return pd.DataFrame()

        result = test_fetch(_mock_fetch=mock_fetch, _quiet=True)
        assert result["status"] == "fail"
        assert result["rows"] == 0

    def test_fail_with_none_response(self):
        from scripts.test_fetch import test_fetch

        def mock_fetch(symbol, tf, start, end):
            return None

        result = test_fetch(_mock_fetch=mock_fetch, _quiet=True)
        assert result["status"] == "fail"

    def test_result_has_required_keys(self):
        from scripts.test_fetch import test_fetch

        result = test_fetch(_mock_fetch=lambda *a, **k: _good_ohlcv(), _quiet=True)
        for key in ("status", "symbol", "days", "rows", "columns", "missing_cols", "message"):
            assert key in result, f"Missing key: {key}"

    def test_days_boundary_5_ok(self):
        from scripts.test_fetch import test_fetch

        result = test_fetch(days=5, _mock_fetch=lambda *a, **k: _good_ohlcv(), _quiet=True)
        assert result["days"] == 5


# ── orchestrator tests ─────────────────────────────────────────────────────────

class TestOrchestrator:
    def _orch(self, tmp_path, **mock_overrides):
        from src.orchestration.orchestrator import Orchestrator

        defaults = dict(
            _mock_preflight_fn=lambda: None,
            _mock_test_fetch_fn=lambda: {"status": "pass", "rows": 5},
            _mock_downloader=_mock_downloader_ok(),
            _mock_run_training_fn=lambda: None,
        )
        defaults.update(mock_overrides)
        return Orchestrator(
            db_path=str(tmp_path / "market.duckdb"),
            output_dir=str(tmp_path / "models"),
            universe=["RELIANCE"],
            start_date=date(2024, 1, 1),
            end_date=date(2024, 6, 30),
            **defaults,
        )

    def test_dry_run_completes_all_stages(self, tmp_path):
        orch = self._orch(tmp_path)
        summary = orch.run(dry_run=True)
        assert "stages_run" in summary
        assert summary["dry_run"] is True

    def test_stage_list_has_eight_entries(self):
        from src.orchestration.orchestrator import Orchestrator, Stage
        assert len(Orchestrator.STAGES) == 8
        assert Orchestrator.STAGES[0] == Stage.PREFLIGHT
        assert Orchestrator.STAGES[-1] == Stage.SAVE_SUMMARY

    def test_completed_stages_are_skipped(self, tmp_path):
        orch = self._orch(tmp_path)
        summary1 = orch.run(dry_run=True)
        first_run_count = len(summary1["stages_run"])

        # Second run — all stages should be in checkpoint now
        summary2 = orch.run(dry_run=True)
        assert len(summary2["stages_run"]) < first_run_count

    def test_start_from_clears_later_stages(self, tmp_path):
        from src.orchestration.orchestrator import Stage
        orch = self._orch(tmp_path)
        orch.run(dry_run=True)  # complete all stages

        # Re-run starting from TRAIN_MODELS
        summary = orch.run(dry_run=True, start_from=Stage.TRAIN_MODELS)
        assert Stage.TRAIN_MODELS.value in summary["stages_run"]

    def test_preflight_failure_blocks_pipeline(self, tmp_path):
        from src.orchestration.preflight import PreflightError

        def bad_preflight():
            raise PreflightError("env var missing")

        orch = self._orch(tmp_path, _mock_preflight_fn=bad_preflight)
        with pytest.raises(PreflightError):
            orch.run(dry_run=True)

    def test_skip_test_fetch_flag(self, tmp_path):
        from src.orchestration.orchestrator import Stage
        fetch_calls = []

        def track_fetch():
            fetch_calls.append(1)
            return {"status": "pass", "rows": 5}

        orch = self._orch(tmp_path, _mock_test_fetch_fn=track_fetch)
        orch.run(dry_run=True, skip_test_fetch=True)
        assert len(fetch_calls) == 0

    def test_checkpoint_file_written(self, tmp_path):
        orch = self._orch(tmp_path)
        orch.run(dry_run=True)
        state_path = tmp_path / "models" / ".orchestration_state.json"
        assert state_path.exists()
        data = json.loads(state_path.read_text(encoding="utf-8"))
        assert "completed" in data
        assert len(data["completed"]) > 0

    def test_non_preflight_failure_includes_start_from_hint(self, tmp_path):
        def bad_training():
            raise RuntimeError("GPU OOM")

        orch = self._orch(tmp_path, _mock_run_training_fn=bad_training)
        with pytest.raises(RuntimeError, match="--start-from"):
            orch.run(dry_run=False)


# ── GPU detection ──────────────────────────────────────────────────────────────

class TestDetectLgbmDevice:
    def test_returns_cpu_when_subprocess_fails(self, monkeypatch):
        import subprocess

        def always_fail(*args, **kwargs):
            raise Exception("simulated GPU probe failure")

        monkeypatch.setattr(subprocess, "check_output", always_fail)
        from src.models.training_pipeline import _detect_lgbm_device
        result = _detect_lgbm_device()
        assert result == ("cpu", 0, 0)

    def test_returns_tuple_type(self):
        from src.models.training_pipeline import _detect_lgbm_device
        result = _detect_lgbm_device()
        assert isinstance(result, tuple)
        assert len(result) == 3
        device_type, platform_id, device_id = result
        assert device_type in ("cpu", "gpu")
        assert isinstance(platform_id, int)
        assert isinstance(device_id, int)

    def test_no_hardcoded_device_strings_in_training_layer(self):
        root = Path(__file__).parent.parent
        model_files = list((root / "src" / "models").glob("*.py"))
        for py_file in model_files:
            src = py_file.read_text(encoding="utf-8")
            assert '"cuda"' not in src, f"{py_file.name} hard-codes 'cuda'"
            assert "'cuda'" not in src, f"{py_file.name} hard-codes 'cuda'"


# ── RUNBOOK exists ─────────────────────────────────────────────────────────────

class TestRunbook:
    def test_runbook_file_exists(self):
        root = Path(__file__).parent.parent
        assert (root / "RUNBOOK.md").exists()

    def test_runbook_mentions_five_steps(self):
        root = Path(__file__).parent.parent
        text = (root / "RUNBOOK.md").read_text(encoding="utf-8")
        # Each step header must be present
        for step in ("Step 1", "Step 2", "Step 3", "Step 4", "Step 5"):
            assert step in text, f"RUNBOOK.md missing '{step}'"

    def test_runbook_mentions_gitignore_for_env(self):
        root = Path(__file__).parent.parent
        text = (root / "RUNBOOK.md").read_text(encoding="utf-8")
        assert "git-ignored" in text or "gitignore" in text.lower()

    def test_runbook_mentions_fix_location(self):
        root = Path(__file__).parent.parent
        text = (root / "RUNBOOK.md").read_text(encoding="utf-8")
        assert "_parse_ohlcv_response" in text
