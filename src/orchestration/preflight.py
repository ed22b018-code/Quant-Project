"""Preflight checks for the Dhan bulk download orchestration.

Runs before the big download begins. Fails fast on any critical problem so
a multi-hour job never launches against a broken prerequisite.

Checks performed
----------------
1. DHAN_CLIENT_ID      — env var present (not blank)
2. DHAN_ACCESS_TOKEN   — env var present (not blank)
3. DHAN_ADJUSTMENT_STATUS — env var present (set to 'adjusted' or 'unadjusted')
4. Disk space          — >= min_free_gb free on the volume that holds db_path
5. DB parent writable  — parent directory is writable
6. API reachability    — optional; only when _connection_test_fn is provided
                         (in the orchestration, the TEST_FETCH stage handles this)

Usage
-----
    from src.orchestration.preflight import run_preflight, PreflightError
    try:
        run_preflight("data/market.duckdb")
    except PreflightError as exc:
        print(f"Preflight failed: {exc}")
        sys.exit(1)
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

_LOG = logging.getLogger(__name__)

MIN_FREE_GB: float = 25.0  # conservative estimate for ~5yr NSE intraday data


class PreflightError(RuntimeError):
    """Raised when one or more critical preflight checks fail."""


@dataclass
class CheckResult:
    """Result of a single preflight check."""
    name: str
    passed: bool
    detail: str
    critical: bool = True  # critical=False means warn, not block


@dataclass
class PreflightReport:
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks if c.critical)

    def print(self) -> None:
        for c in self.checks:
            icon = "  OK" if c.passed else ("FAIL" if c.critical else "WARN")
            print(f"  [{icon}] {c.name}: {c.detail}")

    def failed_critical(self) -> list[CheckResult]:
        return [c for c in self.checks if c.critical and not c.passed]


def run_preflight(
    db_path: str,
    *,
    min_free_gb: float = MIN_FREE_GB,
    _disk_usage_fn: Optional[Callable] = None,
    _connection_test_fn: Optional[Callable[[], bool]] = None,
    verbose: bool = True,
) -> PreflightReport:
    """Run all preflight checks.

    Parameters
    ----------
    db_path : str
        Path where the DuckDB will be written. Disk space is measured on
        the volume that contains this path's parent directory.
    min_free_gb : float
        Minimum free disk space required (default 25 GB).
    _disk_usage_fn : callable | None
        Injectable override for shutil.disk_usage (testing only).
    _connection_test_fn : callable | None
        If provided, called with no args and expected to return True/False.
        In production: pass a lambda that makes a lightweight HTTP call.
        In tests: pass a lambda that returns True/False without HTTP.
        If None, the connection check is skipped (the TEST_FETCH stage
        provides a richer connection + format check).
    verbose : bool
        Print the report to stdout.

    Returns
    -------
    PreflightReport with all check results.

    Raises
    ------
    PreflightError if any critical check fails.
    """
    checks: list[CheckResult] = []

    # ── Check 1: DHAN_CLIENT_ID ──────────────────────────────────────────────
    client_id = os.environ.get("DHAN_CLIENT_ID", "").strip()
    checks.append(CheckResult(
        name="DHAN_CLIENT_ID",
        passed=bool(client_id),
        detail="present" if client_id else "MISSING — add to .env file",
    ))

    # ── Check 2: DHAN_ACCESS_TOKEN ───────────────────────────────────────────
    token = os.environ.get("DHAN_ACCESS_TOKEN", "").strip()
    checks.append(CheckResult(
        name="DHAN_ACCESS_TOKEN",
        passed=bool(token),
        detail="present" if token else "MISSING — add to .env file",
    ))

    # ── Check 3: DHAN_ADJUSTMENT_STATUS ──────────────────────────────────────
    adj = os.environ.get("DHAN_ADJUSTMENT_STATUS", "").strip().lower()
    adj_valid = adj in ("adjusted", "unadjusted")
    checks.append(CheckResult(
        name="DHAN_ADJUSTMENT_STATUS",
        passed=adj_valid,
        detail=(
            f"'{adj}' (valid)" if adj_valid
            else f"'{adj}' — set to 'adjusted' or 'unadjusted' in .env"
        ),
    ))

    # ── Check 4: Disk space ───────────────────────────────────────────────────
    db_parent = Path(db_path).parent
    try:
        db_parent.mkdir(parents=True, exist_ok=True)
        disk_fn = _disk_usage_fn if _disk_usage_fn is not None else shutil.disk_usage
        usage = disk_fn(str(db_parent))
        free_gb = usage.free / (1024 ** 3)
        disk_ok = free_gb >= min_free_gb
        checks.append(CheckResult(
            name="Disk space",
            passed=disk_ok,
            detail=(
                f"{free_gb:.1f} GB free (need {min_free_gb:.0f} GB)"
                if disk_ok
                else f"Only {free_gb:.1f} GB free — need {min_free_gb:.0f} GB for 5yr data"
            ),
        ))
    except Exception as exc:
        checks.append(CheckResult(
            name="Disk space",
            passed=False,
            detail=f"Could not check disk space: {exc}",
        ))

    # ── Check 5: DB path writable ─────────────────────────────────────────────
    try:
        db_parent.mkdir(parents=True, exist_ok=True)
        writable = os.access(str(db_parent), os.W_OK)
        checks.append(CheckResult(
            name="DB path writable",
            passed=writable,
            detail=str(db_parent) if writable else f"Cannot write to {db_parent}",
        ))
    except Exception as exc:
        checks.append(CheckResult(
            name="DB path writable",
            passed=False,
            detail=f"Path check failed: {exc}",
        ))

    # ── Check 6: API reachability (optional) ─────────────────────────────────
    if _connection_test_fn is not None:
        try:
            reachable = bool(_connection_test_fn())
        except Exception as exc:
            reachable = False
            _LOG.debug("Connection test raised: %s", exc)
        checks.append(CheckResult(
            name="Dhan API reachable",
            passed=reachable,
            detail=(
                "HTTP connection succeeded"
                if reachable
                else "Cannot reach Dhan API — check network / credentials"
            ),
        ))

    report = PreflightReport(checks=checks)

    if verbose:
        print("Preflight checks:")
        report.print()

    failed = report.failed_critical()
    if failed:
        names = ", ".join(c.name for c in failed)
        raise PreflightError(
            f"Preflight failed on: {names}\n"
            f"Fix the issues above then re-run.\n"
            f"Credentials go in .env (never committed to git)."
        )

    _LOG.info("Preflight: all %d checks passed.", len(checks))
    return report
