"""Tests for DhanBulkDownloader and DhanInstrumentMaster.

All tests use _mock_fetch — no real network calls, no Dhan credentials.
Key test: test_resume_after_interruption verifies that a killed-mid-run download
resumes correctly, skipping DONE chunks and fetching only pending ones.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data.dhan_downloader import DhanBulkDownloader, DhanInstrumentMaster, _FO_TIMEFRAME
from src.db.duckdb_client import DuckDBClient
from src.db.schema import DUCKDB_SCHEMA_SQL

UTC = timezone.utc

# ── Shared mock instrument master ─────────────────────────────────────────────

_MOCK_MASTER = {("RELIANCE", "EQUITY"): "1333", ("INFY", "EQUITY"): "1594"}


def _make_ohlcv_df(count: int = 5) -> pd.DataFrame:
    """Return a minimal DataFrame matching _parse_ohlcv_response output."""
    base = datetime(2024, 3, 15, 9, 15, tzinfo=UTC)
    return pd.DataFrame({
        "ts":     [pd.Timestamp(base + timedelta(minutes=i)) for i in range(count)],
        "open":   [100.0 + i for i in range(count)],
        "high":   [102.0 + i for i in range(count)],
        "low":    [99.0  + i for i in range(count)],
        "close":  [101.0 + i for i in range(count)],
        "volume": [1000  * (i + 1) for i in range(count)],
    })


def _make_fo_df(count: int = 3) -> pd.DataFrame:
    base = date(2024, 3, 15)
    rows = []
    for i in range(count):
        d = base + timedelta(days=i)
        rows.append({
            "date": d, "expiry_date": date(2024, 3, 28), "strike_price": 1400.0 + i * 50,
            "option_type": "CE", "open": 25.0, "high": 30.0, "low": 20.0,
            "close": 27.0, "settle_price": None, "contracts": 100,
            "oi": 5000, "change_in_oi": 200,
        })
    return pd.DataFrame(rows)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_duckdb_path: Path) -> Path:
    with DuckDBClient(tmp_duckdb_path) as conn:
        conn.execute_schema(DUCKDB_SCHEMA_SQL)
    return tmp_duckdb_path


def _dl(db_path: Path, mock_fetch=None) -> DhanBulkDownloader:
    return DhanBulkDownloader(
        db_path=db_path,
        request_delay_seconds=0,  # no sleep in tests
        days_per_chunk=30,
        _mock_fetch=mock_fetch,
        _mock_instrument_master=_MOCK_MASTER,
    )


def _mark_done(db_path: Path, symbol: str, timeframe: str, chunk_start: date, chunk_end: date) -> None:
    """Pre-seed a DONE job row (simulates a completed chunk from a previous run)."""
    now = datetime.now(UTC)
    with DuckDBClient(db_path) as conn:
        conn.execute(
            """INSERT INTO download_jobs
               (symbol, timeframe, chunk_start, chunk_end, status, rows_downloaded, last_attempted)
               VALUES (?, ?, ?, ?, 'done', 5, ?)
               ON CONFLICT DO NOTHING""",
            (symbol, timeframe, chunk_start, chunk_end, now),
        )


# ── Schema tests ──────────────────────────────────────────────────────────────

def test_schema_idempotency_download_jobs(db):
    with DuckDBClient(db) as conn:
        conn.execute_schema(DUCKDB_SCHEMA_SQL)  # run twice
        tables = conn.list_tables()
    assert "download_jobs" in tables


def test_schema_idempotency_ohlcv_intraday(db):
    with DuckDBClient(db) as conn:
        conn.execute_schema(DUCKDB_SCHEMA_SQL)
        tables = conn.list_tables()
    assert "ohlcv_intraday" in tables


def test_schema_idempotency_fo_history(db):
    with DuckDBClient(db) as conn:
        conn.execute_schema(DUCKDB_SCHEMA_SQL)
        tables = conn.list_tables()
    assert "fo_history" in tables


def test_schema_fo_daily_metrics_present(db):
    with DuckDBClient(db) as conn:
        tables = conn.list_tables()
    assert "fo_daily_metrics" in tables


# ── Chunk building ────────────────────────────────────────────────────────────

def test_build_chunks_single_chunk_under_limit(db):
    dl = _dl(db)
    chunks = dl._build_chunks(date(2024, 1, 1), date(2024, 1, 30))
    assert len(chunks) == 1
    assert chunks[0] == (date(2024, 1, 1), date(2024, 1, 30))


def test_build_chunks_exact_boundary_splits(db):
    dl = _dl(db)  # days_per_chunk=30
    # 31 days → 2 chunks
    chunks = dl._build_chunks(date(2024, 1, 1), date(2024, 1, 31))
    assert len(chunks) == 2
    assert chunks[0] == (date(2024, 1, 1), date(2024, 1, 30))
    assert chunks[1] == (date(2024, 1, 31), date(2024, 1, 31))


def test_build_chunks_no_overlap_or_gap(db):
    dl = _dl(db)
    chunks = dl._build_chunks(date(2024, 1, 1), date(2024, 3, 31))
    # Verify chunks are contiguous (no gaps, no overlaps)
    for i in range(1, len(chunks)):
        prev_end   = chunks[i - 1][1]
        curr_start = chunks[i][0]
        assert curr_start == prev_end + timedelta(days=1), \
            f"Gap/overlap between chunk {i-1} and {i}"


def test_build_chunks_5_years(db):
    dl_90 = DhanBulkDownloader(
        db_path=db, request_delay_seconds=0, days_per_chunk=90,
        _mock_instrument_master=_MOCK_MASTER,
    )
    import math
    chunks = dl_90._build_chunks(date(2019, 1, 1), date(2023, 12, 31))
    total_days = (date(2023, 12, 31) - date(2019, 1, 1)).days + 1
    assert len(chunks) == math.ceil(total_days / 90)


# ── Happy path ────────────────────────────────────────────────────────────────

def test_run_happy_path_marks_jobs_done(db):
    def mock_fetch(symbol, tf, start, end):
        return _make_ohlcv_df(3)

    dl = _dl(db, mock_fetch)
    dl.run(["RELIANCE"], ["1"], date(2024, 1, 1), date(2024, 1, 31))

    with DuckDBClient(db) as conn:
        df = conn.query_df("SELECT status FROM download_jobs WHERE symbol='RELIANCE' AND timeframe='1'")
    assert (df["status"] == "done").all()


def test_run_stores_rows_in_ohlcv_intraday(db):
    def mock_fetch(symbol, tf, start, end):
        return _make_ohlcv_df(5)

    dl = _dl(db, mock_fetch)
    dl.run(["RELIANCE"], ["1"], date(2024, 1, 1), date(2024, 1, 31))

    with DuckDBClient(db) as conn:
        df = conn.query_df("SELECT COUNT(*) AS n FROM ohlcv_intraday WHERE symbol='RELIANCE'")
    assert df["n"].iloc[0] == 5


# ── Idempotency ───────────────────────────────────────────────────────────────

def test_run_idempotent_no_duplicate_rows(db):
    """Re-running on the same range must not add duplicate rows."""
    def mock_fetch(symbol, tf, start, end):
        return _make_ohlcv_df(5)

    dl = _dl(db, mock_fetch)
    dl.run(["RELIANCE"], ["1"], date(2024, 1, 1), date(2024, 1, 31))
    dl.run(["RELIANCE"], ["1"], date(2024, 1, 1), date(2024, 1, 31))

    with DuckDBClient(db) as conn:
        df = conn.query_df("SELECT COUNT(*) AS n FROM ohlcv_intraday WHERE symbol='RELIANCE'")
    assert df["n"].iloc[0] == 5  # same 5 rows, not 10


# ── KEY TEST: Resume after interruption ───────────────────────────────────────

def test_resume_after_interruption(db):
    """
    Simulate a mid-run interruption:
    1. Pre-seed chunk 1 as DONE (rows already in ohlcv_intraday).
    2. Leave chunk 2 as PENDING.
    3. Run the downloader — it must fetch ONLY chunk 2.
    4. Verify total rows = 5 (from pre-seed) + 3 (from chunk 2 fetch).
    5. Verify fetch was called exactly once.
    """
    # 30-day chunks over a 60-day range → 2 chunks
    dl = _dl(db)  # days_per_chunk=30
    start = date(2024, 1, 1)
    end   = date(2024, 3, 1)
    chunks = dl._build_chunks(start, end)
    assert len(chunks) == 3  # 30+30+1 days

    # Pre-seed chunks[0] and chunks[1] as DONE with rows in ohlcv_intraday
    _mark_done(db, "RELIANCE", "1", chunks[0][0], chunks[0][1])
    _mark_done(db, "RELIANCE", "1", chunks[1][0], chunks[1][1])
    # Also insert synthetic rows for those done chunks
    now = datetime.now(UTC)
    with DuckDBClient(db) as conn:
        for i in range(5):
            conn.execute(
                """INSERT INTO ohlcv_intraday
                   (symbol, timeframe, ts, open, high, low, close, volume, source, ingested_at)
                   VALUES ('RELIANCE','1',?,100.0,102.0,99.0,101.0,1000,'dhan_historical',?)
                   ON CONFLICT DO NOTHING""",
                (datetime(2024, 1, 1, 9, 15, tzinfo=UTC) + timedelta(minutes=i), now),
            )

    # Now run: only chunk[2] (PENDING) should be fetched
    fetch_calls: list[tuple] = []
    def counting_mock(symbol, tf, start, end):
        fetch_calls.append((symbol, tf, start, end))
        return _make_ohlcv_df(3)

    dl2 = _dl(db, counting_mock)
    dl2.run(["RELIANCE"], ["1"], start, end)

    # Only 1 fetch call (for chunk[2])
    assert len(fetch_calls) == 1, f"Expected 1 fetch call, got {len(fetch_calls)}: {fetch_calls}"
    # DuckDB returns Timestamp for DATE columns; normalise both sides for comparison
    actual_start = fetch_calls[0][2]
    if hasattr(actual_start, "date"):
        actual_start = actual_start.date()
    assert actual_start == chunks[2][0]  # correct chunk_start

    # Total rows = 5 (pre-seeded) + 3 (from fetch) = 8
    with DuckDBClient(db) as conn:
        df = conn.query_df("SELECT COUNT(*) AS n FROM ohlcv_intraday WHERE symbol='RELIANCE'")
    assert df["n"].iloc[0] == 8


def test_resume_failed_chunks_are_retried(db):
    """FAILED chunks must be retried on restart (not skipped like DONE).
    Uses a 30-day range (days_per_chunk=30) → exactly 1 chunk.
    """
    dl = _dl(db)  # days_per_chunk=30
    start = date(2024, 1, 1)
    end   = date(2024, 1, 30)   # exactly 30 days → 1 chunk
    chunks = dl._build_chunks(start, end)
    assert len(chunks) == 1

    # Pre-seed that chunk as FAILED (simulating a previous failed run)
    with DuckDBClient(db) as conn:
        conn.execute(
            """INSERT INTO download_jobs
               (symbol, timeframe, chunk_start, chunk_end, status)
               VALUES ('RELIANCE','1',?,?,'failed')""",
            (chunks[0][0], chunks[0][1]),
        )

    fetch_calls = [0]
    def mock_fetch(symbol, tf, start, end):
        fetch_calls[0] += 1
        return _make_ohlcv_df(2)

    dl2 = _dl(db, mock_fetch)
    dl2.run(["RELIANCE"], ["1"], start, end)

    assert fetch_calls[0] == 1  # FAILED chunk was retried exactly once


# ── Error handling ────────────────────────────────────────────────────────────

def test_persistent_failure_marks_job_failed(db):
    """When all retries fail, chunk status becomes 'failed'."""
    dl = DhanBulkDownloader(
        db_path=db, request_delay_seconds=0, days_per_chunk=30,
        max_retries=1,
        _mock_fetch=lambda s, tf, st, en: None,  # always fails
        _mock_instrument_master=_MOCK_MASTER,
    )
    dl.run(["RELIANCE"], ["1"], date(2024, 1, 1), date(2024, 1, 31))

    with DuckDBClient(db) as conn:
        df = conn.query_df("SELECT status FROM download_jobs WHERE symbol='RELIANCE'")
    assert (df["status"] == "failed").all()


def test_empty_response_marks_partial(db):
    """Empty DataFrame from fetch → chunk marked 'partial', not 'done' or 'failed'."""
    dl = _dl(db, lambda s, tf, st, en: pd.DataFrame())
    dl.run(["RELIANCE"], ["1"], date(2024, 1, 1), date(2024, 1, 31))

    with DuckDBClient(db) as conn:
        df = conn.query_df("SELECT status FROM download_jobs WHERE symbol='RELIANCE'")
    assert (df["status"] == "partial").all()


def test_multiple_symbols_independent_status(db):
    """RELIANCE completes, INFY fails — statuses are independent."""
    def selective_mock(symbol, tf, start, end):
        return _make_ohlcv_df(3) if symbol == "RELIANCE" else None

    dl = _dl(db, selective_mock)
    dl.run(["RELIANCE", "INFY"], ["1"], date(2024, 1, 1), date(2024, 1, 31))

    with DuckDBClient(db) as conn:
        df = conn.query_df(
            "SELECT symbol, status FROM download_jobs WHERE timeframe='1' ORDER BY symbol"
        )
    status_map = dict(zip(df["symbol"], df["status"]))
    assert status_map["INFY"] == "failed"
    assert status_map["RELIANCE"] == "done"


# ── Provenance ────────────────────────────────────────────────────────────────

def test_source_is_dhan_historical(db):
    dl = _dl(db, lambda s, tf, st, en: _make_ohlcv_df(2))
    dl.run(["RELIANCE"], ["1"], date(2024, 1, 1), date(2024, 1, 31))
    with DuckDBClient(db) as conn:
        df = conn.query_df("SELECT DISTINCT source FROM ohlcv_intraday")
    assert df["source"].iloc[0] == "dhan_historical"


def test_ingested_at_is_utc_aware(db):
    dl = _dl(db, lambda s, tf, st, en: _make_ohlcv_df(2))
    dl.run(["RELIANCE"], ["1"], date(2024, 1, 1), date(2024, 1, 31))
    with DuckDBClient(db) as conn:
        df = conn.query_df("SELECT ingested_at FROM ohlcv_intraday LIMIT 1")
    ts = pd.to_datetime(df["ingested_at"].iloc[0], utc=True)
    assert ts.tzinfo is not None


# ── Coverage report ───────────────────────────────────────────────────────────

def test_coverage_report_no_jobs(db):
    dl = _dl(db)
    report = dl.coverage_report()
    assert "no download_jobs found" in report.lower() or "DHAN BULK" in report


def test_coverage_report_shows_complete(db):
    dl = _dl(db, lambda s, tf, st, en: _make_ohlcv_df(3))
    dl.run(["RELIANCE"], ["1"], date(2024, 1, 1), date(2024, 1, 31))
    report = dl.coverage_report()
    assert "COMPLETE" in report
    assert "RELIANCE" in report


def test_coverage_report_shows_has_errors(db):
    dl = _dl(db, lambda s, tf, st, en: None)
    dl.run(["RELIANCE"], ["1"], date(2024, 1, 1), date(2024, 1, 31))
    report = dl.coverage_report()
    assert "HAS_ERRORS" in report


def test_coverage_report_columns_present(db):
    dl = _dl(db, lambda s, tf, st, en: _make_ohlcv_df(2))
    dl.run(["RELIANCE", "INFY"], ["1", "5"],
           date(2024, 1, 1), date(2024, 1, 31))
    report = dl.coverage_report()
    for keyword in ["Symbol", "TF", "Done", "Total"]:
        assert keyword in report


# ── F&O download ──────────────────────────────────────────────────────────────

def test_run_fo_happy_path_marks_done(db):
    dl = _dl(db, lambda s, tf, st, en: _make_fo_df(3))
    dl.run_fo(["RELIANCE"], date(2024, 1, 1), date(2024, 1, 31))

    with DuckDBClient(db) as conn:
        df = conn.query_df(
            f"SELECT status FROM download_jobs WHERE symbol='RELIANCE' AND timeframe='{_FO_TIMEFRAME}'"
        )
    assert (df["status"] == "done").all()


def test_run_fo_stores_rows_in_fo_history(db):
    dl = _dl(db, lambda s, tf, st, en: _make_fo_df(3))
    dl.run_fo(["RELIANCE"], date(2024, 1, 1), date(2024, 1, 31))

    with DuckDBClient(db) as conn:
        df = conn.query_df("SELECT COUNT(*) AS n FROM fo_history WHERE symbol='RELIANCE'")
    assert df["n"].iloc[0] == 3


def test_run_fo_idempotent(db):
    dl = _dl(db, lambda s, tf, st, en: _make_fo_df(3))
    dl.run_fo(["RELIANCE"], date(2024, 1, 1), date(2024, 1, 31))
    dl.run_fo(["RELIANCE"], date(2024, 1, 1), date(2024, 1, 31))

    with DuckDBClient(db) as conn:
        df = conn.query_df("SELECT COUNT(*) AS n FROM fo_history WHERE symbol='RELIANCE'")
    assert df["n"].iloc[0] == 3


# ── compute_fo_metrics ────────────────────────────────────────────────────────

def _seed_fo_history(db_path: Path, symbol: str) -> None:
    now = datetime.now(UTC)
    rows = [
        # CE rows
        ("CE", 1400.0, 60000, 5000), ("CE", 1450.0, 20000, 1000),
        # PE rows
        ("PE", 1400.0, 80000, 3000), ("PE", 1350.0, 15000, -500),
        # FUT row
        ("FUT", 0.0, 0, 0),
    ]
    with DuckDBClient(db_path) as conn:
        for (opt, strike, oi, chg_oi) in rows:
            settle = 1415.0 if opt == "FUT" else None
            conn.execute(
                """INSERT INTO fo_history
                   (symbol, date, expiry_date, strike_price, option_type,
                    open, high, low, close, settle_price, contracts, oi, change_in_oi,
                    source, ingested_at)
                   VALUES (?, '2024-03-14', '2024-03-28', ?, ?,
                           ?, ?, ?, ?,
                           ?, 100, ?, ?, 'dhan_historical', ?)
                   ON CONFLICT DO NOTHING""",
                (symbol, strike, opt,
                 None, None, None, None,
                 settle, oi, chg_oi, now),
            )


def test_compute_fo_metrics_populates_fo_daily_metrics(db):
    _seed_fo_history(db, "RELIANCE")
    dl = _dl(db)
    rows = dl.compute_fo_metrics(["RELIANCE"])
    assert rows >= 1

    with DuckDBClient(db) as conn:
        df = conn.query_df(
            "SELECT * FROM fo_daily_metrics WHERE symbol='RELIANCE' AND date='2024-03-14'"
        )
    assert not df.empty
    assert df["total_ce_oi"].iloc[0] == 80000   # 60000 + 20000
    assert df["total_pe_oi"].iloc[0] == 95000   # 80000 + 15000


def test_compute_fo_metrics_pcr_correct(db):
    _seed_fo_history(db, "INFY")
    dl = _dl(db)
    dl.compute_fo_metrics(["INFY"])

    with DuckDBClient(db) as conn:
        df = conn.query_df(
            "SELECT pcr_oi FROM fo_daily_metrics WHERE symbol='INFY' AND date='2024-03-14'"
        )
    assert not df.empty
    expected_pcr = 80000 / 95000
    assert df["pcr_oi"].iloc[0] == pytest.approx(expected_pcr, rel=1e-3)


# ── Instrument master ─────────────────────────────────────────────────────────

def test_instrument_master_mock_lookup(db):
    master = DhanInstrumentMaster(db_path=db, _mock_master=_MOCK_MASTER)
    assert master.get_security_id("RELIANCE", "EQUITY") == "1333"
    assert master.get_security_id("INFY",     "EQUITY") == "1594"


def test_instrument_master_missing_returns_empty_string(db):
    master = DhanInstrumentMaster(db_path=db, _mock_master={})
    sec_id = master.get_security_id("UNKNOWNTICKER", "EQUITY")
    assert sec_id == ""


def test_instrument_master_refresh_with_mock_returns_count(db):
    master = DhanInstrumentMaster(
        db_path=db,
        _mock_master={("RELIANCE", "EQUITY"): "1333"},
    )
    count = master.refresh()
    assert count == 1
