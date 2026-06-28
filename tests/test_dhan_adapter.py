"""Tests for DhanDataSource and DhanFOMetricsSource.

All tests run against mocked responses — no real Dhan credentials or network
calls.  Covers: DataSource contract, SDK quarantine, PIT enforcement, tick
buffer, option chain cache/fallback, F&O metrics, and invariants.
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest

from src.data.base import DataLeakageError, DataNotAvailableError, DataSource
from src.data.dhan_adapter import DhanDataSource, DhanFOMetricsSource, FO_DAILY_METRICS_COLS
from src.data.historical import OHLCV_DAILY_COLS, OHLCV_MINUTE_COLS, OPTION_CHAIN_COLS, TICK_COLS
from src.db.duckdb_client import DuckDBClient
from src.db.schema import DUCKDB_SCHEMA_SQL

UTC = timezone.utc
IST = timezone(timedelta(hours=5, minutes=30))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def db(tmp_duckdb_path: Path) -> Path:
    """Temporary DuckDB with full schema."""
    with DuckDBClient(tmp_duckdb_path) as conn:
        conn.execute_schema(DUCKDB_SCHEMA_SQL)
    return tmp_duckdb_path


def _insert_intraday(db_path: Path, symbol: str, timeframe: str, bars: list[dict]) -> None:
    now = datetime.now(UTC)
    with DuckDBClient(db_path) as conn:
        for bar in bars:
            conn.execute(
                """INSERT INTO ohlcv_intraday
                   (symbol, timeframe, ts, open, high, low, close, volume, source, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'dhan_historical', ?)
                   ON CONFLICT DO NOTHING""",
                (symbol, timeframe, bar["ts"], bar["open"], bar["high"],
                 bar["low"], bar["close"], bar["volume"], now),
            )


def _insert_fo_metrics(db_path: Path, symbol: str, rows: list[dict]) -> None:
    now = datetime.now(UTC)
    with DuckDBClient(db_path) as conn:
        for r in rows:
            conn.execute(
                """INSERT INTO fo_daily_metrics
                   (symbol, date, pcr_oi, pcr_volume, total_ce_oi, total_pe_oi,
                    net_oi_change, max_pain_strike, futures_settle, source, ingested_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'dhan_fo_metrics', ?)
                   ON CONFLICT DO NOTHING""",
                (symbol, r["date"], r.get("pcr_oi"), r.get("pcr_volume"),
                 r.get("total_ce_oi", 0), r.get("total_pe_oi", 0),
                 r.get("net_oi_change", 0), r.get("max_pain_strike"),
                 r.get("futures_settle"), now),
            )


def _make_bars(symbol: str, timeframe: str, count: int = 5) -> list[dict]:
    """Create `count` synthetic intraday bars, each 1 minute apart."""
    base = datetime(2024, 3, 15, 9, 15, 0, tzinfo=UTC)
    return [
        {
            "ts":     base + timedelta(minutes=i),
            "open":   100.0 + i,
            "high":   102.0 + i,
            "low":    99.0  + i,
            "close":  101.0 + i,
            "volume": 1000 * (i + 1),
        }
        for i in range(count)
    ]


# ── Contract tests ────────────────────────────────────────────────────────────

def test_dhan_data_source_is_datasource_subclass(db):
    src = DhanDataSource(db_path=db)
    assert isinstance(src, DataSource)


def test_dhan_fo_metrics_source_is_datasource_subclass(db):
    src = DhanFOMetricsSource(db_path=db)
    assert isinstance(src, DataSource)


def test_metadata_fields_present(db):
    src = DhanDataSource(db_path=db)
    assert src.data_source_name
    assert src.data_granularity
    assert src.data_source_limitations


def test_dhan_fo_metrics_backtest_only(db):
    src = DhanFOMetricsSource(db_path=db)
    assert src.BACKTEST_ONLY is True


def test_dhan_data_source_backtest_only_false(db):
    src = DhanDataSource(db_path=db)
    assert src.BACKTEST_ONLY is False


# ── SDK quarantine ────────────────────────────────────────────────────────────

def test_sdk_quarantine_only_dhan_adapter_imports_dhanhq():
    """Only dhan_adapter.py may import dhanhq — grep all other src/ files."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "-c",
         "import ast, pathlib\n"
         "src = pathlib.Path('src')\n"
         "bad = []\n"
         "for f in src.rglob('*.py'):\n"
         "    if f.name == 'dhan_adapter.py': continue\n"
         "    try:\n"
         "        tree = ast.parse(f.read_text())\n"
         "    except: continue\n"
         "    for node in ast.walk(tree):\n"
         "        if isinstance(node, (ast.Import, ast.ImportFrom)):\n"
         "            names = [a.name for a in getattr(node, 'names', [])] + \\\n"
         "                    ([node.module] if hasattr(node, 'module') and node.module else [])\n"
         "            if any('dhanhq' in (n or '') for n in names): bad.append(str(f))\n"
         "print('|'.join(bad))"
        ],
        capture_output=True, text=True,
    )
    violations = [p for p in result.stdout.strip().split("|") if p]
    assert violations == [], f"dhanhq imported outside dhan_adapter.py: {violations}"


def test_no_gpu_imports_in_dhan_adapter():
    """dhan_adapter.py must not import torch, cupy, tensorflow, or jax."""
    import ast
    from pathlib import Path
    source = Path("src/data/dhan_adapter.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    gpu_libs = {"torch", "cupy", "tensorflow", "jax"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in gpu_libs, \
                    f"GPU library {alias.name!r} in dhan_adapter.py"
        elif isinstance(node, ast.ImportFrom):
            mod = (node.module or "").split(".")[0]
            assert mod not in gpu_libs, f"GPU library {node.module!r} in dhan_adapter.py"


# ── connect() without SDK ─────────────────────────────────────────────────────

def test_connect_without_dhanhq_raises_import_error(db, monkeypatch):
    """connect() raises ImportError with actionable message when dhanhq not installed."""
    import builtins, sys
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "dhanhq":
            raise ImportError("No module named 'dhanhq'")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    src = DhanDataSource(db_path=db)  # no _inject_ws → not connected
    with pytest.raises(ImportError, match="dhanhq"):
        src.connect()


# ── get_ohlcv_daily ───────────────────────────────────────────────────────────

def test_get_ohlcv_daily_empty_db_returns_empty_df(db, monkeypatch):
    monkeypatch.setenv("DHAN_ADJUSTMENT_STATUS", "PRE_ADJUSTED")
    src = DhanDataSource(db_path=db)
    as_of = datetime(2024, 3, 16, 10, 0, tzinfo=IST)
    df = src.get_ohlcv_daily("RELIANCE", as_of)
    assert df.empty
    assert list(df.columns) == list(OHLCV_DAILY_COLS.keys())


def test_get_ohlcv_daily_returns_correct_schema(db, monkeypatch):
    monkeypatch.setenv("DHAN_ADJUSTMENT_STATUS", "PRE_ADJUSTED")
    bars = [
        {"ts": datetime(2024, 3, 14, 15, 30, tzinfo=UTC),
         "open": 1400.0, "high": 1420.0, "low": 1390.0, "close": 1410.0, "volume": 500000},
    ]
    _insert_intraday(db, "RELIANCE", "D", bars)
    src = DhanDataSource(db_path=db)
    as_of = datetime(2024, 3, 15, 9, 15, tzinfo=IST)
    df = src.get_ohlcv_daily("RELIANCE", as_of)
    assert not df.empty
    assert list(df.columns) == list(OHLCV_DAILY_COLS.keys())
    assert df["open"].iloc[0] == pytest.approx(1400.0)


def test_get_ohlcv_daily_pit_block_future(db, monkeypatch):
    """Bar dated today should be blocked when as_of is today's start."""
    monkeypatch.setenv("DHAN_ADJUSTMENT_STATUS", "PRE_ADJUSTED")
    today = date(2024, 3, 15)
    bars = [
        {"ts": datetime(2024, 3, 15, 15, 30, tzinfo=UTC),
         "open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 10000},
    ]
    _insert_intraday(db, "RELIANCE", "D", bars)
    src = DhanDataSource(db_path=db)
    # as_of = midnight IST of 2024-03-15 → bar for 2024-03-15 is today, not past
    as_of = datetime(2024, 3, 15, 0, 0, tzinfo=IST)
    df = src.get_ohlcv_daily("RELIANCE", as_of)
    assert df.empty  # bar for today is blocked by PIT (date >= as_of.date())


def test_get_ohlcv_daily_provenance(db, monkeypatch):
    monkeypatch.setenv("DHAN_ADJUSTMENT_STATUS", "PRE_ADJUSTED")
    bars = [
        {"ts": datetime(2024, 3, 14, 15, 30, tzinfo=UTC),
         "open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 10000},
    ]
    _insert_intraday(db, "INFY", "D", bars)
    src = DhanDataSource(db_path=db)
    as_of = datetime(2024, 3, 15, 9, 0, tzinfo=IST)
    df = src.get_ohlcv_daily("INFY", as_of)
    assert not df.empty
    assert df["source"].iloc[0] == "dhan_historical"


def test_get_ohlcv_daily_warns_when_adjustment_unknown(db, monkeypatch, caplog):
    monkeypatch.delenv("DHAN_ADJUSTMENT_STATUS", raising=False)
    bars = [
        {"ts": datetime(2024, 3, 14, 15, 30, tzinfo=UTC),
         "open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 10000},
    ]
    _insert_intraday(db, "RELIANCE", "D", bars)
    src = DhanDataSource(db_path=db)
    import logging
    with caplog.at_level(logging.WARNING):
        src.get_ohlcv_daily("RELIANCE", datetime(2024, 3, 15, 9, 0, tzinfo=IST))
    assert "DHAN_ADJUSTMENT_STATUS" in caplog.text


# ── get_ohlcv_minute ──────────────────────────────────────────────────────────

def test_get_ohlcv_minute_returns_correct_schema(db, monkeypatch):
    monkeypatch.setenv("DHAN_ADJUSTMENT_STATUS", "PRE_ADJUSTED")
    bars = _make_bars("RELIANCE", "1")
    _insert_intraday(db, "RELIANCE", "1", bars)
    src = DhanDataSource(db_path=db)
    as_of = datetime(2024, 3, 15, 10, 0, tzinfo=UTC)
    df = src.get_ohlcv_minute("RELIANCE", as_of)
    assert not df.empty
    assert list(df.columns) == list(OHLCV_MINUTE_COLS.keys())


def test_get_ohlcv_minute_pit_block(db, monkeypatch):
    """Bar with ts AFTER as_of is excluded by the SQL WHERE clause — returns empty DF."""
    monkeypatch.setenv("DHAN_ADJUSTMENT_STATUS", "PRE_ADJUSTED")
    future_bar = [{"ts": datetime(2024, 3, 15, 10, 0, tzinfo=UTC),
                   "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0, "volume": 500}]
    _insert_intraday(db, "RELIANCE", "1", future_bar)
    src = DhanDataSource(db_path=db)
    # as_of is before the bar → SQL WHERE excludes it → empty result (no leakage)
    as_of = datetime(2024, 3, 15, 9, 30, tzinfo=UTC)
    df = src.get_ohlcv_minute("RELIANCE", as_of)
    assert df.empty


def test_get_ohlcv_minute_pit_enforce_raises_on_bad_data(db, monkeypatch):
    """_enforce_pit raises DataLeakageError if a future bar somehow bypasses the query."""
    monkeypatch.setenv("DHAN_ADJUSTMENT_STATUS", "PRE_ADJUSTED")
    src = DhanDataSource(db_path=db)
    # Simulate a future bar leaking through by calling _enforce_pit directly
    future_ts = datetime(2024, 3, 15, 10, 0, tzinfo=UTC)
    as_of = datetime(2024, 3, 15, 9, 30, tzinfo=UTC)
    bad_df = pd.DataFrame({"ts": [pd.Timestamp(future_ts)]})
    with pytest.raises(DataLeakageError):
        src._enforce_pit(bad_df, "ts", as_of)


def test_get_ohlcv_intraday_5min(db, monkeypatch):
    monkeypatch.setenv("DHAN_ADJUSTMENT_STATUS", "PRE_ADJUSTED")
    bars = _make_bars("TCS", "5")
    _insert_intraday(db, "TCS", "5", bars)
    src = DhanDataSource(db_path=db)
    as_of = datetime(2024, 3, 15, 10, 0, tzinfo=UTC)
    df = src.get_ohlcv_intraday("TCS", as_of, timeframe="5")
    assert not df.empty
    assert list(df.columns) == list(OHLCV_MINUTE_COLS.keys())


def test_get_ohlcv_intraday_invalid_timeframe(db):
    src = DhanDataSource(db_path=db)
    with pytest.raises(ValueError, match="timeframe"):
        src.get_ohlcv_intraday("RELIANCE", datetime.now(UTC), timeframe="3")


# ── get_ticks ─────────────────────────────────────────────────────────────────

def test_get_ticks_empty_buffer_returns_empty_df(db):
    src = DhanDataSource(db_path=db, _inject_ws=object())
    df = src.get_ticks("RELIANCE", datetime.now(UTC))
    assert df.empty
    assert list(df.columns) == list(TICK_COLS.keys())


def test_get_ticks_with_injected_ticks(db):
    src = DhanDataSource(db_path=db, _inject_ws=object())
    ts = datetime(2024, 3, 15, 9, 30, tzinfo=UTC)
    src._on_tick({"symbol": "RELIANCE", "ts": ts, "ltp": 1410.5,
                  "volume": 200, "bid": 1410.0, "ask": 1411.0, "oi": 5000})
    as_of = datetime(2024, 3, 15, 9, 31, tzinfo=UTC)
    df = src.get_ticks("RELIANCE", as_of)
    assert len(df) == 1
    assert df["ltp"].iloc[0] == pytest.approx(1410.5)
    assert df["source"].iloc[0] == "dhan_live"


def test_get_ticks_pit_block_future_tick(db):
    src = DhanDataSource(db_path=db, _inject_ws=object())
    future_ts = datetime(2024, 3, 15, 10, 0, tzinfo=UTC)
    src._on_tick({"symbol": "RELIANCE", "ts": future_ts, "ltp": 1410.5,
                  "volume": 200, "bid": 1410.0, "ask": 1411.0, "oi": 5000})
    as_of = datetime(2024, 3, 15, 9, 59, tzinfo=UTC)  # before tick
    with pytest.raises(DataLeakageError):
        src.get_ticks("RELIANCE", as_of)


def test_get_ticks_correct_tick_schema(db):
    src = DhanDataSource(db_path=db, _inject_ws=object())
    ts = datetime(2024, 3, 15, 9, 30, tzinfo=UTC)
    src._on_tick({"symbol": "TCS", "ts": ts, "ltp": 4000.0,
                  "volume": 100, "bid": 3999.0, "ask": 4001.0, "oi": 0})
    df = src.get_ticks("TCS", datetime(2024, 3, 15, 9, 31, tzinfo=UTC))
    assert list(df.columns) == list(TICK_COLS.keys())


# ── get_option_chain ──────────────────────────────────────────────────────────

def _oc_mock_success(symbol, _expiry):
    return {
        "status": "success",
        "data": [
            {
                "expiry": "2024-03-28",
                "strikePrice": 1400.0,
                "callOptions": {"marketData": {"ltp": 25.0, "oi": 5000, "prevOi": 4500, "volume": 100, "iv": 18.5}},
                "putOptions":  {"marketData": {"ltp": 15.0, "oi": 4000, "prevOi": 3800, "volume":  80, "iv": 17.2}},
            },
        ],
    }


def test_get_option_chain_mock_fetch(db):
    fetch_ts = datetime(2024, 3, 15, 9, 30, tzinfo=UTC)
    src = DhanDataSource(db_path=db, _mock_fetch=_oc_mock_success, _fetch_ts_override=fetch_ts)
    as_of = datetime(2024, 3, 15, 9, 31, tzinfo=UTC)
    df = src.get_option_chain("RELIANCE", as_of)
    assert not df.empty
    assert list(df.columns) == list(OPTION_CHAIN_COLS.keys())
    assert set(df["option_type"]) == {"CE", "PE"}


def test_get_option_chain_ttl_cache_hit(db):
    call_count = 0
    def mock_fetch(symbol, _expiry):
        nonlocal call_count
        call_count += 1
        return _oc_mock_success(symbol, _expiry)

    fetch_ts = datetime(2024, 3, 15, 9, 30, tzinfo=UTC)
    src = DhanDataSource(db_path=db, _mock_fetch=mock_fetch,
                         cache_ttl_seconds=60, _fetch_ts_override=fetch_ts)
    as_of = datetime(2024, 3, 15, 9, 31, tzinfo=UTC)
    src.get_option_chain("RELIANCE", as_of)
    src.get_option_chain("RELIANCE", as_of)
    assert call_count == 1  # second call is a cache hit


def test_get_option_chain_last_good_fallback(db):
    calls = [0]
    def fail_after_first(symbol, _expiry):
        calls[0] += 1
        if calls[0] == 1:
            return _oc_mock_success(symbol, _expiry)
        raise RuntimeError("network error")

    fetch_ts = datetime(2024, 3, 15, 9, 30, tzinfo=UTC)
    src = DhanDataSource(db_path=db, _mock_fetch=fail_after_first,
                         cache_ttl_seconds=0, max_retries=1,
                         _fetch_ts_override=fetch_ts)
    as_of = datetime(2024, 3, 15, 9, 31, tzinfo=UTC)
    df1 = src.get_option_chain("RELIANCE", as_of)  # first call succeeds
    df2 = src.get_option_chain("RELIANCE", as_of)  # second call fails → fallback
    assert not df2.empty
    assert len(df2) == len(df1)


def test_get_option_chain_no_fallback_raises(db):
    src = DhanDataSource(db_path=db, _mock_fetch=lambda s, e: (_ for _ in ()).throw(RuntimeError("fail")),
                         max_retries=1)
    with pytest.raises(DataNotAvailableError):
        src.get_option_chain("RELIANCE", datetime.now(UTC))


# ── get_news ──────────────────────────────────────────────────────────────────

def test_get_news_raises_data_not_available(db):
    src = DhanDataSource(db_path=db)
    with pytest.raises(DataNotAvailableError, match="news"):
        src.get_news("RELIANCE", datetime.now(UTC))


# ── DhanFOMetricsSource ───────────────────────────────────────────────────────

def test_fo_metrics_empty_db_returns_empty(db):
    src = DhanFOMetricsSource(db_path=db)
    as_of = datetime(2024, 3, 15, 10, 0, tzinfo=IST)
    df = src.get_fo_metrics("RELIANCE", as_of)
    assert df.empty
    assert list(df.columns) == list(FO_DAILY_METRICS_COLS.keys())


def test_fo_metrics_returns_correct_schema(db):
    _insert_fo_metrics(db, "RELIANCE", [
        {"date": date(2024, 3, 14), "pcr_oi": 1.2, "pcr_volume": 0.9,
         "total_ce_oi": 100000, "total_pe_oi": 120000, "net_oi_change": 5000,
         "max_pain_strike": 1400.0, "futures_settle": 1415.0},
    ])
    src = DhanFOMetricsSource(db_path=db)
    as_of = datetime(2024, 3, 15, 9, 0, tzinfo=IST)
    df = src.get_fo_metrics("RELIANCE", as_of)
    assert not df.empty
    assert list(df.columns) == list(FO_DAILY_METRICS_COLS.keys())
    assert df["pcr_oi"].iloc[0] == pytest.approx(1.2)


def test_fo_metrics_pit_block(db):
    _insert_fo_metrics(db, "RELIANCE", [
        {"date": date(2024, 3, 15), "pcr_oi": 1.1, "pcr_volume": 0.8,
         "total_ce_oi": 80000, "total_pe_oi": 90000, "net_oi_change": 1000,
         "max_pain_strike": 1390.0, "futures_settle": 1400.0},
    ])
    src = DhanFOMetricsSource(db_path=db)
    # as_of = midnight of 2024-03-15 → today's metric is blocked
    as_of = datetime(2024, 3, 15, 0, 0, tzinfo=IST)
    df = src.get_fo_metrics("RELIANCE", as_of)
    assert df.empty


def test_fo_metrics_standard_methods_raise(db):
    src = DhanFOMetricsSource(db_path=db)
    as_of = datetime.now(UTC)
    with pytest.raises(DataNotAvailableError):
        src.get_ohlcv_daily("RELIANCE", as_of)
    with pytest.raises(DataNotAvailableError):
        src.get_ohlcv_minute("RELIANCE", as_of)
    with pytest.raises(DataNotAvailableError):
        src.get_ticks("RELIANCE", as_of)
    with pytest.raises(DataNotAvailableError):
        src.get_news("RELIANCE", as_of)
    with pytest.raises(DataNotAvailableError):
        src.get_option_chain("RELIANCE", as_of)
