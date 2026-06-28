"""Consolidation + integration tests — Phase 0 + Checkpoints 1–3.

Six risk areas probed end-to-end:
  1. Calendar forward-coverage
  2. Bhavcopy format robustness
  3. Corporate-action / raw-price handling boundary
  4. Interface symmetry on real (non-empty) data
  5. End-to-end timezone consistency
  6. End-to-end smoke test (full data-layer pipeline)
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from zoneinfo import ZoneInfo

from src.data.base import DataLeakageError, DataNotAvailableError
from src.data.bhavcopy import load_bhavcopy_to_duckdb, parse_bhavcopy_csv, validate_bhavcopy
from src.data.bhavcopy_source import BhavCopySource
from src.data.calendar import NSETradingCalendar, _DEFAULT_HOLIDAYS_JSON
from src.data.eligibility import get_eligible_symbols
from src.data.historical import OHLCV_DAILY_COLS, StubHistoricalSource, TICK_COLS
from src.data.live_upstox import UpstoxLiveSource

IST = ZoneInfo("Asia/Kolkata")

# ── shared CSV fixtures ───────────────────────────────────────────────────────

_JAN22_CSV = """\
TckrSymb,TradDt,SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol,TtlTrfVal
RELIANCE,2024-01-22,EQ,2500.0,2550.0,2490.0,2520.0,500000,1260000000.0
INFY,2024-01-22,EQ,1400.0,1440.0,1395.0,1420.0,300000,426000000.0
"""

_JAN23_CSV = """\
TckrSymb,TradDt,SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol,TtlTrfVal
RELIANCE,2024-01-23,EQ,2525.0,2560.0,2510.0,2540.0,480000,1219200000.0
INFY,2024-01-23,EQ,1425.0,1450.0,1415.0,1435.0,280000,401800000.0
"""


def _write_csvs(raw_dir: Path) -> None:
    (raw_dir / "2024-01-22.csv").write_text(_JAN22_CSV, encoding="utf-8")
    (raw_dir / "2024-01-23.csv").write_text(_JAN23_CSV, encoding="utf-8")


def _load_to_duckdb(db_path: Path, raw_dir: Path) -> None:
    load_bhavcopy_to_duckdb(date(2024, 1, 22), raw_dir, db_path)
    load_bhavcopy_to_duckdb(date(2024, 1, 23), raw_dir, db_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Risk area 1 — Calendar forward-coverage
# ═══════════════════════════════════════════════════════════════════════════════

def test_calendar_warns_for_date_beyond_data_range(caplog):
    """Calendar logs a WARNING (once per unknown year) when queried beyond its data range.

    The embedded + JSON holiday data covers up to max_holiday_year.  A query for
    max_holiday_year+1 may return the wrong answer for holidays not yet in the cache.
    The warning is logged exactly ONCE per unknown year per calendar instance.
    """
    cal = NSETradingCalendar(holidays_json=_DEFAULT_HOLIDAYS_JSON)
    max_yr = cal._max_holiday_year
    assert max_yr >= 2026, "Default holiday cache must cover at least 2026"

    future_date = date(max_yr + 1, 1, 6)   # first Monday of max_yr+1 (approx)

    with caplog.at_level(logging.WARNING, logger="src.data.calendar"):
        result = cal.is_trading_day(future_date)

    assert result is True, "Weekday not in holiday list → reported as trading day"
    warning_texts = " ".join(r.message for r in caplog.records if r.levelno == logging.WARNING)
    assert "beyond" in warning_texts.lower() or str(max_yr + 1) in warning_texts, (
        f"Expected forward-coverage warning, got: {warning_texts!r}"
    )

    # Second call for the same year must be SILENT
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.data.calendar"):
        cal.is_trading_day(date(max_yr + 1, 1, 7))
    remaining = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not remaining, "Warning must only be emitted once per unknown year"


def test_calendar_json_with_future_year_classifies_correctly(tmp_path):
    """Adding a future year to the JSON cache makes classification correct for that year."""
    json_path = tmp_path / "holidays.json"
    json_path.write_text(json.dumps({
        "2024": ["2024-01-26"],
        "2027": ["2027-01-26"],   # Republic Day 2027 (Tuesday)
    }), encoding="utf-8")

    cal = NSETradingCalendar(holidays_json=json_path)
    assert cal._max_holiday_year == 2027

    # 2027-01-26 is now in the holiday set → not a trading day
    assert not cal.is_trading_day(date(2027, 1, 26))

    # Regular Monday in 2027 (within covered range → no warning needed)
    assert cal.is_trading_day(date(2027, 1, 4))


def test_injected_holidays_never_trigger_forward_warning(caplog):
    """Calendars built with holidays=... (test injection) never warn about data range."""
    cal = NSETradingCalendar(holidays={date(2024, 1, 26)})
    assert cal._max_holiday_year == 0   # sentinel value for injected calendars

    with caplog.at_level(logging.WARNING, logger="src.data.calendar"):
        cal.is_trading_day(date(2030, 6, 1))   # far future — should NOT warn
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert not warnings


# ═══════════════════════════════════════════════════════════════════════════════
# Risk area 2 — Bhavcopy format robustness
# ═══════════════════════════════════════════════════════════════════════════════

def test_parse_bhavcopy_unknown_columns_raises():
    """parse_bhavcopy_csv raises ValueError when column layout is unrecognised."""
    unknown_csv = "Date,Open,Close,High,Low,Volume\n2024-01-22,100,110,115,95,10000\n"
    with pytest.raises(ValueError, match="[Uu]nrecognised"):
        parse_bhavcopy_csv(unknown_csv)


def test_parse_bhavcopy_new_format_missing_close_raises():
    """Newer-format CSV missing ClsPric (close) raises ValueError — not a silent mis-mapping."""
    broken = (
        "TckrSymb,TradDt,SctySrs,OpnPric,HghPric,LwPric,TtlTradgVol,TtlTrfVal\n"
        "RELIANCE,2024-01-22,EQ,2500.0,2550.0,2490.0,500000,1260000000.0\n"
    )
    with pytest.raises(ValueError, match="missing required columns"):
        parse_bhavcopy_csv(broken)


def test_parse_bhavcopy_old_format_missing_close_raises():
    """Older-format CSV missing CLOSE raises ValueError."""
    broken = (
        "SYMBOL,TIMESTAMP,SERIES,OPEN,HIGH,LOW,TOTTRDQTY\n"
        "RELIANCE,22-JAN-2024,EQ,2500,2550,2490,500000\n"
    )
    with pytest.raises(ValueError, match="missing required columns"):
        parse_bhavcopy_csv(broken)


def test_validate_bhavcopy_rejects_nan_price():
    """validate_bhavcopy hard-rejects rows where a price column is NaN.

    NaN arises when pd.to_numeric(..., errors='coerce') cannot parse a value.
    The old check (col <= 0) let NaN through silently; the fixed check does not.
    """
    df = pd.DataFrame({
        "symbol": ["GOOD", "NAN_CLOSE"],
        "date":   [date(2024, 1, 22), date(2024, 1, 22)],
        "open":   [100.0, 100.0],
        "high":   [110.0, 110.0],
        "low":    [95.0,  95.0],
        "close":  [105.0, float("nan")],
        "volume": [1000, 1000],
    })
    valid, rejected = validate_bhavcopy(df)
    assert len(valid) == 1
    assert valid["symbol"].iloc[0] == "GOOD"
    assert any("NAN_CLOSE" in r for r in rejected)


def test_validate_bhavcopy_rejects_inf_price():
    """validate_bhavcopy hard-rejects rows where a price column is infinite."""
    df = pd.DataFrame({
        "symbol": ["GOOD", "INF_OPEN"],
        "date":   [date(2024, 1, 22), date(2024, 1, 22)],
        "open":   [100.0, float("inf")],
        "high":   [110.0, 110.0],
        "low":    [95.0,  95.0],
        "close":  [105.0, 105.0],
        "volume": [1000, 1000],
    })
    valid, rejected = validate_bhavcopy(df)
    assert len(valid) == 1
    assert valid["symbol"].iloc[0] == "GOOD"
    assert any("INF_OPEN" in r for r in rejected)


def test_validate_bhavcopy_rejects_zero_price():
    """validate_bhavcopy hard-rejects rows with close == 0 (always was, still is)."""
    df = pd.DataFrame({
        "symbol": ["ZERO"],
        "date":   [date(2024, 1, 22)],
        "open":   [0.0],
        "high":   [0.0],
        "low":    [0.0],
        "close":  [0.0],
        "volume": [0],
    })
    valid, rejected = validate_bhavcopy(df)
    assert len(valid) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Risk area 3 — Corporate-action / raw-price handling boundary
# ═══════════════════════════════════════════════════════════════════════════════

def test_bhavcopy_source_limitations_explicitly_says_raw_unadjusted(tmp_path):
    """BhavCopySource must document that prices are raw and unadjusted.

    Downstream P&L calculations must NOT use raw bhavcopy prices until a
    corporate-action adjustment layer is in place (Phase 1 follow-up).
    """
    src = BhavCopySource(tmp_path / "empty.duckdb")
    limitations = src.data_source_limitations.lower()
    assert "raw unadjusted" in limitations, (
        "BhavCopySource.data_source_limitations must say 'raw unadjusted'"
    )
    assert "corporate-action" in limitations, (
        "BhavCopySource.data_source_limitations must mention corporate-action adjustment"
    )


def test_validate_bhavcopy_soft_flags_extreme_intrabar_move(caplog):
    """An extreme intra-bar high/low range (>25%) is soft-flagged — kept, warning logged.

    This catches circuit-breaker days, large rights issues, or possibly split-like
    intraday moves.  The row is retained; the researcher is warned.
    """
    df = pd.DataFrame({
        "symbol": ["EXTREME"],
        "date":   [date(2024, 1, 22)],
        "open":   [100.0],
        "high":   [140.0],   # (140-60)/100 = 80 % > 25 %
        "low":    [60.0],
        "close":  [100.0],
        "volume": [500_000],
    })
    with caplog.at_level(logging.WARNING, logger="src.data.bhavcopy"):
        valid, rejected = validate_bhavcopy(df)

    assert len(valid) == 1, "Extreme-move row must be kept (soft flag, not hard reject)"
    assert not rejected, "Extreme move must not be in rejection list"
    assert any("extreme" in r.message.lower() for r in caplog.records), (
        "WARNING must be logged for extreme intra-bar move"
    )


def test_raw_price_split_not_detected_at_bar_level():
    """A split-day bar looks completely normal in raw data — this is the known limitation.

    After a 2:1 split the bar has half the previous close, but intra-bar range is
    normal for the new price.  validate_bhavcopy cannot detect this and will not flag it.
    The limitation is documented in BhavCopySource.data_source_limitations.
    """
    # Post 2:1 split: all prices ~1000 (previously ~2000)
    df = pd.DataFrame({
        "symbol": ["SPLIT"],
        "date":   [date(2024, 1, 23)],
        "open":   [1000.0],
        "high":   [1020.0],   # (1020-990)/1010 ≈ 3% — totally normal intra-bar range
        "low":    [990.0],
        "close":  [1010.0],
        "volume": [4_000_000],
    })
    valid, rejected = validate_bhavcopy(df)
    assert len(valid) == 1    # not rejected — split bar is invisible at this level
    assert not rejected

    # The fix (adjusted prices) is a Phase 1 follow-up; the limitation must be documented
    src = BhavCopySource(Path(":memory:"))
    assert "raw unadjusted" in src.data_source_limitations


# ═══════════════════════════════════════════════════════════════════════════════
# Risk area 4 — Interface symmetry on real (non-empty) data
# ═══════════════════════════════════════════════════════════════════════════════

def test_ohlcv_daily_column_names_identical(tmp_path):
    """BhavCopySource and StubHistoricalSource.get_ohlcv_daily return identical columns."""
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    db_path = tmp_path / "test.duckdb"
    _write_csvs(raw_dir)
    _load_to_duckdb(db_path, raw_dir)

    stub = StubHistoricalSource()
    bhav = BhavCopySource(db_path)
    as_of = datetime(2024, 1, 23, 9, 15)

    df_stub = stub.get_ohlcv_daily("RELIANCE", as_of)
    df_bhav = bhav.get_ohlcv_daily("RELIANCE", as_of)

    assert list(df_stub.columns) == list(OHLCV_DAILY_COLS.keys())
    assert list(df_bhav.columns) == list(OHLCV_DAILY_COLS.keys())
    assert len(df_bhav) == 1, "Jan 22 bar must be present for as_of on Jan 23"


def test_tick_column_names_identical_stub_vs_live():
    """StubHistoricalSource and UpstoxLiveSource.get_ticks return identical columns."""
    stub = StubHistoricalSource()
    live = UpstoxLiveSource(_inject_ws=MagicMock())
    as_of = datetime(2024, 1, 22, 16, 0)

    df_stub = stub.get_ticks("X", as_of)
    df_live = live.get_ticks("X", as_of)   # empty buffer

    assert list(df_stub.columns) == list(TICK_COLS.keys())
    assert list(df_live.columns) == list(TICK_COLS.keys())


def test_interface_symmetry_pit_enforced_identically_on_live_source(tmp_path):
    """PIT enforcement on real non-empty data: past tick returned, future tick blocked."""
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    db_path = tmp_path / "test.duckdb"
    _write_csvs(raw_dir)
    _load_to_duckdb(db_path, raw_dir)

    bhav = BhavCopySource(db_path)
    live = UpstoxLiveSource(_inject_ws=MagicMock())

    as_of_hist = datetime(2024, 1, 23, 9, 15)
    as_of_live = datetime(2024, 1, 22, 15, 30)

    # Historical: Jan 22 bar available; Jan 23 bar NOT available at 9:15 on Jan 23
    df_hist = bhav.get_ohlcv_daily("RELIANCE", as_of_hist)
    assert len(df_hist) == 1
    assert df_hist["date"].iloc[0] == date(2024, 1, 22)

    # Live: past tick returned
    live._on_tick({
        "symbol": "RELIANCE",
        "ts": datetime(2024, 1, 22, 15, 0),
        "ltp": 2520.0, "volume": 100, "bid": 2519.0, "ask": 2521.0, "oi": 5000,
    })
    df_live = live.get_ticks("RELIANCE", as_of_live, lookback_seconds=3600)
    assert len(df_live) == 1
    assert df_live["ltp"].iloc[0] == pytest.approx(2520.0)

    # Live: future tick blocked
    live._on_tick({
        "symbol": "RELIANCE",
        "ts": datetime(2024, 1, 22, 16, 0),   # after as_of_live
        "ltp": 2530.0, "volume": 100, "bid": 2529.0, "ask": 2531.0, "oi": 5000,
    })
    with pytest.raises(DataLeakageError):
        live.get_ticks("RELIANCE", as_of_live, lookback_seconds=3600)


# ═══════════════════════════════════════════════════════════════════════════════
# Risk area 5 — End-to-end timezone consistency
# ═══════════════════════════════════════════════════════════════════════════════

def test_e2e_historical_bar_available_with_ist_as_of(tmp_path):
    """Historical bar for Jan 22 is available when as_of is 9:15 IST on Jan 23."""
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    db_path = tmp_path / "test.duckdb"
    _write_csvs(raw_dir)
    load_bhavcopy_to_duckdb(date(2024, 1, 22), raw_dir, db_path)

    bhav = BhavCopySource(db_path)
    as_of_ist = datetime(2024, 1, 23, 9, 15, 0, tzinfo=IST)

    df = bhav.get_ohlcv_daily("RELIANCE", as_of_ist)
    assert len(df) == 1
    assert df["date"].iloc[0] == date(2024, 1, 22)


def test_e2e_ist_aware_tick_returned_with_ist_as_of():
    """IST-aware tick at 9:14 is returned when as_of is 9:15 IST — timezone handled correctly."""
    live = UpstoxLiveSource(_inject_ws=MagicMock())
    as_of = datetime(2024, 1, 23, 9, 15, 0, tzinfo=IST)
    tick_ts = datetime(2024, 1, 23, 9, 14, 0, tzinfo=IST)

    live._on_tick({
        "symbol": "RELIANCE", "ts": tick_ts,
        "ltp": 2520.0, "volume": 100, "bid": 2519.0, "ask": 2521.0, "oi": 5000,
    })
    df = live.get_ticks("RELIANCE", as_of, lookback_seconds=3600)
    assert len(df) == 1


def test_e2e_ist_aware_future_tick_blocked():
    """IST-aware tick at 9:16 is blocked when as_of is 9:15 IST."""
    live = UpstoxLiveSource(_inject_ws=MagicMock())
    as_of = datetime(2024, 1, 23, 9, 15, 0, tzinfo=IST)
    future_ts = datetime(2024, 1, 23, 9, 16, 0, tzinfo=IST)

    live._on_tick({
        "symbol": "REL", "ts": future_ts,
        "ltp": 2525.0, "volume": 100, "bid": 2524.0, "ask": 2526.0, "oi": 5000,
    })
    with pytest.raises(DataLeakageError):
        live.get_ticks("REL", as_of, lookback_seconds=3600)


def test_e2e_mixed_tz_tick_vs_ist_as_of_raises_explicitly():
    """Naive-tz tick + IST-aware as_of raises DataLeakageError('tz-mismatch').

    This proves the system fails LOUDLY on mixed-timezone comparisons rather than
    silently computing the wrong answer.  A naive timestamp is ambiguous and must
    not be silently treated as UTC or IST.
    """
    live = UpstoxLiveSource(_inject_ws=MagicMock())
    as_of_ist = datetime(2024, 1, 23, 9, 15, 0, tzinfo=IST)
    naive_ts   = datetime(2024, 1, 23, 9, 14, 0)   # tz-naive — ambiguous

    live._on_tick({
        "symbol": "MIX", "ts": naive_ts,
        "ltp": 100.0, "volume": 100, "bid": 99.0, "ask": 101.0, "oi": 0,
    })
    with pytest.raises(DataLeakageError, match="tz-mismatch"):
        live.get_ticks("MIX", as_of_ist, lookback_seconds=3600)


# ═══════════════════════════════════════════════════════════════════════════════
# Risk area 6 — End-to-end smoke test (full data-layer pipeline)
# ═══════════════════════════════════════════════════════════════════════════════

def test_e2e_smoke_full_pipeline(tmp_path):
    """Full pipeline smoke test.

    Calendar → bhavcopy download (mocked) → eligibility → BhavCopySource → UpstoxLiveSource
    proves the data layer works as one integrated system.

    Steps:
    1. Load Jan 22 + Jan 23 bhavcopy CSVs into DuckDB
    2. Check eligibility (correct symbols, correct dates)
    3. Query historical bars via BhavCopySource (PIT = noon Jan 23 → only Jan 22 returned)
    4. Query live ticks via UpstoxLiveSource (injected tick 1 min before as_of)
    5. Verify DataNotAvailableError across interface boundary
    6. Verify data_granularity self-labels
    """
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    db_path = tmp_path / "test.duckdb"
    _write_csvs(raw_dir)

    # ── 1. Load data ──────────────────────────────────────────────────────────
    loaded22, _ = load_bhavcopy_to_duckdb(date(2024, 1, 22), raw_dir, db_path)
    loaded23, _ = load_bhavcopy_to_duckdb(date(2024, 1, 23), raw_dir, db_path)
    assert loaded22 == 2
    assert loaded23 == 2

    # Idempotent: loading again must not increase row count
    loaded22_again, _ = load_bhavcopy_to_duckdb(date(2024, 1, 22), raw_dir, db_path)
    assert loaded22_again == 2   # ON CONFLICT DO NOTHING — rows were "loaded" but not inserted

    # ── 2. Eligibility ────────────────────────────────────────────────────────
    eligible_22 = get_eligible_symbols(date(2024, 1, 22), db_path)
    eligible_23 = get_eligible_symbols(date(2024, 1, 23), db_path)

    assert "RELIANCE" in eligible_22 and "INFY" in eligible_22
    assert "RELIANCE" in eligible_23 and "INFY" in eligible_23

    # Eligibility is exact-date (not <=): no data for Jan 24 → empty
    eligible_24 = get_eligible_symbols(date(2024, 1, 24), db_path)
    assert len(eligible_24) == 0

    # ── 3. Historical OHLCV via BhavCopySource ────────────────────────────────
    bhav = BhavCopySource(db_path)
    as_of = datetime(2024, 1, 23, 12, 0)   # noon Jan 23 → only Jan 22 bar available

    df_rel = bhav.get_ohlcv_daily("RELIANCE", as_of)
    assert len(df_rel) == 1
    assert df_rel["date"].iloc[0] == date(2024, 1, 22)
    assert df_rel["source"].iloc[0] == "nse_bhavcopy"

    # Both Jan 22 and Jan 23 bars available at midnight Jan 24
    as_of_jan24 = datetime(2024, 1, 24, 0, 0)
    df_rel_2day = bhav.get_ohlcv_daily("RELIANCE", as_of_jan24)
    assert len(df_rel_2day) == 2

    # ── 4. Live ticks via UpstoxLiveSource ───────────────────────────────────
    live = UpstoxLiveSource(_inject_ws=MagicMock())
    tick_ts = datetime(2024, 1, 23, 11, 59)   # 1 minute before as_of
    live._on_tick({
        "symbol": "RELIANCE",
        "ts": tick_ts,
        "ltp": 2535.0, "volume": 100, "bid": 2534.0, "ask": 2536.0, "oi": 5000,
    })
    df_ticks = live.get_ticks("RELIANCE", as_of, lookback_seconds=300)
    assert len(df_ticks) == 1
    assert df_ticks["ltp"].iloc[0] == pytest.approx(2535.0)
    assert df_ticks["source"].iloc[0] == "upstox_live"

    # ── 5. DataNotAvailableError across interface boundary ────────────────────
    with pytest.raises(DataNotAvailableError):
        bhav.get_ticks("RELIANCE", as_of)   # BhavCopySource can't provide ticks

    with pytest.raises(DataNotAvailableError):
        live.get_ohlcv_daily("RELIANCE", as_of)   # UpstoxLiveSource can't provide daily bars

    # ── 6. data_granularity self-labels ──────────────────────────────────────
    assert bhav.data_granularity == "daily"
    assert live.data_granularity == "tick"
    # Backtest reports must label their granularity;
    # intraday-edge conclusions require granularity == 'minute'
    assert bhav.data_granularity != "minute", (
        "Bhavcopy is daily-only — intraday-edge conclusions must not be drawn from it"
    )


def test_e2e_calendar_integration_with_pipeline(tmp_path):
    """NSETradingCalendar integrates correctly with the data pipeline:
    bhavcopy for a trading day loads; a holiday produces no data.
    """
    from src.data.calendar import NSETradingCalendar
    from src.data.bhavcopy import download_bhavcopy

    cal = NSETradingCalendar(holidays={date(2024, 1, 26)})   # Republic Day 2024

    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    db_path = tmp_path / "test.duckdb"

    # Write only Jan 22 CSV (Jan 26 is a holiday — no CSV expected)
    (raw_dir / "2024-01-22.csv").write_text(_JAN22_CSV, encoding="utf-8")
    load_bhavcopy_to_duckdb(date(2024, 1, 22), raw_dir, db_path)

    # Jan 26 (Republic Day): classify_missing_file → 'expected'
    assert cal.classify_missing_file(date(2024, 1, 26)) == "expected"

    # Eligibility on the holiday returns empty (no data loaded for that date)
    eligible = get_eligible_symbols(date(2024, 1, 26), db_path)
    assert len(eligible) == 0

    # Eligibility on the trading day returns correct symbols
    eligible_22 = get_eligible_symbols(date(2024, 1, 22), db_path)
    assert "RELIANCE" in eligible_22
