"""
Tests that DataSource._enforce_pit blocks all future-dated rows.

These are the primary mechanical defense against data leakage.  A failure here
means backtest results may be fictional — treat any regression as critical.

Coverage:
  A. Empty DataFrame always passes.
  B. Daily bars (date column): boundary, future, mixed past+future.
  C. Datetime columns: past, exact boundary (strict <), microsecond precision,
     future, one-row-past-among-future.
  D. Timezone handling: both-tz-aware (cross-zone correct), both-naive,
     mixed-tz raises immediately (no silent wrong comparison).
  E. News: PIT column is first_knowable_at — NOT published_at. Explicit proof
     that using the wrong column would allow leakage.
  F. Option chain: ts column, same datetime rules.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from src.data.base import DataLeakageError, DataSource

IST = ZoneInfo("Asia/Kolkata")   # UTC+05:30

# ── Test double ───────────────────────────────────────────────────────────────

class _InjectableSource(DataSource):
    """DataSource that returns whatever DataFrames we inject — for testing only."""

    data_source_name = "injectable_test"
    data_source_limitations = "test double"
    data_granularity = "none"

    def __init__(self) -> None:
        self._daily: pd.DataFrame = pd.DataFrame()
        self._minute: pd.DataFrame = pd.DataFrame()
        self._ticks: pd.DataFrame = pd.DataFrame()
        self._news: pd.DataFrame = pd.DataFrame()
        self._options: pd.DataFrame = pd.DataFrame()

    def get_ohlcv_daily(self, symbol, as_of, lookback_days=252):
        return self._enforce_pit(self._daily.copy(), "date", as_of)

    def get_ohlcv_minute(self, symbol, as_of, lookback_bars=390):
        return self._enforce_pit(self._minute.copy(), "ts", as_of)

    def get_ticks(self, symbol, as_of, lookback_seconds=300):
        return self._enforce_pit(self._ticks.copy(), "ts", as_of)

    def get_news(self, symbol, as_of, lookback_hours=24.0):
        return self._enforce_pit(self._news.copy(), "first_knowable_at", as_of)

    def get_option_chain(self, symbol, as_of):
        return self._enforce_pit(self._options.copy(), "ts", as_of)


# ── Reference timestamps ──────────────────────────────────────────────────────
# T = prediction time in all tests below.

T_NAIVE = datetime(2024, 6, 15, 10, 30, 0, 0)                    # tz-naive
T_UTC   = datetime(2024, 6, 15, 10, 30, 0, 0, tzinfo=timezone.utc)  # UTC-aware
# 10:30 UTC = 16:00 IST
T_IST   = datetime(2024, 6, 15, 16,  0, 0, 0, tzinfo=IST)           # IST-aware (same instant as T_UTC)


# ── A. Empty DataFrame ────────────────────────────────────────────────────────

def test_empty_dataframe_always_passes() -> None:
    src = _InjectableSource()
    src._daily = pd.DataFrame({"date": pd.Series([], dtype="object"), "close": []})
    assert src.get_ohlcv_daily("X", T_UTC).empty


def test_all_null_ts_column_always_passes() -> None:
    src = _InjectableSource()
    src._minute = pd.DataFrame({"ts": [None, None], "close": [1.0, 2.0]})
    assert len(src.get_ohlcv_minute("X", T_NAIVE)) == 2


# ── B. Daily bars (date column) ───────────────────────────────────────────────

def test_daily_bars_strictly_before_as_of_allowed() -> None:
    src = _InjectableSource()
    src._daily = pd.DataFrame({
        "date": [date(2024, 6, 13), date(2024, 6, 14)],
        "close": [100.0, 101.0],
    })
    assert len(src.get_ohlcv_daily("X", T_UTC)) == 2


def test_daily_bar_same_date_as_as_of_is_blocked() -> None:
    """Bar for the same date as as_of is blocked (conservative: bar D knowable from D+1)."""
    src = _InjectableSource()
    src._daily = pd.DataFrame({"date": [date(2024, 6, 15)], "close": [102.0]})
    with pytest.raises(DataLeakageError, match="PIT violation"):
        src.get_ohlcv_daily("X", T_UTC)


def test_daily_bar_future_date_blocked() -> None:
    src = _InjectableSource()
    src._daily = pd.DataFrame({"date": [date(2024, 6, 16)], "close": [103.0]})
    with pytest.raises(DataLeakageError):
        src.get_ohlcv_daily("X", T_UTC)


def test_daily_bar_one_past_one_future_both_blocked() -> None:
    """Even one future row among past rows must raise — not silently filtered."""
    src = _InjectableSource()
    src._daily = pd.DataFrame({
        "date": [date(2024, 6, 14), date(2024, 6, 15)],
        "close": [100.0, 101.0],
    })
    with pytest.raises(DataLeakageError):
        src.get_ohlcv_daily("X", T_UTC)


# ── C. Datetime columns — boundary and precision ──────────────────────────────

def test_naive_past_allowed() -> None:
    src = _InjectableSource()
    src._ticks = pd.DataFrame({"ts": [T_NAIVE - timedelta(minutes=1)], "ltp": [2400.0]})
    assert len(src.get_ticks("X", T_NAIVE)) == 1


def test_naive_exact_boundary_is_blocked() -> None:
    """Row stamped exactly at T (ts == as_of) must be blocked — rule is strict <."""
    src = _InjectableSource()
    src._ticks = pd.DataFrame({"ts": [T_NAIVE], "ltp": [2400.0]})
    with pytest.raises(DataLeakageError, match="PIT violation"):
        src.get_ticks("X", T_NAIVE)


def test_naive_one_microsecond_before_boundary_is_allowed() -> None:
    """T − 1 μs must be allowed; ensures the boundary is strictly < not <=."""
    src = _InjectableSource()
    src._ticks = pd.DataFrame({"ts": [T_NAIVE - timedelta(microseconds=1)], "ltp": [2400.0]})
    assert len(src.get_ticks("X", T_NAIVE)) == 1


def test_naive_future_blocked() -> None:
    src = _InjectableSource()
    src._ticks = pd.DataFrame({"ts": [T_NAIVE + timedelta(minutes=1)], "ltp": [2400.5]})
    with pytest.raises(DataLeakageError):
        src.get_ticks("X", T_NAIVE)


def test_naive_minute_bar_exact_boundary_blocked() -> None:
    src = _InjectableSource()
    src._minute = pd.DataFrame({"ts": [T_NAIVE], "close": [100.0]})
    with pytest.raises(DataLeakageError):
        src.get_ohlcv_minute("X", T_NAIVE)


# ── D. Timezone handling ──────────────────────────────────────────────────────

# D-1  Both tz-aware, same zone

def test_utc_aware_exact_boundary_blocked() -> None:
    """Both column and as_of are UTC-aware: row at exactly T is blocked."""
    src = _InjectableSource()
    src._ticks = pd.DataFrame({"ts": [T_UTC], "ltp": [2400.0]})
    with pytest.raises(DataLeakageError, match="PIT violation"):
        src.get_ticks("X", T_UTC)


def test_utc_aware_one_second_before_allowed() -> None:
    src = _InjectableSource()
    src._ticks = pd.DataFrame({"ts": [T_UTC - timedelta(seconds=1)], "ltp": [2400.0]})
    assert len(src.get_ticks("X", T_UTC)) == 1


def test_utc_aware_one_second_after_blocked() -> None:
    src = _InjectableSource()
    src._ticks = pd.DataFrame({"ts": [T_UTC + timedelta(seconds=1)], "ltp": [2400.0]})
    with pytest.raises(DataLeakageError):
        src.get_ticks("X", T_UTC)


# D-2  Both tz-aware, different zones — cross-zone comparison must be correct

def test_column_ist_as_of_utc_same_instant_is_blocked() -> None:
    """
    Column in IST, as_of in UTC, same physical instant — must be blocked.
    10:30 UTC == 16:00 IST.  The tz-stripping approach would compare 10:30 vs 16:00
    naively and wrongly ALLOW this; proper UTC conversion blocks it.
    """
    src = _InjectableSource()
    src._ticks = pd.DataFrame({"ts": [T_IST], "ltp": [2400.0]})   # IST-aware, same instant
    with pytest.raises(DataLeakageError, match="PIT violation"):
        src.get_ticks("X", T_UTC)


def test_column_ist_as_of_utc_genuinely_past_allowed() -> None:
    """
    15:59 IST = 10:29 UTC — one minute before T_UTC. Must be allowed.
    Naive comparison would give 15:59 vs 10:30 → blocked (WRONG).
    UTC comparison gives 10:29 < 10:30 → allowed (CORRECT).
    """
    past_ist = T_IST - timedelta(minutes=1)   # 15:59 IST = 10:29 UTC
    src = _InjectableSource()
    src._ticks = pd.DataFrame({"ts": [past_ist], "ltp": [2400.0]})
    df = src.get_ticks("X", T_UTC)
    assert len(df) == 1


def test_column_ist_as_of_utc_genuinely_future_blocked() -> None:
    """
    16:01 IST = 10:31 UTC — one minute after T_UTC. Must be blocked.
    """
    future_ist = T_IST + timedelta(minutes=1)  # 16:01 IST = 10:31 UTC
    src = _InjectableSource()
    src._ticks = pd.DataFrame({"ts": [future_ist], "ltp": [2400.0]})
    with pytest.raises(DataLeakageError):
        src.get_ticks("X", T_UTC)


# D-3  Mixed tz (one aware, one naive) — must raise, not silently compare wrong

def test_tz_mismatch_aware_as_of_naive_column_raises() -> None:
    """
    as_of is tz-aware UTC, column is tz-naive.
    tz-stripping would allow 16:29 naive through against as_of=16:30 IST-stripped,
    even though 16:29 UTC = 21:59 IST = future.  We refuse to guess.
    """
    src = _InjectableSource()
    src._ticks = pd.DataFrame({"ts": [T_NAIVE], "ltp": [2400.0]})   # naive
    with pytest.raises(DataLeakageError, match="tz-mismatch"):
        src.get_ticks("X", T_UTC)   # UTC-aware


def test_tz_mismatch_naive_as_of_aware_column_raises() -> None:
    """as_of is naive, column is tz-aware UTC — also a mismatch."""
    src = _InjectableSource()
    src._ticks = pd.DataFrame({"ts": [T_UTC], "ltp": [2400.0]})   # UTC-aware
    with pytest.raises(DataLeakageError, match="tz-mismatch"):
        src.get_ticks("X", T_NAIVE)   # naive


# ── E. News: PIT column is first_knowable_at, NOT published_at ────────────────

def _make_news(published_at: datetime, first_knowable_at: datetime) -> pd.DataFrame:
    return pd.DataFrame({
        "news_id": ["n1"],
        "headline": ["Test story"],
        "body": ["Body text"],
        "source": ["reuters"],
        "published_at": [published_at],
        "first_knowable_at": [first_knowable_at],
    })


def test_news_published_before_t_first_knowable_before_t_allowed() -> None:
    """Story published and knowable before T: allowed."""
    src = _InjectableSource()
    src._news = _make_news(
        published_at=T_NAIVE - timedelta(hours=1),
        first_knowable_at=T_NAIVE - timedelta(minutes=30),
    )
    assert len(src.get_news("X", T_NAIVE)) == 1


def test_news_published_before_t_but_first_knowable_at_t_is_blocked() -> None:
    """
    Core news PIT test: story published 30 min before T, but not ingested /
    knowable until exactly T.  If PIT used published_at it would allow this
    through — that would be leakage.  Using first_knowable_at blocks it.
    """
    src = _InjectableSource()
    src._news = _make_news(
        published_at=T_NAIVE - timedelta(minutes=30),  # old — would PASS if PIT used this
        first_knowable_at=T_NAIVE,                      # exactly at T — must BLOCK
    )
    with pytest.raises(DataLeakageError, match="PIT violation"):
        src.get_news("X", T_NAIVE)


def test_news_does_not_use_published_at_for_pit() -> None:
    """
    Explicit proof that the wrong PIT column causes leakage.

    Construct a case where:
      - published_at < T  → would PASS if PIT enforced published_at (leakage)
      - first_knowable_at > T → must BLOCK (correct behavior)

    The adapter calls _enforce_pit with 'first_knowable_at', so this blocks.
    If someone accidentally changed the column to 'published_at', this test fails.
    """
    src = _InjectableSource()
    src._news = _make_news(
        published_at=T_NAIVE - timedelta(hours=2),      # clearly past
        first_knowable_at=T_NAIVE + timedelta(minutes=15),  # clearly future — must block
    )
    with pytest.raises(DataLeakageError, match="PIT violation"):
        src.get_news("X", T_NAIVE)


def test_news_exact_first_knowable_at_t_is_blocked() -> None:
    """Boundary: first_knowable_at == as_of is blocked (strict <)."""
    src = _InjectableSource()
    src._news = _make_news(
        published_at=T_NAIVE - timedelta(hours=1),
        first_knowable_at=T_NAIVE,   # exactly at T
    )
    with pytest.raises(DataLeakageError):
        src.get_news("X", T_NAIVE)


def test_news_one_microsecond_before_first_knowable_allowed() -> None:
    """T − 1 μs for first_knowable_at: allowed (strict < holds)."""
    src = _InjectableSource()
    src._news = _make_news(
        published_at=T_NAIVE - timedelta(hours=1),
        first_knowable_at=T_NAIVE - timedelta(microseconds=1),
    )
    assert len(src.get_news("X", T_NAIVE)) == 1


def test_news_published_after_t_but_first_knowable_before_t_allowed() -> None:
    """
    Edge case: first_knowable_at < T even though published_at > T.
    Unusual (first_knowable < published shouldn't happen) but the rule is unambiguous:
    only first_knowable_at is the PIT column.  If it is < T, the story is allowed.
    """
    src = _InjectableSource()
    src._news = _make_news(
        published_at=T_NAIVE + timedelta(hours=1),          # after T (unusual)
        first_knowable_at=T_NAIVE - timedelta(minutes=30),  # before T — allowed
    )
    assert len(src.get_news("X", T_NAIVE)) == 1


# ── F. Option chain ───────────────────────────────────────────────────────────

def test_option_chain_past_ts_allowed() -> None:
    src = _InjectableSource()
    src._options = pd.DataFrame({"ts": [T_NAIVE - timedelta(minutes=5)], "strike": [2400.0]})
    assert len(src.get_option_chain("X", T_NAIVE)) == 1


def test_option_chain_exact_ts_blocked() -> None:
    src = _InjectableSource()
    src._options = pd.DataFrame({"ts": [T_NAIVE], "strike": [2400.0]})
    with pytest.raises(DataLeakageError):
        src.get_option_chain("X", T_NAIVE)


def test_option_chain_future_ts_blocked() -> None:
    src = _InjectableSource()
    src._options = pd.DataFrame({"ts": [T_NAIVE + timedelta(minutes=1)], "strike": [2400.0]})
    with pytest.raises(DataLeakageError):
        src.get_option_chain("X", T_NAIVE)
