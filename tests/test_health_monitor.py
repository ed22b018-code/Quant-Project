"""Tests for DataHealthMonitor — Checkpoint 4.

Proof areas:
  1. Fresh feed → healthy (market open or closed)
  2. Stale feed during market hours → unhealthy → confidence_multiplier = 0.0
  3. Stale feed when market is closed → healthy → confidence_multiplier = 1.0
  4. Feed never seen during market hours → unhealthy
  5. Feed never seen when market is closed → healthy
  6. confidence_multiplier: 1.0 when healthy, 0.0 when unhealthy
  7. check_value: first observation (None prev) → True
  8. check_value: zero prev → True (no divide-by-zero)
  9. check_value: normal change → True
 10. check_value: extreme jump → False + WARNING logged
 11. as_of naive → ValueError (consistent with is_market_open)
 12. tz-mismatch between as_of and last-seen → ValueError
 13. Calendar integration: real NSETradingCalendar market-open/close windows
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

from src.data.calendar import IST, NSETradingCalendar
from src.data.health_monitor import DataHealthMonitor

# ── Fixtures ─────────────────────────────────────────────────────────────────

# Known trading day (Monday, not a holiday in 2024)
_TRADING_DAY = datetime(2024, 1, 22, tzinfo=IST)

# Market-open window: 09:15–15:30 IST
_MARKET_OPEN_DT = _TRADING_DAY.replace(hour=10, minute=0)   # 10:00 IST
_MARKET_CLOSED_DT = _TRADING_DAY.replace(hour=16, minute=0)  # 16:00 IST

_THRESHOLD_S = 30.0  # 30-second staleness threshold for tests


def _cal():
    return NSETradingCalendar()


def _monitor(threshold=_THRESHOLD_S) -> DataHealthMonitor:
    return DataHealthMonitor(stale_threshold_seconds=threshold)


# ── 1. Fresh feed ─────────────────────────────────────────────────────────────

def test_fresh_feed_healthy_during_market_hours():
    m = _monitor()
    last_seen = _MARKET_OPEN_DT - timedelta(seconds=10)  # 10 s old < 30 s threshold
    m.update_feed("ticks", last_seen)
    assert m.is_feed_healthy("ticks", _MARKET_OPEN_DT, _cal())


def test_fresh_feed_healthy_after_market_close():
    m = _monitor()
    last_seen = _MARKET_CLOSED_DT - timedelta(seconds=10)
    m.update_feed("ticks", last_seen)
    assert m.is_feed_healthy("ticks", _MARKET_CLOSED_DT, _cal())


# ── 2. Stale feed during market hours ────────────────────────────────────────

def test_stale_feed_unhealthy_during_market_hours():
    m = _monitor()
    last_seen = _MARKET_OPEN_DT - timedelta(seconds=120)  # 120 s old > 30 s threshold
    m.update_feed("ticks", last_seen)
    assert not m.is_feed_healthy("ticks", _MARKET_OPEN_DT, _cal())


def test_stale_confidence_zero_during_market_hours():
    m = _monitor()
    m.update_feed("ticks", _MARKET_OPEN_DT - timedelta(seconds=120))
    assert m.confidence_multiplier("ticks", _MARKET_OPEN_DT, _cal()) == pytest.approx(0.0)


# ── 3. Stale feed when market is closed ──────────────────────────────────────

def test_stale_feed_healthy_after_market_close():
    m = _monitor()
    last_seen = _MARKET_CLOSED_DT - timedelta(seconds=3600)  # 1 hour old
    m.update_feed("ticks", last_seen)
    # Market is closed at 16:00 → stale is OK
    assert m.is_feed_healthy("ticks", _MARKET_CLOSED_DT, _cal())


def test_stale_confidence_one_after_market_close():
    m = _monitor()
    m.update_feed("ticks", _MARKET_CLOSED_DT - timedelta(seconds=3600))
    assert m.confidence_multiplier("ticks", _MARKET_CLOSED_DT, _cal()) == pytest.approx(1.0)


# ── 4/5. Never-seen feed ──────────────────────────────────────────────────────

def test_never_seen_unhealthy_during_market_hours():
    m = _monitor()
    assert not m.is_feed_healthy("unknown_feed", _MARKET_OPEN_DT, _cal())


def test_never_seen_healthy_after_market_close():
    m = _monitor()
    assert m.is_feed_healthy("unknown_feed", _MARKET_CLOSED_DT, _cal())


# ── 6. Confidence multiplier ─────────────────────────────────────────────────

def test_confidence_multiplier_healthy():
    m = _monitor()
    m.update_feed("news", _MARKET_OPEN_DT - timedelta(seconds=5))
    result = m.confidence_multiplier("news", _MARKET_OPEN_DT, _cal())
    assert result == pytest.approx(1.0)


def test_confidence_multiplier_stale():
    m = _monitor()
    m.update_feed("news", _MARKET_OPEN_DT - timedelta(seconds=999))
    result = m.confidence_multiplier("news", _MARKET_OPEN_DT, _cal())
    assert result == pytest.approx(0.0)


# ── 7-10. check_value ─────────────────────────────────────────────────────────

def test_check_value_first_observation():
    m = _monitor()
    assert m.check_value("ltp", 100.0, prev_value=None) is True


def test_check_value_zero_prev():
    m = _monitor()
    assert m.check_value("ltp", 100.0, prev_value=0) is True


def test_check_value_normal_change():
    m = _monitor()
    # 5% change — well within 50% threshold
    assert m.check_value("ltp", 105.0, prev_value=100.0) is True


def test_check_value_extreme_jump_flagged(caplog):
    m = _monitor()
    with caplog.at_level(logging.WARNING):
        result = m.check_value("ltp", 300.0, prev_value=100.0)  # 200% jump
    assert result is False
    assert "ltp" in caplog.text or "jump" in caplog.text.lower()


def test_check_value_exactly_at_threshold():
    m = DataHealthMonitor(jump_threshold=0.5)
    # Exactly 50% change — should NOT flag (threshold is strict >)
    result = m.check_value("ltp", 150.0, prev_value=100.0)
    assert result is True


def test_check_value_just_over_threshold_flagged():
    m = DataHealthMonitor(jump_threshold=0.5)
    # 50.1% change — should flag
    result = m.check_value("ltp", 150.1, prev_value=100.0)
    assert result is False


# ── 11. Naive as_of raises ────────────────────────────────────────────────────

def test_naive_as_of_raises():
    m = _monitor()
    naive_dt = datetime(2024, 1, 22, 10, 0, 0)  # no tz
    with pytest.raises(ValueError, match="tz-aware"):
        m.is_feed_healthy("ticks", naive_dt, _cal())


def test_naive_as_of_confidence_raises():
    m = _monitor()
    naive_dt = datetime(2024, 1, 22, 10, 0, 0)
    with pytest.raises(ValueError):
        m.confidence_multiplier("ticks", naive_dt, _cal())


# ── 12. tz-mismatch in last-seen ─────────────────────────────────────────────

def test_tz_mismatch_last_seen_raises():
    m = _monitor()
    # Store a naive timestamp
    naive_ts = datetime(2024, 1, 22, 9, 55, 0)
    m.update_feed("ticks", naive_ts)
    # Query with tz-aware as_of → mismatch
    with pytest.raises(ValueError, match="tz-mismatch"):
        m.is_feed_healthy("ticks", _MARKET_OPEN_DT, _cal())


# ── 13. Calendar integration ─────────────────────────────────────────────────

def test_real_calendar_market_open_window():
    m = _monitor()
    # 09:30 IST is open; 09:00 IST is before open
    open_time = _TRADING_DAY.replace(hour=9, minute=30)
    before_open = _TRADING_DAY.replace(hour=9, minute=0)

    m.update_feed("ticks", open_time - timedelta(seconds=120))  # stale
    assert not m.is_feed_healthy("ticks", open_time, _cal())  # market open → unhealthy
    assert m.is_feed_healthy("ticks", before_open, _cal())  # market closed → healthy


def test_real_calendar_weekend_stale_is_ok():
    m = _monitor()
    # Saturday — market is always closed
    saturday = datetime(2024, 1, 20, 10, 0, 0, tzinfo=IST)  # 2024-01-20 is a Saturday
    m.update_feed("ticks", saturday - timedelta(hours=8))  # very stale
    assert m.is_feed_healthy("ticks", saturday, _cal())
