"""
Tests for NSETradingCalendar.

Coverage:
  A. is_trading_day  — holiday, Saturday, Sunday, normal weekday.
  B. is_market_open  — before open, at open, during, at close (exclusive),
                       holiday, weekend, tz-naive raises, UTC datetime input.
  C. next/prev trading day — normal skip, skip weekend, skip holiday+weekend.
  D. trading_days_in_range — range with holiday and weekend.
  E. classify_missing_file — holiday, weekend, trading day.
  F. Fallback — missing JSON logs warning and falls back to embedded; same for
     malformed JSON; embedded list contains known holidays.
  G. refresh_nse_holidays — returns False and logs warning on network failure
     without modifying the cache.
  H. get_calendar — module-level factory returns a usable instance.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest

from src.data.calendar import (
    NSETradingCalendar,
    IST,
    get_calendar,
    refresh_nse_holidays,
)

# ---------------------------------------------------------------------------
# Reference dates — all verified by day-of-week arithmetic.
# Jan 1 2024 = Monday; therefore:
#   Jan 22 = Mon, Jan 23 = Tue, Jan 24 = Wed, Jan 25 = Thu,
#   Jan 26 = Fri (Republic Day), Jan 27 = Sat, Jan 28 = Sun, Jan 29 = Mon.
# March 1 2024 = Friday; therefore Mar 18 = Mon, Mar 19 = Tue (Holi), Mar 20 = Wed.
# ---------------------------------------------------------------------------

HOLIDAYS_2024 = {
    date(2024, 1, 26),   # Republic Day (Friday)
    date(2024, 3, 19),   # Holi (Tuesday)
    date(2024, 3, 29),   # Good Friday
    date(2024, 8, 15),   # Independence Day (Thursday)
    date(2024, 11, 1),   # Diwali Laxmi Puja (Friday)
}


@pytest.fixture
def cal() -> NSETradingCalendar:
    """Calendar with a controlled holiday set for deterministic tests."""
    return NSETradingCalendar(holidays=HOLIDAYS_2024)


# ── A. is_trading_day ────────────────────────────────────────────────────────

def test_known_holiday_is_not_trading_day(cal: NSETradingCalendar) -> None:
    assert not cal.is_trading_day(date(2024, 1, 26))   # Republic Day


def test_saturday_is_not_trading_day(cal: NSETradingCalendar) -> None:
    assert not cal.is_trading_day(date(2024, 1, 27))   # Sat after Republic Day


def test_sunday_is_not_trading_day(cal: NSETradingCalendar) -> None:
    assert not cal.is_trading_day(date(2024, 1, 28))   # Sun


def test_normal_monday_is_trading_day(cal: NSETradingCalendar) -> None:
    assert cal.is_trading_day(date(2024, 1, 29))        # Mon after the holiday+weekend


def test_thursday_before_holiday_is_trading_day(cal: NSETradingCalendar) -> None:
    assert cal.is_trading_day(date(2024, 1, 25))        # Thu before Republic Day


# ── B. is_market_open ────────────────────────────────────────────────────────

_TRADING_MON = date(2024, 1, 29)   # Monday, not a holiday in HOLIDAYS_2024


def _ist(d: date, h: int, m: int, s: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, h, m, s, tzinfo=IST)


def test_before_open_not_market_open(cal: NSETradingCalendar) -> None:
    assert not cal.is_market_open(_ist(_TRADING_MON, 9, 14, 59))


def test_at_open_market_is_open(cal: NSETradingCalendar) -> None:
    assert cal.is_market_open(_ist(_TRADING_MON, 9, 15, 0))


def test_during_market_is_open(cal: NSETradingCalendar) -> None:
    assert cal.is_market_open(_ist(_TRADING_MON, 12, 0, 0))


def test_just_before_close_is_open(cal: NSETradingCalendar) -> None:
    assert cal.is_market_open(_ist(_TRADING_MON, 15, 29, 59))


def test_at_close_not_market_open(cal: NSETradingCalendar) -> None:
    """15:30:00 IST is the first second OUTSIDE market hours."""
    assert not cal.is_market_open(_ist(_TRADING_MON, 15, 30, 0))


def test_after_close_not_market_open(cal: NSETradingCalendar) -> None:
    assert not cal.is_market_open(_ist(_TRADING_MON, 16, 0, 0))


def test_market_open_on_holiday_is_false(cal: NSETradingCalendar) -> None:
    assert not cal.is_market_open(_ist(date(2024, 1, 26), 10, 0, 0))


def test_market_open_on_weekend_is_false(cal: NSETradingCalendar) -> None:
    assert not cal.is_market_open(_ist(date(2024, 1, 27), 10, 0, 0))


def test_naive_datetime_raises_value_error(cal: NSETradingCalendar) -> None:
    naive = datetime(2024, 1, 29, 10, 0, 0)
    with pytest.raises(ValueError, match="tz-aware"):
        cal.is_market_open(naive)


def test_utc_datetime_at_market_open(cal: NSETradingCalendar) -> None:
    # 09:15 IST = 03:45 UTC  (IST = UTC + 5h30m)
    utc_open = datetime(2024, 1, 29, 3, 45, 0, tzinfo=timezone.utc)
    assert cal.is_market_open(utc_open)


def test_utc_datetime_before_open(cal: NSETradingCalendar) -> None:
    # 03:44 UTC = 09:14 IST — one minute before open
    utc_before = datetime(2024, 1, 29, 3, 44, 0, tzinfo=timezone.utc)
    assert not cal.is_market_open(utc_before)


# ── C. next_trading_day / prev_trading_day ───────────────────────────────────

def test_next_trading_day_normal(cal: NSETradingCalendar) -> None:
    # Mon Jan 29 → next is Tue Jan 30
    assert cal.next_trading_day(date(2024, 1, 29)) == date(2024, 1, 30)


def test_next_trading_day_skips_holiday_and_weekend(cal: NSETradingCalendar) -> None:
    # Thu Jan 25 → skip Fri holiday (Jan 26) + Sat + Sun → Mon Jan 29
    assert cal.next_trading_day(date(2024, 1, 25)) == date(2024, 1, 29)


def test_next_trading_day_skips_single_holiday(cal: NSETradingCalendar) -> None:
    # Mon Mar 18 → skip Tue Holi (Mar 19) → Wed Mar 20
    assert cal.next_trading_day(date(2024, 3, 18)) == date(2024, 3, 20)


def test_prev_trading_day_normal(cal: NSETradingCalendar) -> None:
    # Mon Jan 29 → prev is Thu Jan 25 (skip Fri holiday + Sat + Sun going backwards)
    assert cal.prev_trading_day(date(2024, 1, 29)) == date(2024, 1, 25)


def test_prev_trading_day_skips_single_holiday(cal: NSETradingCalendar) -> None:
    # Wed Mar 20 → skip Tue Holi (Mar 19) → Mon Mar 18
    assert cal.prev_trading_day(date(2024, 3, 20)) == date(2024, 3, 18)


def test_prev_trading_day_normal_no_holiday(cal: NSETradingCalendar) -> None:
    # Tue Jan 30 → Mon Jan 29
    assert cal.prev_trading_day(date(2024, 1, 30)) == date(2024, 1, 29)


# ── D. trading_days_in_range ─────────────────────────────────────────────────

def test_trading_days_in_range_skips_holiday_and_weekend(cal: NSETradingCalendar) -> None:
    # Jan 22 (Mon) to Jan 29 (Mon): skip Jan 26 holiday, Jan 27 Sat, Jan 28 Sun
    days = cal.trading_days_in_range(date(2024, 1, 22), date(2024, 1, 29))
    expected = [
        date(2024, 1, 22),
        date(2024, 1, 23),
        date(2024, 1, 24),
        date(2024, 1, 25),
        date(2024, 1, 29),
    ]
    assert days == expected


def test_trading_days_in_range_single_trading_day(cal: NSETradingCalendar) -> None:
    days = cal.trading_days_in_range(date(2024, 1, 29), date(2024, 1, 29))
    assert days == [date(2024, 1, 29)]


def test_trading_days_in_range_single_holiday_empty(cal: NSETradingCalendar) -> None:
    days = cal.trading_days_in_range(date(2024, 1, 26), date(2024, 1, 26))
    assert days == []


def test_trading_days_in_range_reversed_empty(cal: NSETradingCalendar) -> None:
    # start > end → empty list
    assert cal.trading_days_in_range(date(2024, 1, 30), date(2024, 1, 29)) == []


# ── E. classify_missing_file ─────────────────────────────────────────────────

def test_classify_holiday_is_expected(cal: NSETradingCalendar) -> None:
    assert cal.classify_missing_file(date(2024, 1, 26)) == "expected"


def test_classify_saturday_is_expected(cal: NSETradingCalendar) -> None:
    assert cal.classify_missing_file(date(2024, 1, 27)) == "expected"


def test_classify_sunday_is_expected(cal: NSETradingCalendar) -> None:
    assert cal.classify_missing_file(date(2024, 1, 28)) == "expected"


def test_classify_trading_day_is_unexpected(cal: NSETradingCalendar) -> None:
    assert cal.classify_missing_file(date(2024, 1, 29)) == "unexpected"


# ── F. Fallback behaviour ─────────────────────────────────────────────────────

def test_missing_json_falls_back_to_embedded_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    missing = tmp_path / "does_not_exist.json"
    with caplog.at_level(logging.WARNING, logger="src.data.calendar"):
        cal = NSETradingCalendar(holidays_json=missing)

    assert any(r.levelno == logging.WARNING for r in caplog.records), (
        "Expected a WARNING when the holidays file is missing"
    )
    # Calendar must still function correctly with the embedded list
    assert not cal.is_trading_day(date(2024, 1, 27))   # Saturday — never a trading day


def test_malformed_json_falls_back_to_embedded_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ this is not valid json }", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="src.data.calendar"):
        cal = NSETradingCalendar(holidays_json=bad_json)

    assert any(r.levelno == logging.WARNING for r in caplog.records), (
        "Expected a WARNING when the holidays JSON is malformed"
    )
    assert not cal.is_trading_day(date(2024, 1, 27))


def test_embedded_list_contains_republic_day_2024(tmp_path: Path) -> None:
    """Embedded fallback must know 2024-01-26 is a holiday."""
    missing = tmp_path / "no_file.json"
    cal = NSETradingCalendar(holidays_json=missing)
    assert not cal.is_trading_day(date(2024, 1, 26))


def test_embedded_list_contains_independence_day_2024(tmp_path: Path) -> None:
    missing = tmp_path / "no_file.json"
    cal = NSETradingCalendar(holidays_json=missing)
    assert not cal.is_trading_day(date(2024, 8, 15))


def test_valid_custom_json_loaded_correctly(tmp_path: Path) -> None:
    """A correctly-formatted JSON override is used in preference to the embedded list."""
    custom_path = tmp_path / "custom.json"
    custom_path.write_text(
        json.dumps({"2024": ["2024-02-14"]}),  # Valentine's Day (not in embedded)
        encoding="utf-8",
    )
    cal = NSETradingCalendar(holidays_json=custom_path)
    assert not cal.is_trading_day(date(2024, 2, 14))   # custom holiday
    assert cal.is_trading_day(date(2024, 1, 29))        # Jan 29 NOT in custom list → trading


# ── G. refresh_nse_holidays — fails gracefully ───────────────────────────────

def test_refresh_returns_false_on_network_failure(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    save_path = tmp_path / "holidays.json"
    with patch("urllib.request.urlopen", side_effect=OSError("connection refused")):
        with caplog.at_level(logging.WARNING, logger="src.data.calendar"):
            result = refresh_nse_holidays(2024, save_path=save_path)

    assert result is False, "refresh_nse_holidays must return False on network failure"
    assert not save_path.exists(), "Cache must not be written on failure"
    assert any(r.levelno == logging.WARNING for r in caplog.records), (
        "Expected a WARNING on network failure"
    )


def test_refresh_does_not_raise_on_failure(tmp_path: Path) -> None:
    """Network failure must never propagate as an exception."""
    save_path = tmp_path / "holidays.json"
    with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
        try:
            refresh_nse_holidays(2025, save_path=save_path)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"refresh_nse_holidays raised unexpectedly: {exc}")


# ── H. get_calendar ───────────────────────────────────────────────────────────

def test_get_calendar_returns_instance() -> None:
    cal = get_calendar()
    assert isinstance(cal, NSETradingCalendar)


def test_get_calendar_is_functional() -> None:
    cal = get_calendar()
    # Sanity: weekends are never trading days regardless of holiday list
    assert not cal.is_trading_day(date(2024, 1, 27))   # Saturday
    assert not cal.is_trading_day(date(2024, 1, 28))   # Sunday
