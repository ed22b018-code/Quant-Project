"""Tests for OptionChainSource — Checkpoint 4.

Proof areas:
  1. Cache hit / cache miss / TTL expiry
  2. Exponential backoff on repeated HTTP failure
  3. Last-good fallback returned on failure; ts preserved (staleness surfaced)
  4. PIT enforcement: chain with future ts raises DataLeakageError
  5. OPTION_CHAIN_COLS schema correct on both real and empty response
  6. DataNotAvailableError on all non-option-chain methods
  7. No SDK import anywhere in module source
  8. _mock_fetch bypasses HTTP; _fetch_ts_override controls ts column
  9. Metadata fields present and correct
 10. Backoff delays: 1 s / 2 s / stops after max_retries
"""
from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from src.data.base import DataLeakageError, DataNotAvailableError
from src.data.historical import OPTION_CHAIN_COLS
from src.data.option_chain import OptionChainSource

# ── Helpers ──────────────────────────────────────────────────────────────────

_PAST = datetime(2024, 1, 22, 10, 0, 0, tzinfo=timezone.utc)
_AS_OF = datetime(2024, 1, 22, 10, 30, 0, tzinfo=timezone.utc)  # 30 min after fetch

_MOCK_RESPONSE = {
    "status": "success",
    "data": [
        {
            "expiry": "2024-01-25",
            "strike_price": 20000.0,
            "call_options": {
                "market_data": {
                    "ltp": 152.50,
                    "oi": 1250000,
                    "prev_oi": 1100000,
                    "volume": 35000,
                    "iv": 12.5,
                }
            },
            "put_options": {
                "market_data": {
                    "ltp": 82.30,
                    "oi": 980000,
                    "prev_oi": 1050000,
                    "volume": 28000,
                    "iv": 13.2,
                }
            },
        }
    ],
}


def _make_source(*, fetch_ts=None, mock_response=_MOCK_RESPONSE, ttl=60, max_retries=3):
    """Convenience: build OptionChainSource with a mock fetch and controlled ts."""
    fetch_ts = fetch_ts or _PAST
    return OptionChainSource(
        cache_ttl_seconds=ttl,
        max_retries=max_retries,
        _mock_fetch=lambda sym, exp: mock_response,
        _fetch_ts_override=fetch_ts,
    )


# ── 1. Schema ─────────────────────────────────────────────────────────────────

def test_option_chain_cols_schema():
    src = _make_source()
    df = src.get_option_chain("TCS", _AS_OF)
    assert list(df.columns) == list(OPTION_CHAIN_COLS.keys())


def test_option_chain_returns_ce_and_pe_rows():
    src = _make_source()
    df = src.get_option_chain("TCS", _AS_OF)
    assert set(df["option_type"].unique()) == {"CE", "PE"}


def test_option_chain_numeric_types():
    src = _make_source()
    df = src.get_option_chain("TCS", _AS_OF)
    assert df["ltp"].dtype == "float64"
    assert df["oi"].dtype in ("int64", "int32")
    assert df["iv"].dtype == "float64"


def test_empty_response_returns_correct_schema():
    src = OptionChainSource(
        cache_ttl_seconds=60,
        _mock_fetch=lambda s, e: {"status": "success", "data": []},
        _fetch_ts_override=_PAST,
    )
    df = src.get_option_chain("TCS", _AS_OF)
    assert list(df.columns) == list(OPTION_CHAIN_COLS.keys())
    assert df.empty


# ── 2. PIT enforcement ────────────────────────────────────────────────────────

def test_pit_passes_when_ts_before_as_of():
    src = _make_source(fetch_ts=_PAST)
    df = src.get_option_chain("TCS", _AS_OF)
    assert not df.empty


def test_pit_blocks_when_ts_after_as_of():
    future_ts = datetime(2030, 1, 1, tzinfo=timezone.utc)
    src = _make_source(fetch_ts=future_ts)
    with pytest.raises(DataLeakageError):
        src.get_option_chain("TCS", _AS_OF)


# ── 3. Cache hit / miss ───────────────────────────────────────────────────────

def test_cache_hit_avoids_second_fetch():
    call_count = 0

    def counting_fetch(sym, exp):
        nonlocal call_count
        call_count += 1
        return _MOCK_RESPONSE

    src = OptionChainSource(
        cache_ttl_seconds=60,
        _mock_fetch=counting_fetch,
        _fetch_ts_override=_PAST,
    )
    src.get_option_chain("TCS", _AS_OF)
    src.get_option_chain("TCS", _AS_OF)
    assert call_count == 1, "second call should hit cache, not re-fetch"


def test_cache_miss_after_ttl_expiry():
    call_count = 0

    def counting_fetch(sym, exp):
        nonlocal call_count
        call_count += 1
        return _MOCK_RESPONSE

    src = OptionChainSource(
        cache_ttl_seconds=0,  # 0-second TTL → always stale
        _mock_fetch=counting_fetch,
        _fetch_ts_override=_PAST,
    )
    src.get_option_chain("TCS", _AS_OF)
    src.get_option_chain("TCS", _AS_OF)
    assert call_count == 2, "TTL expired — should re-fetch on second call"


# ── 4. Last-good fallback ─────────────────────────────────────────────────────

def test_fallback_returned_on_fetch_failure(caplog):
    attempts = [0]

    def fail_after_one(sym, exp):
        attempts[0] += 1
        if attempts[0] > 1:
            raise RuntimeError("network down")
        return _MOCK_RESPONSE

    # TTL=0 so every call re-fetches; first succeeds, second fails
    src = OptionChainSource(
        cache_ttl_seconds=0,
        max_retries=1,  # fail fast
        _mock_fetch=fail_after_one,
        _fetch_ts_override=_PAST,
    )
    df1 = src.get_option_chain("TCS", _AS_OF)
    assert not df1.empty

    with caplog.at_level("WARNING"):
        df2 = src.get_option_chain("TCS", _AS_OF)

    assert not df2.empty, "fallback should return the last good chain"
    assert "last-good" in caplog.text.lower() or "fallback" in caplog.text.lower()


def test_fallback_ts_preserved():
    """The fallback chain's ts column records the original fetch time — staleness surfaced."""
    attempts = [0]
    original_ts = datetime(2024, 1, 22, 9, 0, 0, tzinfo=timezone.utc)

    def fail_after_one(sym, exp):
        attempts[0] += 1
        if attempts[0] > 1:
            raise RuntimeError("fail")
        return _MOCK_RESPONSE

    src = OptionChainSource(
        cache_ttl_seconds=0,
        max_retries=1,
        _mock_fetch=fail_after_one,
        _fetch_ts_override=original_ts,
    )

    src.get_option_chain("TCS", _AS_OF)
    df_fallback = src.get_option_chain("TCS", _AS_OF)

    # ts should be the original fetch time, not now()
    ts_max = pd.to_datetime(df_fallback["ts"].max())
    assert ts_max == pd.Timestamp(original_ts), (
        f"Fallback ts should be {original_ts}, got {ts_max}"
    )


def test_no_fallback_raises_data_not_available():
    def always_fail(sym, exp):
        raise RuntimeError("always fails")

    src = OptionChainSource(
        max_retries=1,
        _mock_fetch=always_fail,
    )
    with pytest.raises(DataNotAvailableError):
        src.get_option_chain("TCS", _AS_OF)


# ── 5. Exponential backoff ────────────────────────────────────────────────────

def test_backoff_sleeps_1s_2s_on_http_429():
    """HTTP 429 triggers backoff sleep of 1 s then 2 s."""
    import urllib.error

    def http_429(sym, exp):
        raise urllib.error.HTTPError(url=None, code=429, msg="rate limited", hdrs=None, fp=None)

    sleep_calls: list[float] = []

    src = OptionChainSource(
        max_retries=3,
        _mock_fetch=http_429,
        _fetch_ts_override=_PAST,
    )

    with patch("src.data.option_chain.time.sleep", side_effect=lambda s: sleep_calls.append(s)):
        result = src._fetch_with_backoff("TCS")

    assert result is None
    assert len(sleep_calls) == 3, f"Expected 3 sleeps, got {sleep_calls}"
    assert sleep_calls[0] == pytest.approx(1.0)
    assert sleep_calls[1] == pytest.approx(2.0)
    assert sleep_calls[2] == pytest.approx(4.0)


def test_backoff_stops_at_max_retries():
    attempts = [0]

    def always_fail(sym, exp):
        attempts[0] += 1
        raise RuntimeError("error")

    src = OptionChainSource(
        max_retries=2,
        _mock_fetch=always_fail,
    )

    with patch("src.data.option_chain.time.sleep"):
        result = src._fetch_with_backoff("TCS")

    assert result is None
    assert attempts[0] == 2


# ── 6. DataNotAvailableError on wrong methods ─────────────────────────────────

def test_get_ohlcv_daily_raises():
    src = _make_source()
    with pytest.raises(DataNotAvailableError):
        src.get_ohlcv_daily("TCS", _AS_OF)


def test_get_ohlcv_minute_raises():
    src = _make_source()
    with pytest.raises(DataNotAvailableError):
        src.get_ohlcv_minute("TCS", _AS_OF)


def test_get_ticks_raises():
    src = _make_source()
    with pytest.raises(DataNotAvailableError):
        src.get_ticks("TCS", _AS_OF)


def test_get_news_raises():
    src = _make_source()
    with pytest.raises(DataNotAvailableError):
        src.get_news("TCS", _AS_OF)


# ── 7. No SDK import ─────────────────────────────────────────────────────────

def test_no_upstox_sdk_import_in_module():
    import inspect
    import src.data.option_chain as mod

    source = inspect.getsource(mod)
    assert "upstox_client" not in source, "upstox_client SDK must not be imported here"
    assert "import upstox" not in source


# ── 8. Metadata ───────────────────────────────────────────────────────────────

def test_metadata_fields_correct():
    src = _make_source()
    assert src.data_source_name == "upstox_option_chain"
    assert src.data_granularity == "tick"
    assert "live option chain" in src.data_source_limitations
    assert "last-good" in src.data_source_limitations or "fallback" in src.data_source_limitations


def test_is_datasource_subclass():
    from src.data.base import DataSource
    assert isinstance(_make_source(), DataSource)
