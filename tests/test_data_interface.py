"""Tests for the DataSource contract and StubHistoricalSource."""
from __future__ import annotations

from datetime import datetime, timezone

from src.data.base import DataSource
from src.data.historical import (
    NEWS_COLS,
    OHLCV_DAILY_COLS,
    OHLCV_MINUTE_COLS,
    OPTION_CHAIN_COLS,
    TICK_COLS,
    StubHistoricalSource,
)

AS_OF = datetime(2024, 6, 15, 10, 30, tzinfo=timezone.utc)


def test_stub_is_datasource_subclass() -> None:
    assert issubclass(StubHistoricalSource, DataSource)


def test_stub_is_instantiable() -> None:
    src = StubHistoricalSource()
    assert src is not None


def test_stub_returns_empty_ohlcv_daily() -> None:
    df = StubHistoricalSource().get_ohlcv_daily("RELIANCE", AS_OF)
    assert df.empty


def test_stub_returns_empty_ohlcv_minute() -> None:
    df = StubHistoricalSource().get_ohlcv_minute("RELIANCE", AS_OF)
    assert df.empty


def test_stub_returns_empty_ticks() -> None:
    df = StubHistoricalSource().get_ticks("RELIANCE", AS_OF)
    assert df.empty


def test_stub_returns_empty_news() -> None:
    df = StubHistoricalSource().get_news("RELIANCE", AS_OF)
    assert df.empty


def test_stub_returns_empty_option_chain() -> None:
    df = StubHistoricalSource().get_option_chain("RELIANCE", AS_OF)
    assert df.empty


def test_stub_ohlcv_daily_correct_columns() -> None:
    df = StubHistoricalSource().get_ohlcv_daily("TCS", AS_OF)
    assert list(df.columns) == list(OHLCV_DAILY_COLS.keys())


def test_stub_ohlcv_minute_correct_columns() -> None:
    df = StubHistoricalSource().get_ohlcv_minute("TCS", AS_OF)
    assert list(df.columns) == list(OHLCV_MINUTE_COLS.keys())


def test_stub_ticks_correct_columns() -> None:
    df = StubHistoricalSource().get_ticks("TCS", AS_OF)
    assert list(df.columns) == list(TICK_COLS.keys())


def test_stub_news_correct_columns() -> None:
    df = StubHistoricalSource().get_news("TCS", AS_OF)
    assert list(df.columns) == list(NEWS_COLS.keys())


def test_stub_option_chain_correct_columns() -> None:
    df = StubHistoricalSource().get_option_chain("TCS", AS_OF)
    assert list(df.columns) == list(OPTION_CHAIN_COLS.keys())


def test_stub_metadata_fields_present() -> None:
    src = StubHistoricalSource()
    assert src.data_source_name
    assert src.data_source_limitations
    assert src.data_granularity == "none"


def test_stub_works_for_any_symbol() -> None:
    """DataSource must not raise for unknown symbols — it returns empty data."""
    src = StubHistoricalSource()
    assert src.get_ohlcv_daily("NONEXISTENT", AS_OF).empty
