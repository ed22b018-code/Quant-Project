"""Stub historical data adapter — Phase 0 placeholder.

All methods return empty DataFrames with the correct column schema.
Phase 1 replaces the method bodies with real DuckDB queries against the
NSE bhavcopy tables.  The interface (method signatures, column names/types,
PIT enforcement) is frozen here and must not change.

Why a stub rather than raising NotImplementedError?
  Downstream code (feature pipeline, backtester) can import and call these methods
  without any data being present, which lets us write and test the full call-chain
  skeleton before Phase 1 data exists.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data.base import DataSource

# ── Column schemas ────────────────────────────────────────────────────────────
# Shared between the stub and the Phase 1 real adapter so column expectations
# are defined in one place.

OHLCV_DAILY_COLS: dict[str, str] = {
    "date":          "object",    # datetime.date; do NOT convert to datetime64 for daily
    "open":          "float64",
    "high":          "float64",
    "low":           "float64",
    "close":         "float64",
    "volume":        "int64",
    "traded_value":  "float64",
    "source":        "object",
}

OHLCV_MINUTE_COLS: dict[str, str] = {
    "ts":      "datetime64[ns]",
    "open":    "float64",
    "high":    "float64",
    "low":     "float64",
    "close":   "float64",
    "volume":  "int64",
    "source":  "object",
}

TICK_COLS: dict[str, str] = {
    "ts":      "datetime64[ns]",
    "ltp":     "float64",
    "volume":  "int64",
    "bid":     "float64",
    "ask":     "float64",
    "oi":      "int64",
    "source":  "object",
}

NEWS_COLS: dict[str, str] = {
    "news_id":           "object",
    "headline":          "object",
    "body":              "object",
    "source":            "object",
    "published_at":      "datetime64[ns]",
    "first_knowable_at": "datetime64[ns]",
    "symbols":           "object",   # list of tagged ticker strings; [] for untagged
}

OPTION_CHAIN_COLS: dict[str, str] = {
    "expiry":       "object",    # datetime.date
    "strike":       "float64",
    "option_type":  "object",    # CE / PE
    "ltp":          "float64",
    "oi":           "int64",
    "oi_change":    "int64",
    "volume":       "int64",
    "iv":           "float64",
    "ts":           "datetime64[ns]",
}


def _empty(schema: dict[str, str]) -> pd.DataFrame:
    """Return an empty DataFrame with the given column names (types not enforced on empty)."""
    return pd.DataFrame(columns=list(schema.keys()))


# ── Stub adapter ──────────────────────────────────────────────────────────────

class StubHistoricalSource(DataSource):
    """Phase 0 stub — returns empty DataFrames with correct schemas.

    Phase 1 will replace each method body with a DuckDB query; nothing outside
    this file needs to change.
    """

    data_source_name = "stub_historical"
    data_source_limitations = (
        "Phase 0 stub: returns no data.  "
        "Phase 1 wires NSE bhavcopy daily bars (data_granularity='daily').  "
        "Intraday (minute/tick) data requires a Kite or supplied dataset — "
        "no intraday-edge conclusions may be drawn until that source is wired."
    )
    data_granularity = "none"

    def get_ohlcv_daily(
        self,
        symbol: str,
        as_of: datetime,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        return self._enforce_pit(_empty(OHLCV_DAILY_COLS), "date", as_of)

    def get_ohlcv_minute(
        self,
        symbol: str,
        as_of: datetime,
        lookback_bars: int = 390,
    ) -> pd.DataFrame:
        return self._enforce_pit(_empty(OHLCV_MINUTE_COLS), "ts", as_of)

    def get_ticks(
        self,
        symbol: str,
        as_of: datetime,
        lookback_seconds: int = 300,
    ) -> pd.DataFrame:
        return self._enforce_pit(_empty(TICK_COLS), "ts", as_of)

    def get_news(
        self,
        symbol: str,
        as_of: datetime,
        lookback_hours: float = 24.0,
    ) -> pd.DataFrame:
        return self._enforce_pit(_empty(NEWS_COLS), "first_knowable_at", as_of)

    def get_option_chain(
        self,
        symbol: str,
        as_of: datetime,
    ) -> pd.DataFrame:
        return self._enforce_pit(_empty(OPTION_CHAIN_COLS), "ts", as_of)
