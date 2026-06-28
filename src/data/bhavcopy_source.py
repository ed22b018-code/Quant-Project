"""BhavCopySource — DataSource backed by NSE bhavcopy daily data in DuckDB.

This is the first real (non-stub) DataSource.  It provides daily OHLCV only.
All other data types raise DataNotAvailableError with an explicit message so
that callers cannot accidentally treat an empty DataFrame as "no data found"
when the truth is "this source cannot provide that data type at all".

Metadata (recorded in every experiment run):
  data_source_name       = "nse_bhavcopy"
  data_granularity       = "daily"
  data_source_limitations = (see below)

Key design decisions:
  - Prices are raw/unadjusted — do NOT use for backtest P&L without
    corporate-action adjustment (Phase 1 follow-up).
  - Survivorship is documentable: use eligibility.get_eligible_symbols()
    to reconstruct the per-date universe, never a static list.
  - PIT: every query filters date < as_of.date() BEFORE calling _enforce_pit,
    so the PIT guard is both a pre-filter AND a final safety net.
"""
from __future__ import annotations

from datetime import datetime as _datetime
from pathlib import Path

import pandas as pd

from src.data.base import DataNotAvailableError, DataSource
from src.db.duckdb_client import DuckDBClient
from src.db.schema import DUCKDB_SCHEMA_SQL


class BhavCopySource(DataSource):
    """DataSource backed by NSE daily bhavcopy data stored in DuckDB."""

    data_source_name = "nse_bhavcopy"
    data_source_limitations = (
        "daily bars only; raw unadjusted prices; "
        "survivorship-documentable via dated files; "
        "no intraday (minute/tick) data; "
        "corporate-action adjustment pending (Phase 1 follow-up)"
    )
    data_granularity = "daily"

    def __init__(self, db_path: Path) -> None:
        self._db_path = Path(db_path)

    # ── Daily OHLCV — the one supported data type ────────────────────────────

    def get_ohlcv_daily(
        self,
        symbol: str,
        as_of: _datetime,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """Daily OHLCV bars for `symbol` with date strictly < as_of.date().

        Conservative rule: bar for date D is not available until midnight of
        D+1, so we query date < as_of.date() and then call _enforce_pit() as
        a final safety net.

        Returns an empty DataFrame (correct schema) if no data is present.
        """
        cutoff: object
        if isinstance(as_of, _datetime):
            cutoff = as_of.date()
        else:
            cutoff = as_of  # already a date

        sql = """
            SELECT date, open, high, low, close, volume, traded_value, source
            FROM ohlcv_daily
            WHERE symbol = ?
              AND date < ?
            ORDER BY date DESC
            LIMIT ?
        """
        with DuckDBClient(self._db_path) as db:
            db.execute_schema(DUCKDB_SCHEMA_SQL)
            df = db.query_df(sql, (symbol, cutoff, lookback_days))

        # DuckDB may return DATE columns as pd.Timestamp (datetime64[ns]) depending on
        # version.  Normalise to datetime.date objects so _enforce_pit uses the
        # date-only branch regardless of as_of timezone.
        if not df.empty and "date" in df.columns:
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"]).dt.date

        return self._enforce_pit(df, "date", as_of)

    # ── All other data types — not available from daily bhavcopy ─────────────

    def get_ohlcv_minute(
        self,
        symbol: str,
        as_of: _datetime,
        lookback_bars: int = 390,
    ) -> pd.DataFrame:
        raise DataNotAvailableError(
            f"BhavCopySource (data_granularity='daily') cannot provide minute bars "
            f"for {symbol!r}.  "
            "Requires Kite or a supplied intraday dataset — not available from nse_bhavcopy; "
            "any backtest using BhavCopySource cannot validate intraday edge."
        )

    def get_ticks(
        self,
        symbol: str,
        as_of: _datetime,
        lookback_seconds: int = 300,
    ) -> pd.DataFrame:
        raise DataNotAvailableError(
            f"BhavCopySource does not provide tick data for {symbol!r}.  "
            f"Use the Upstox live adapter (src/data/live_upstox.py) for ticks."
        )

    def get_news(
        self,
        symbol: str,
        as_of: _datetime,
        lookback_hours: float = 24.0,
    ) -> pd.DataFrame:
        raise DataNotAvailableError(
            f"BhavCopySource does not provide news for {symbol!r}.  "
            f"Use the RSS news ingest (src/data/news_ingest.py) for news."
        )

    def get_option_chain(
        self,
        symbol: str,
        as_of: _datetime,
    ) -> pd.DataFrame:
        raise DataNotAvailableError(
            f"BhavCopySource does not provide option chain data for {symbol!r}.  "
            f"Use the Upstox option chain fetcher for live option chains."
        )
