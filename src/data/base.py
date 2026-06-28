"""Abstract data-source interface — enforces point-in-time (PIT) access.

DESIGN PRINCIPLE (DESIGN.md §2 rule 1):
  At prediction time T, only data timestamped STRICTLY BEFORE T may be returned.
  This is enforced mechanically: every query method requires `as_of: datetime`, and
  the base class `_enforce_pit()` raises DataLeakageError if any returned row violates
  the constraint.  Subclasses MUST call `_enforce_pit()` before returning data.

The same interface is used by the backtester (fed historical data) and the live system
(fed real-time data) — the prediction code never knows which is running.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date as _date
from datetime import datetime as _datetime

import pandas as pd


class DataLeakageError(RuntimeError):
    """Raised when a data query would expose future-dated rows to the model."""


class DataNotAvailableError(RuntimeError):
    """Raised when a DataSource does not provide a given data type.

    Differs from DataLeakageError: this is a capability limitation, not a
    safety violation.  E.g., BhavCopySource raises this on get_ohlcv_minute
    because it only holds daily bars — the caller must acknowledge the
    limitation rather than silently receiving an empty DataFrame.
    """


class DataSource(ABC):
    """Base class for all data adapters (historical stub, bhavcopy, Upstox live, …).

    CONTRACT
    --------
    * Every public query method accepts `as_of: datetime`.
    * The returned DataFrame must contain ONLY rows with the relevant timestamp
      column strictly < as_of.
    * Subclasses MUST call `self._enforce_pit(df, ts_col, as_of)` before returning.
    * The live and historical adapters share this interface; prediction code must
      import only this module, never a concrete adapter.
    """

    # Subclasses declare these to self-document their limitations.
    data_source_name: str = "unknown"
    data_source_limitations: str = "unknown"
    data_granularity: str = "unknown"   # 'daily' | 'minute' | 'tick' | 'none'

    # ── OHLCV ────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_ohlcv_daily(
        self,
        symbol: str,
        as_of: _datetime,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """Daily OHLCV bars with date < as_of.date() (conservative: bar D available D+1).

        Columns: date (datetime.date), open, high, low, close (float64),
                 volume (int64), traded_value (float64), source (object).
        """

    @abstractmethod
    def get_ohlcv_minute(
        self,
        symbol: str,
        as_of: _datetime,
        lookback_bars: int = 390,
    ) -> pd.DataFrame:
        """1-min OHLCV bars with ts < as_of.

        Columns: ts (datetime64), open, high, low, close (float64),
                 volume (int64), source (object).
        """

    # ── Ticks ────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_ticks(
        self,
        symbol: str,
        as_of: _datetime,
        lookback_seconds: int = 300,
    ) -> pd.DataFrame:
        """Raw ticks with ts < as_of.

        Columns: ts (datetime64), ltp (float64), volume (int64),
                 bid, ask (float64), oi (int64), source (object).
        """

    # ── News ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_news(
        self,
        symbol: str,
        as_of: _datetime,
        lookback_hours: float = 24.0,
    ) -> pd.DataFrame:
        """News items with first_knowable_at < as_of.

        Uses first_knowable_at (not published_at) as the PIT column — this is the
        most conservative timestamp at which the story could realistically have been
        seen and acted upon (published_at + ingestion-delay buffer).

        Columns: news_id, headline, body, source (object),
                 published_at, first_knowable_at (datetime64).
        """

    # ── Option chain ─────────────────────────────────────────────────────────

    @abstractmethod
    def get_option_chain(
        self,
        symbol: str,
        as_of: _datetime,
    ) -> pd.DataFrame:
        """Most-recent option chain snapshot with ts < as_of.

        Columns: expiry (datetime.date), strike (float64), option_type (object),
                 ltp, oi, oi_change, volume (int64), iv (float64), ts (datetime64).
        """

    # ── PIT enforcement ──────────────────────────────────────────────────────

    def _enforce_pit(
        self,
        df: pd.DataFrame,
        ts_col: str,
        as_of: _datetime,
    ) -> pd.DataFrame:
        """Raise DataLeakageError if any row in ts_col is >= as_of.

        Handles three sub-cases for datetime columns:

        1. Both tz-aware → convert both to UTC before comparing.
           Correct even when column and as_of are in different zones (e.g., column
           in IST, as_of in UTC).

        2. Both tz-naive → compare directly (same implicit baseline assumed by caller).

        3. Mixed (one tz-aware, one naive) → raise immediately with a "tz-mismatch:"
           prefix.  Silent tz-stripping can let genuinely future rows through: e.g.,
           as_of = 16:30 IST-aware (stripped → 16:30 naive), column = 16:29 UTC-aware
           (stripped → 16:29 naive) → allowed, but 16:29 UTC = 21:59 IST = future.
           Refusing to guess is safer than guessing wrong.

        Date-only columns use a separate conservative rule: bar for date D is not
        available until midnight of D+1, so cutoff = as_of.date().
        """
        if df.empty:
            return df

        col = df[ts_col]
        non_null = col.dropna()
        if non_null.empty:
            return df

        first_val = non_null.iloc[0]
        is_date_only = isinstance(first_val, _date) and not isinstance(first_val, _datetime)

        if is_date_only:
            cutoff = as_of.date() if isinstance(as_of, _datetime) else as_of
            bad_mask = col.map(lambda v: v >= cutoff if isinstance(v, _date) else False)
        else:
            col_dt = pd.to_datetime(col)
            as_of_ts = pd.Timestamp(as_of)
            col_has_tz = col_dt.dt.tz is not None
            as_of_has_tz = as_of_ts.tzinfo is not None

            if col_has_tz and as_of_has_tz:
                col_cmp = col_dt.dt.tz_convert("UTC")
                as_of_cmp = as_of_ts.tz_convert("UTC")
            elif not col_has_tz and not as_of_has_tz:
                col_cmp = col_dt
                as_of_cmp = as_of_ts
            else:
                raise DataLeakageError(
                    f"tz-mismatch in PIT check for column '{ts_col}': "
                    f"column is {'tz-aware' if col_has_tz else 'tz-naive'}, "
                    f"as_of is {'tz-aware' if as_of_has_tz else 'tz-naive'}. "
                    f"Normalize all timestamps to a common timezone (UTC) before querying."
                )

            bad_mask = col_cmp >= as_of_cmp

        n_bad = int(bad_mask.sum())
        if n_bad > 0:
            raise DataLeakageError(
                f"PIT violation in column '{ts_col}': {n_bad} future-dated row(s) "
                f"(as_of={as_of!r}).  Check the data adapter."
            )
        return df
