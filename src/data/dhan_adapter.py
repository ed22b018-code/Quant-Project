"""Dhan Data Adapter — DataSource implementation for Dhan API.

SDK QUARANTINE: This is the ONLY file in the project permitted to import dhanhq.
The import is deferred to connect() so this module is usable (and fully testable)
without the SDK installed.  All other modules import DataSource from src.data.base.

Architecture
------------
* Historical methods (get_ohlcv_daily, get_ohlcv_minute, get_ohlcv_intraday,
  DhanFOMetricsSource.get_fo_metrics) query DuckDB tables populated by
  DhanBulkDownloader — no live network call for historical data.
* Live methods (get_ticks, get_option_chain) call the Dhan WebSocket / REST API.
* get_news raises DataNotAvailableError — Dhan does not provide news.

Corporate-action adjustment status
-----------------------------------
Dhan's adjustment policy is unknown until the key arrives and the API docs are
checked.  The env var DHAN_ADJUSTMENT_STATUS controls behaviour:
  PRE_ADJUSTED  — Dhan adjusts server-side; data returned as-is.
  RAW           — unadjusted; our compute_adjustment_factors() layer applies.
  (unset)       — UNKNOWN: WARNING logged on every historical query; data returned
                  as-is.  Set the env var immediately after confirming.

TODO: Set DHAN_ADJUSTMENT_STATUS in .env after confirming Dhan adjustment policy.

CPU-only — no GPU library imports (torch, cupy, etc.).
"""
from __future__ import annotations

import collections
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import date as _date
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from src.data.base import DataLeakageError, DataNotAvailableError, DataSource
from src.data.historical import (
    OHLCV_DAILY_COLS,
    OHLCV_MINUTE_COLS,
    OPTION_CHAIN_COLS,
    TICK_COLS,
    _empty,
)
from src.db.duckdb_client import DuckDBClient

_log = logging.getLogger(__name__)

_DHAN_OPTION_CHAIN_URL = "https://api.dhan.co/v2/optionchain"
_BACKOFF_BASE: float = 1.0
_RECONNECT_BASE: float = 1.0

# ── F&O daily metrics column schema ──────────────────────────────────────────
# Used by DhanFOMetricsSource.get_fo_metrics().  Phase 2 feature engineering
# reads these columns directly to compute pcr_oi, pcr_volume, max_pain_distance.
FO_DAILY_METRICS_COLS: dict[str, str] = {
    "date":            "object",   # datetime.date
    "symbol":          "object",
    "pcr_oi":          "float64",
    "pcr_volume":      "float64",
    "total_ce_oi":     "int64",
    "total_pe_oi":     "int64",
    "net_oi_change":   "int64",
    "max_pain_strike": "float64",
    "futures_settle":  "float64",
    "source":          "object",
}

_ADJUSTMENT_STATUS_KNOWN = {"PRE_ADJUSTED", "RAW"}


# ─────────────────────────────────────────────────────────────────────────────
# DhanDataSource
# ─────────────────────────────────────────────────────────────────────────────

class DhanDataSource(DataSource):
    """Dhan Data Adapter — historical queries from DuckDB, live via Dhan API.

    SDK QUARANTINE: This is the only file that may import dhanhq.
    Deferred import inside connect() — module usable without SDK installed.

    Construction (production)
    -------------------------
    source = DhanDataSource(db_path="data/market.duckdb")
    source.connect()   # starts WebSocket for live ticks
    df = source.get_ohlcv_minute("RELIANCE", as_of=datetime.now(UTC))

    Construction (tests — no SDK, no network)
    -----------------------------------------
    source = DhanDataSource(db_path=tmp_db, _inject_ws=object(), _mock_fetch=my_mock)
    # populate DuckDB with test data, then call get_ohlcv_minute etc.
    """

    data_source_name = "dhan"
    data_source_limitations = (
        "Dhan Data API; historical queries from DuckDB (populated by DhanBulkDownloader); "
        "live ticks via Dhan WebSocket; "
        "get_ohlcv_minute returns 1-min bars; use get_ohlcv_intraday for other timeframes; "
        "traded_value not provided — column is NaN in get_ohlcv_daily; "
        "news not available — use NewsSource (RSS) for news; "
        "credentials from DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN env vars; "
        "adjustment status: see DHAN_ADJUSTMENT_STATUS env var"
    )
    data_granularity = "minute"

    # No BACKTEST_ONLY — this source serves both historical (from DuckDB) and live.
    BACKTEST_ONLY = False

    def __init__(
        self,
        db_path: str | Any,
        *,
        client_id: str | None = None,
        access_token: str | None = None,
        cache_ttl_seconds: int = 60,
        max_retries: int = 3,
        buffer_maxlen: int = 10_000,
        max_reconnect_attempts: int = 5,
        _mock_fetch: Callable | None = None,      # test hook: replaces live HTTP calls
        _inject_ws: Any = None,                    # test hook: bypasses connect()
        _fetch_ts_override: datetime | None = None, # test hook: controls ts column
    ) -> None:
        self._db_path = str(db_path)
        self._client_id = client_id or os.environ.get("DHAN_CLIENT_ID", "")
        self._access_token = access_token or os.environ.get("DHAN_ACCESS_TOKEN", "")
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._max_retries = max_retries
        self._buffer_maxlen = buffer_maxlen
        self._max_reconnect = max_reconnect_attempts
        self._mock_fetch = _mock_fetch
        self._fetch_ts_override = _fetch_ts_override

        # Live tick buffer (same pattern as UpstoxLiveSource)
        self._lock = threading.Lock()
        self._buffers: dict[str, collections.deque] = {}
        self._ws: Any = _inject_ws
        self._connected = _inject_ws is not None
        self._reconnect_attempts = 0

        # Option chain cache: symbol → (DataFrame, wall_clock_fetch_time)
        self._oc_cache: dict[str, tuple[pd.DataFrame, datetime]] = {}
        self._oc_last_good: dict[str, pd.DataFrame] = {}

    # ── Connection management (live WebSocket) ────────────────────────────────

    def connect(self) -> None:
        """Establish Dhan WebSocket connection.

        ONLY place in the project that imports dhanhq.
        Raises ImportError with an actionable message if SDK not installed.
        """
        if self._connected:
            return
        try:
            import dhanhq  # noqa: F401, PLC0415  ← sole SDK import site
        except ImportError as exc:
            raise ImportError(
                "dhanhq SDK is not installed.  Install it for live Dhan feeds:\n"
                "  pip install dhanhq\n"
                "Historical queries from DuckDB work without the SDK."
            ) from exc
        _log.info("DhanDataSource: SDK available, WebSocket started")
        self._connected = True
        self._reconnect_attempts = 0

    def disconnect(self) -> None:
        if self._ws is not None:
            try:
                self._ws.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self._connected = False
        _log.info("DhanDataSource: disconnected")

    def _on_close(self, reason: str = "") -> None:
        """WebSocket close callback with exponential backoff reconnect."""
        self._connected = False
        if self._reconnect_attempts >= self._max_reconnect:
            _log.error(
                "Dhan WebSocket closed (%r). Max reconnect attempts (%d) reached.",
                reason, self._max_reconnect,
            )
            return
        wait = _RECONNECT_BASE * (2 ** self._reconnect_attempts)
        self._reconnect_attempts += 1
        _log.warning(
            "Dhan WebSocket closed (%r). Reconnecting in %.0fs (attempt %d/%d).",
            reason, wait, self._reconnect_attempts, self._max_reconnect,
        )
        if self._ws is None:  # no injected WS → don't actually sleep in tests
            time.sleep(wait)
        self.connect()

    # ── Tick ingest callback (WebSocket thread) ───────────────────────────────

    def _on_tick(self, message: dict) -> None:
        """Append one tick to the in-memory buffer.  Thread-safe."""
        symbol = str(message.get("symbol", "")).strip()
        if not symbol:
            return
        row = {
            "ts":     message.get("ts"),
            "ltp":    message.get("ltp"),
            "volume": message.get("volume"),
            "bid":    message.get("bid"),
            "ask":    message.get("ask"),
            "oi":     message.get("oi"),
            "source": "dhan_live",
        }
        with self._lock:
            if symbol not in self._buffers:
                self._buffers[symbol] = collections.deque(maxlen=self._buffer_maxlen)
            self._buffers[symbol].append(row)

    # ── Historical: get_ohlcv_daily ───────────────────────────────────────────

    def get_ohlcv_daily(
        self,
        symbol: str,
        as_of: datetime,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """Daily bars from DuckDB ohlcv_intraday (timeframe='D'), PIT-correct.

        traded_value is NaN — Dhan daily API does not return value-traded.
        """
        self._warn_if_adjustment_unknown()
        return self._query_intraday(symbol, "D", as_of, lookback_bars=lookback_days, daily=True)

    # ── Historical: get_ohlcv_minute (interface method — always 1-min bars) ───

    def get_ohlcv_minute(
        self,
        symbol: str,
        as_of: datetime,
        lookback_bars: int = 390,
    ) -> pd.DataFrame:
        """1-minute bars from DuckDB.  For other timeframes use get_ohlcv_intraday."""
        self._warn_if_adjustment_unknown()
        return self._query_intraday(symbol, "1", as_of, lookback_bars=lookback_bars)

    # ── Historical: get_ohlcv_intraday (Dhan extension — any timeframe) ───────

    def get_ohlcv_intraday(
        self,
        symbol: str,
        as_of: datetime,
        lookback_bars: int = 390,
        timeframe: str = "1",
    ) -> pd.DataFrame:
        """OHLCV bars at the given timeframe ('1','5','15','25','60') from DuckDB."""
        valid = {"1", "5", "15", "25", "60", "D"}
        if timeframe not in valid:
            raise ValueError(f"timeframe must be one of {valid}, got {timeframe!r}")
        self._warn_if_adjustment_unknown()
        return self._query_intraday(symbol, timeframe, as_of, lookback_bars=lookback_bars)

    # ── Live: get_ticks ───────────────────────────────────────────────────────

    def get_ticks(
        self,
        symbol: str,
        as_of: datetime,
        lookback_seconds: int = 300,
    ) -> pd.DataFrame:
        """Return buffered live ticks for `symbol` with ts < as_of."""
        with self._lock:
            snapshot = list(self._buffers.get(symbol, []))

        if not snapshot:
            return self._enforce_pit(_empty(TICK_COLS), "ts", as_of)

        df = pd.DataFrame(snapshot)

        if "ts" in df.columns:
            try:
                ts_series = pd.to_datetime(df["ts"])
                cutoff = pd.Timestamp(as_of) - pd.Timedelta(seconds=lookback_seconds)
                df = df[ts_series >= cutoff].copy()
            except Exception:  # noqa: BLE001
                pass

        if df.empty:
            return self._enforce_pit(_empty(TICK_COLS), "ts", as_of)

        df["ts"]     = pd.to_datetime(df["ts"])
        df["ltp"]    = pd.to_numeric(df["ltp"],    errors="coerce").astype("float64")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
        df["bid"]    = pd.to_numeric(df["bid"],    errors="coerce").astype("float64")
        df["ask"]    = pd.to_numeric(df["ask"],    errors="coerce").astype("float64")
        df["oi"]     = pd.to_numeric(df["oi"],     errors="coerce").fillna(0).astype("int64")
        df["source"] = "dhan_live"

        for col in TICK_COLS:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[list(TICK_COLS.keys())]
        return self._enforce_pit(df, "ts", as_of)

    # ── Live: get_option_chain ────────────────────────────────────────────────

    def get_option_chain(self, symbol: str, as_of: datetime) -> pd.DataFrame:
        """Live Dhan option chain with TTL cache and last-good fallback."""
        if symbol in self._oc_cache:
            cached_df, fetch_wall_time = self._oc_cache[symbol]
            if (datetime.now() - fetch_wall_time) < self._cache_ttl:
                _log.debug("DhanDataSource: option chain cache hit for %r", symbol)
                return self._enforce_pit(cached_df, "ts", as_of)

        df = self._oc_fetch_with_backoff(symbol)

        if df is not None:
            self._oc_cache[symbol] = (df, datetime.now())
            self._oc_last_good[symbol] = df
            return self._enforce_pit(df, "ts", as_of)

        if symbol in self._oc_last_good:
            fallback = self._oc_last_good[symbol]
            if not fallback.empty and "ts" in fallback.columns:
                try:
                    age_s = (
                        datetime.now(timezone.utc)
                        - pd.to_datetime(fallback["ts"].max()).tz_convert("UTC")
                    ).total_seconds()
                except Exception:
                    age_s = float("nan")
                _log.warning(
                    "DhanDataSource: option chain fetch failed for %r; "
                    "using last-good (%.0f s old).",
                    symbol, age_s,
                )
            return self._enforce_pit(fallback, "ts", as_of)

        raise DataNotAvailableError(
            f"DhanDataSource: option chain fetch failed for {symbol!r} and no "
            "fallback available.  Check DHAN_ACCESS_TOKEN."
        )

    # ── Not available ─────────────────────────────────────────────────────────

    def get_news(
        self, symbol: str, as_of: datetime, lookback_hours: float = 24.0,
    ) -> pd.DataFrame:
        raise DataNotAvailableError(
            f"DhanDataSource does not provide news for {symbol!r}.  "
            "Use NewsSource (RSS ingest adapter) for news."
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _warn_if_adjustment_unknown(self) -> None:
        status = os.environ.get("DHAN_ADJUSTMENT_STATUS", "").strip().upper()
        if status not in _ADJUSTMENT_STATUS_KNOWN:
            _log.warning(
                "DHAN_ADJUSTMENT_STATUS not set (current: %r).  "
                "Set to 'PRE_ADJUSTED' or 'RAW' in .env after confirming Dhan's "
                "corporate-action policy.  Data returned as-is until then.",
                status or "(empty)",
            )

    def _query_intraday(
        self,
        symbol: str,
        timeframe: str,
        as_of: datetime,
        lookback_bars: int,
        daily: bool = False,
    ) -> pd.DataFrame:
        """Query ohlcv_intraday table from DuckDB, apply PIT, return DataFrame."""
        as_of_ts = pd.Timestamp(as_of)
        if as_of_ts.tzinfo is not None:
            as_of_utc = as_of_ts.tz_convert("UTC").to_pydatetime()
        else:
            as_of_utc = as_of_ts.to_pydatetime()

        try:
            with DuckDBClient(self._db_path) as db:
                rows = db.query_df(
                    """
                    SELECT ts, open, high, low, close, volume, source
                    FROM ohlcv_intraday
                    WHERE symbol = ? AND timeframe = ? AND ts < ?
                    ORDER BY ts DESC
                    LIMIT ?
                    """,
                    (symbol, timeframe, as_of_utc, lookback_bars),
                )
        except Exception as exc:  # noqa: BLE001
            _log.error("DhanDataSource: DuckDB query failed: %s", exc)
            rows = _empty(OHLCV_MINUTE_COLS)

        if rows.empty:
            if daily:
                return self._enforce_pit(_empty(OHLCV_DAILY_COLS), "date", as_of)
            return self._enforce_pit(_empty(OHLCV_MINUTE_COLS), "ts", as_of)

        if daily:
            rows["ts"] = pd.to_datetime(rows["ts"], utc=True)
            rows["date"] = rows["ts"].dt.tz_convert("Asia/Kolkata").dt.date
            rows["traded_value"] = float("nan")
            rows["volume"] = rows["volume"].astype("int64")
            rows = rows.sort_values("date", ascending=True)
            result = rows[list(OHLCV_DAILY_COLS.keys())].copy()
            result["date"] = result["date"].apply(
                lambda v: v if isinstance(v, _date) else v
            )
            return self._enforce_pit(result, "date", as_of)

        rows["ts"] = pd.to_datetime(rows["ts"], utc=True)
        rows["volume"] = rows["volume"].astype("int64")
        rows = rows.sort_values("ts", ascending=True)
        result = rows[list(OHLCV_MINUTE_COLS.keys())].copy()
        return self._enforce_pit(result, "ts", as_of)

    def _get_fetch_ts(self) -> datetime:
        return (
            self._fetch_ts_override
            if self._fetch_ts_override is not None
            else datetime.now(timezone.utc)
        )

    def _oc_do_fetch(self, symbol: str) -> dict:
        if self._mock_fetch is not None:
            return self._mock_fetch(symbol, None)
        req = urllib.request.Request(
            f"{_DHAN_OPTION_CHAIN_URL}?underlying={symbol}",
            headers={
                "access-token": self._access_token,
                "client-id":    self._client_id,
                "Accept":       "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _oc_fetch_with_backoff(self, symbol: str) -> pd.DataFrame | None:
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                fetch_ts = self._get_fetch_ts()
                raw = self._oc_do_fetch(symbol)
                return self._oc_parse_response(raw, fetch_ts)
            except urllib.error.HTTPError as exc:
                last_exc = exc
                wait = _BACKOFF_BASE * (2 ** attempt)
                if exc.code == 429:
                    _log.warning(
                        "DhanDataSource: option chain rate-limited (429) for %r; "
                        "waiting %.0fs (attempt %d/%d).",
                        symbol, wait, attempt + 1, self._max_retries,
                    )
                else:
                    _log.warning(
                        "DhanDataSource: option chain HTTP %d for %r (attempt %d/%d).",
                        exc.code, symbol, attempt + 1, self._max_retries,
                    )
                if attempt < self._max_retries - 1:
                    time.sleep(wait)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                _log.warning(
                    "DhanDataSource: option chain fetch error for %r: %s (attempt %d/%d).",
                    symbol, exc, attempt + 1, self._max_retries,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(_BACKOFF_BASE * (2 ** attempt))
        _log.error(
            "DhanDataSource: all %d option chain fetch attempts failed for %r: %s",
            self._max_retries, symbol, last_exc,
        )
        return None

    def _oc_parse_response(self, raw: dict, fetch_ts: datetime) -> pd.DataFrame:
        """Parse Dhan option chain JSON into OPTION_CHAIN_COLS DataFrame.

        Expected response structure (may vary — validate on key arrival):
          {"status": "success", "data": [{"expiry": "YYYY-MM-DD", "strikePrice": 1000,
           "callOptions": {"marketData": {"ltp": ..., "oi": ..., ...}},
           "putOptions":  {"marketData": {"ltp": ..., "oi": ..., ...}}}]}
        """
        if raw.get("status") != "success":
            raise ValueError(f"Unexpected Dhan API status: {raw.get('status')!r}")
        data = raw.get("data", [])
        if not data:
            return _empty(OPTION_CHAIN_COLS)

        rows = []
        for item in data:
            expiry_str = item.get("expiry", "")
            try:
                expiry = _date.fromisoformat(str(expiry_str))
            except (ValueError, TypeError):
                _log.warning("DhanDataSource: cannot parse expiry %r, skipping.", expiry_str)
                continue
            strike = float(item.get("strikePrice", 0) or 0)
            for opt_type, key in (("CE", "callOptions"), ("PE", "putOptions")):
                md = (item.get(key) or {}).get("marketData") or {}
                oi_val = int(md.get("oi", 0) or 0)
                prev_oi = int(md.get("prevOi", 0) or 0)
                rows.append({
                    "expiry":      expiry,
                    "strike":      strike,
                    "option_type": opt_type,
                    "ltp":         float(md.get("ltp", 0) or 0),
                    "oi":          oi_val,
                    "oi_change":   oi_val - prev_oi,
                    "volume":      int(md.get("volume", 0) or 0),
                    "iv":          float(md.get("iv", 0) or 0),
                    "ts":          fetch_ts,
                })

        if not rows:
            return _empty(OPTION_CHAIN_COLS)
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df[list(OPTION_CHAIN_COLS.keys())]


# ─────────────────────────────────────────────────────────────────────────────
# DhanFOMetricsSource
# ─────────────────────────────────────────────────────────────────────────────

class DhanFOMetricsSource(DataSource):
    """F&O daily metrics source — reads fo_daily_metrics from DuckDB.

    BACKTEST_ONLY = True — this source holds historical data only.
    Live F&O snapshots are served by DhanDataSource.get_option_chain().

    get_fo_metrics() is an extension method (not in DataSource base).  Phase 2
    feature engineering imports DhanFOMetricsSource directly to call it.
    """

    BACKTEST_ONLY = True
    data_source_name = "dhan_fo_metrics"
    data_source_limitations = (
        "BACKTEST USE ONLY; queries fo_daily_metrics table populated by "
        "DhanBulkDownloader.compute_fo_metrics(); "
        "live F&O snapshots: use DhanDataSource.get_option_chain(); "
        "get_fo_metrics() is an extension method — not in the DataSource base interface"
    )
    data_granularity = "daily"

    def __init__(self, db_path: str | Any) -> None:
        self._db_path = str(db_path)

    # ── Extension method ──────────────────────────────────────────────────────

    def get_fo_metrics(
        self,
        symbol: str,
        as_of: datetime,
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """F&O daily aggregates for `symbol` with date < as_of.date().

        Columns: date, symbol, pcr_oi, pcr_volume, total_ce_oi, total_pe_oi,
                 net_oi_change, max_pain_strike, futures_settle, source.
        """
        as_of_date = as_of.date() if isinstance(as_of, datetime) else as_of
        cutoff = as_of_date - timedelta(days=lookback_days)
        try:
            with DuckDBClient(self._db_path) as db:
                df = db.query_df(
                    """
                    SELECT date, symbol, pcr_oi, pcr_volume, total_ce_oi, total_pe_oi,
                           net_oi_change, max_pain_strike, futures_settle, source
                    FROM fo_daily_metrics
                    WHERE symbol = ? AND date >= ? AND date < ?
                    ORDER BY date ASC
                    """,
                    (symbol, cutoff, as_of_date),
                )
        except Exception as exc:  # noqa: BLE001
            _log.error("DhanFOMetricsSource: DuckDB query failed: %s", exc)
            return self._enforce_pit(_empty(FO_DAILY_METRICS_COLS), "date", as_of)

        if df.empty:
            return self._enforce_pit(_empty(FO_DAILY_METRICS_COLS), "date", as_of)

        # Normalise types
        df["date"]          = df["date"].apply(
            lambda v: v if isinstance(v, _date) and not isinstance(v, datetime) else
            (v.date() if isinstance(v, datetime) else _date.fromisoformat(str(v)))
        )
        df["total_ce_oi"]   = pd.to_numeric(df["total_ce_oi"],   errors="coerce").fillna(0).astype("int64")
        df["total_pe_oi"]   = pd.to_numeric(df["total_pe_oi"],   errors="coerce").fillna(0).astype("int64")
        df["net_oi_change"] = pd.to_numeric(df["net_oi_change"], errors="coerce").fillna(0).astype("int64")

        return self._enforce_pit(df[list(FO_DAILY_METRICS_COLS.keys())], "date", as_of)

    # ── Standard interface — not available ────────────────────────────────────

    def get_ohlcv_daily(self, symbol, as_of, lookback_days=252):
        raise DataNotAvailableError(
            f"DhanFOMetricsSource does not provide OHLCV data for {symbol!r}.  "
            "Use DhanDataSource for OHLCV."
        )

    def get_ohlcv_minute(self, symbol, as_of, lookback_bars=390):
        raise DataNotAvailableError(
            f"DhanFOMetricsSource does not provide minute bars for {symbol!r}."
        )

    def get_ticks(self, symbol, as_of, lookback_seconds=300):
        raise DataNotAvailableError(
            f"DhanFOMetricsSource does not provide tick data for {symbol!r}."
        )

    def get_news(self, symbol, as_of, lookback_hours=24.0):
        raise DataNotAvailableError(
            f"DhanFOMetricsSource does not provide news for {symbol!r}."
        )

    def get_option_chain(self, symbol, as_of):
        raise DataNotAvailableError(
            f"DhanFOMetricsSource does not provide option chain for {symbol!r}.  "
            "Use DhanDataSource.get_option_chain() for live snapshots."
        )
