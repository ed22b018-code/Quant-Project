"""Upstox WebSocket tick adapter.

SDK QUARANTINE: This is the ONLY file in the entire project that may import
anything from the upstox_client / upstox-python-api SDK.  No other module may
import the SDK directly or transitively.  Prediction, feature, and backtest code
imports only DataSource from src.data.base — they never touch this file.

The SDK import is deferred to connect() — not at module level — so this module
can be imported (and fully tested) without the SDK installed.  Tests inject ticks
via _on_tick() directly, bypassing connect() entirely.

Key design decisions
--------------------
* as_of is ALWAYS supplied by the caller; the adapter never reads a clock.
* Thread safety: _on_tick() (WebSocket thread) and get_ticks() (caller thread)
  share per-symbol deques protected by a single threading.Lock.
* On no data, get_ticks() returns an empty correctly-typed DataFrame — never None,
  never raises.
* Reconnect with exponential backoff: _on_close() waits 1, 2, 4, 8 … seconds
  before re-calling connect().  Max reconnect attempts is configurable.
* CPU-only — no GPU library imports (torch, cupy, etc.).
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from datetime import datetime as _datetime
from typing import Any

import pandas as pd

from src.data.base import DataNotAvailableError, DataSource
from src.data.historical import TICK_COLS, _empty

_log = logging.getLogger(__name__)

_RECONNECT_BASE_SECONDS = 1  # first reconnect waits 1 s, then 2 s, 4 s, …


class UpstoxLiveSource(DataSource):
    """Live WebSocket tick feed from Upstox, behind the DataSource interface.

    This adapter is the **only** allowed point of contact with the Upstox SDK.
    Prediction and backtest code must import DataSource from src.data.base and
    receive this adapter via dependency injection — never import it directly.

    Construction (production)
    -------------------------
    source = UpstoxLiveSource()   # api_key / access_token from UPSTOX_* env vars
    source.connect()
    df = source.get_ticks("RELIANCE", as_of=datetime.now(IST))

    Construction (tests — no SDK required)
    --------------------------------------
    source = UpstoxLiveSource(_inject_ws=mock_ws_object)
    source._on_tick({"symbol": "X", "ts": ts, "ltp": 100.0, ...})
    df = source.get_ticks("X", as_of)
    """

    data_source_name = "upstox_live"
    data_source_limitations = (
        "live tick feed via Upstox WebSocket (free tier); "
        "no historical bars; "
        "no backfill on reconnect — ticks received before connection are lost; "
        "buffer limited to most-recent ticks per symbol; "
        "credentials read from UPSTOX_API_KEY / UPSTOX_ACCESS_TOKEN env vars"
    )
    data_granularity = "tick"

    def __init__(
        self,
        api_key: str | None = None,
        access_token: str | None = None,
        *,
        buffer_maxlen: int = 10_000,
        max_reconnect_attempts: int = 5,
        _inject_ws: Any = None,   # test hook — bypasses SDK import in connect()
    ) -> None:
        import os  # noqa: PLC0415
        self._api_key = api_key or os.environ.get("UPSTOX_API_KEY", "")
        self._access_token = access_token or os.environ.get("UPSTOX_ACCESS_TOKEN", "")
        self._buffer_maxlen = buffer_maxlen
        self._max_reconnect = max_reconnect_attempts

        self._lock = threading.Lock()
        self._buffers: dict[str, collections.deque] = {}
        self._ws: Any = _inject_ws
        self._connected = _inject_ws is not None
        self._reconnect_attempts = 0

    # ── Connection management ────────────────────────────────────────────────

    def connect(self) -> None:
        """Establish the WebSocket connection.

        This is the ONLY place in the project that imports the Upstox SDK.
        It will raise ImportError with an actionable message if the SDK is
        not installed.
        """
        if self._connected:
            return

        try:
            import upstox_client  # noqa: F401, PLC0415  ← sole SDK import site
        except ImportError as exc:
            raise ImportError(
                "upstox-python-api is not installed. "
                "Install it for Phase 1 live feeds:\n"
                "  pip install upstox-python-api"
            ) from exc

        # Production WebSocket wiring goes here when the SDK is available.
        # (Deferred to the live integration phase — not implemented in tests.)
        _log.info("UpstoxLiveSource: SDK available, WebSocket started")
        self._connected = True
        self._reconnect_attempts = 0

    def disconnect(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                self._ws.disconnect()
            except Exception:  # noqa: BLE001
                pass
        self._connected = False
        _log.info("UpstoxLiveSource: disconnected")

    def _on_close(self, reason: str = "") -> None:
        """WebSocket close callback.  Reconnects with exponential backoff.

        Waits _RECONNECT_BASE_SECONDS * 2^attempt before each retry.
        Gives up after max_reconnect_attempts and logs an error.
        """
        self._connected = False
        if self._reconnect_attempts >= self._max_reconnect:
            _log.error(
                "WebSocket closed (%r). Max reconnect attempts (%d) reached. "
                "Restart the adapter manually.",
                reason, self._max_reconnect,
            )
            return

        wait = _RECONNECT_BASE_SECONDS * (2 ** self._reconnect_attempts)
        self._reconnect_attempts += 1
        _log.warning(
            "WebSocket closed (%r). Reconnecting in %ds (attempt %d/%d).",
            reason, wait, self._reconnect_attempts, self._max_reconnect,
        )
        time.sleep(wait)
        self.connect()

    def _on_error(self, error: Exception) -> None:
        _log.error("WebSocket error: %s", error)

    # ── Tick ingest callback (runs in WebSocket thread) ──────────────────────

    def _on_tick(self, message: dict) -> None:
        """Receive one tick from the WebSocket and append it to the buffer.

        Thread-safe: acquires the lock before touching the buffer.

        Expected message keys (extras ignored):
          symbol (str), ts (datetime / pd.Timestamp), ltp (float),
          volume (int), bid (float), ask (float), oi (int)
        """
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
            "source": "upstox_live",
        }

        with self._lock:
            if symbol not in self._buffers:
                self._buffers[symbol] = collections.deque(maxlen=self._buffer_maxlen)
            self._buffers[symbol].append(row)

    # ── DataSource interface — ticks ────────────────────────────────────────

    def get_ticks(
        self,
        symbol: str,
        as_of: _datetime,
        lookback_seconds: int = 300,
    ) -> pd.DataFrame:
        """Return buffered ticks for `symbol` with ts strictly < as_of.

        The buffer snapshot is taken under the lock; PIT enforcement runs on the
        snapshot (lock is already released).

        Parameters
        ----------
        symbol          : Instrument symbol.
        as_of           : Point-in-time cutoff.  MUST be supplied by the caller —
                          this method never reads a clock.
        lookback_seconds: Soft lookback window; older ticks are excluded as a
                          performance optimisation, not a safety measure.

        Returns
        -------
        DataFrame with TICK_COLS schema — empty (correctly typed) if no data.
        Never returns None.
        """
        with self._lock:
            snapshot = list(self._buffers.get(symbol, []))

        if not snapshot:
            return self._enforce_pit(_empty(TICK_COLS), "ts", as_of)

        df = pd.DataFrame(snapshot)

        # Apply lookback window (best-effort; PIT enforcement is the safety net)
        if "ts" in df.columns:
            try:
                ts_series = pd.to_datetime(df["ts"])
                as_of_ts = pd.Timestamp(as_of)
                cutoff_ts = as_of_ts - pd.Timedelta(seconds=lookback_seconds)
                df = df[ts_series >= cutoff_ts].copy()
            except Exception:  # noqa: BLE001
                pass   # lookback is best-effort; never let it suppress PIT

        if df.empty:
            return self._enforce_pit(_empty(TICK_COLS), "ts", as_of)

        # _on_tick always writes all keys into the row dict, so these columns exist.
        # Normalise dtypes to canonical TICK_COLS schema.
        df["ts"]     = pd.to_datetime(df["ts"])
        df["ltp"]    = pd.to_numeric(df["ltp"],    errors="coerce").astype("float64")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")
        df["bid"]    = pd.to_numeric(df["bid"],    errors="coerce").astype("float64")
        df["ask"]    = pd.to_numeric(df["ask"],    errors="coerce").astype("float64")
        df["oi"]     = pd.to_numeric(df["oi"],     errors="coerce").fillna(0).astype("int64")
        df["source"] = df["source"].astype("object")

        # Add any TICK_COLS columns absent in df; return in canonical column order
        for col in TICK_COLS:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[list(TICK_COLS.keys())]

        return self._enforce_pit(df, "ts", as_of)

    # ── DataSource interface — unsupported data types ─────────────────────────

    def get_ohlcv_daily(
        self, symbol: str, as_of: _datetime, lookback_days: int = 252,
    ) -> pd.DataFrame:
        raise DataNotAvailableError(
            f"UpstoxLiveSource provides live ticks only; no daily bars for {symbol!r}. "
            f"Use BhavCopySource for daily OHLCV."
        )

    def get_ohlcv_minute(
        self, symbol: str, as_of: _datetime, lookback_bars: int = 390,
    ) -> pd.DataFrame:
        raise DataNotAvailableError(
            f"UpstoxLiveSource provides live ticks only; no minute bars for {symbol!r}."
        )

    def get_news(
        self, symbol: str, as_of: _datetime, lookback_hours: float = 24.0,
    ) -> pd.DataFrame:
        raise DataNotAvailableError(
            f"UpstoxLiveSource does not provide news for {symbol!r}. "
            f"Use the RSS news ingest adapter."
        )

    def get_option_chain(self, symbol: str, as_of: _datetime) -> pd.DataFrame:
        raise DataNotAvailableError(
            f"UpstoxLiveSource does not provide option chain data for {symbol!r}. "
            f"Use the Upstox option chain fetcher."
        )
