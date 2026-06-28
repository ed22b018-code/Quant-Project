"""OptionChainSource — live Upstox option chain fetcher.

Fetches option chain data via plain urllib.request (no SDK import anywhere in
this file).  Caches in-memory per symbol for up to `cache_ttl_seconds` (default
60 s) to respect Upstox rate limits.  On fetch failure falls back to the last
successfully fetched chain for the symbol, emitting a WARNING that names the
staleness age — callers can detect staleness by inspecting ts.max() vs as_of.

PIT enforcement:
  The `ts` column records when the chain was actually fetched from the API.
  All rows in a snapshot share the same ts (the fetch time).  _enforce_pit()
  will reject rows where ts >= as_of.  For live use (as_of ≈ now) this always
  passes.  For backtest use with historical as_of, ts will be in the future →
  DataLeakageError — intentional; option chain snapshots are live-only.

Not a DataSource subclass:
  Option chain data lives alongside tick feeds; it is not part of the daily /
  minute / tick hierarchy.  It exposes get_option_chain() directly.

CPU-only — no GPU library imports (torch, cupy, etc.).
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Callable

import pandas as pd

from src.data.base import DataLeakageError, DataNotAvailableError, DataSource
from src.data.historical import OPTION_CHAIN_COLS, _empty

_log = logging.getLogger(__name__)

_OPTION_CHAIN_URL = "https://api.upstox.com/v2/option/chain"
_BACKOFF_BASE: float = 1.0  # seconds


class OptionChainSource(DataSource):
    """Live Upstox option chain fetcher with in-memory cache and fallback.

    Parameters
    ----------
    access_token:
        Upstox OAuth2 access token.  Falls back to UPSTOX_ACCESS_TOKEN env var.
    cache_ttl_seconds:
        Seconds before a cached chain is considered stale and re-fetched.
    max_retries:
        Maximum HTTP fetch attempts (with exponential backoff between them).
    _mock_fetch:
        Test hook — callable(symbol, expiry_date) → raw JSON dict.  When set,
        bypasses all HTTP logic.  Supply ``None`` in production.
    _fetch_ts_override:
        Test hook — datetime to use as the fetch timestamp instead of
        datetime.now(UTC).  Lets tests control the ts column value.
    """

    data_source_name = "upstox_option_chain"
    data_source_limitations = (
        "live option chain only (Upstox free tier); "
        "no historical snapshots — backtest use will raise DataLeakageError "
        "because ts (fetch time) > historical as_of; "
        "cached in-memory for cache_ttl_seconds (default 60 s) to respect rate "
        "limits; fallback to last-good chain on fetch failure — ts reflects the "
        "original fetch time so callers can detect staleness via ts.max(); "
        "front-month expiry fetched; credentials via UPSTOX_ACCESS_TOKEN env var"
    )
    data_granularity = "tick"

    def __init__(
        self,
        access_token: str | None = None,
        *,
        cache_ttl_seconds: int = 60,
        max_retries: int = 3,
        _mock_fetch: Callable | None = None,
        _fetch_ts_override: datetime | None = None,
    ) -> None:
        import os
        self._access_token = access_token or os.environ.get("UPSTOX_ACCESS_TOKEN", "")
        self._cache_ttl = timedelta(seconds=cache_ttl_seconds)
        self._max_retries = max_retries
        self._mock_fetch = _mock_fetch
        self._fetch_ts_override = _fetch_ts_override

        # (DataFrame, wall_clock_fetch_time) — used for TTL checks only
        self._cache: dict[str, tuple[pd.DataFrame, datetime]] = {}
        # last successfully fetched chain per symbol (may be stale vs as_of)
        self._last_good: dict[str, pd.DataFrame] = {}

    # ── Public API ───────────────────────────────────────────────────────────

    def get_option_chain(self, symbol: str, as_of: datetime) -> pd.DataFrame:
        """Return option chain for `symbol` with all ts < as_of.

        Fetches live from Upstox if the cache is stale; falls back to the last
        good chain (with a WARNING) if the live fetch fails.

        Raises
        ------
        DataLeakageError
            If any row has ts >= as_of (e.g. backtest with historical as_of).
        DataNotAvailableError
            If the fetch fails and no fallback is available.
        """
        # Check in-memory cache (wall-clock staleness, not as_of staleness)
        if symbol in self._cache:
            cached_df, fetch_wall_time = self._cache[symbol]
            cache_age = datetime.now() - fetch_wall_time
            if cache_age < self._cache_ttl:
                _log.debug(
                    "OptionChainSource: cache hit for %r (age %.1f s)",
                    symbol, cache_age.total_seconds(),
                )
                return self._enforce_pit(cached_df, "ts", as_of)

        # Cache miss — fetch live
        df = self._fetch_with_backoff(symbol)

        if df is not None:
            self._cache[symbol] = (df, datetime.now())
            self._last_good[symbol] = df
            return self._enforce_pit(df, "ts", as_of)

        # Fetch failed — try fallback
        if symbol in self._last_good:
            fallback = self._last_good[symbol]
            if not fallback.empty and "ts" in fallback.columns:
                last_ts = pd.to_datetime(fallback["ts"].max())
                try:
                    age_s = (datetime.now(timezone.utc) - last_ts.tz_convert("UTC")).total_seconds()
                except Exception:
                    age_s = float("nan")
                _log.warning(
                    "OptionChainSource: live fetch failed for %r; "
                    "using last-good chain (%.0f s old, ts=%s). "
                    "Callers can detect staleness via df['ts'].max() vs as_of.",
                    symbol, age_s, last_ts,
                )
            return self._enforce_pit(fallback, "ts", as_of)

        raise DataNotAvailableError(
            f"OptionChainSource: live fetch failed for {symbol!r} and no fallback "
            "is available.  Check network connectivity and UPSTOX_ACCESS_TOKEN."
        )

    # ── DataSource abstract method stubs ─────────────────────────────────────

    def get_ohlcv_daily(self, symbol, as_of, lookback_days=252):
        raise DataNotAvailableError(
            f"OptionChainSource does not provide OHLCV data for {symbol!r}."
        )

    def get_ohlcv_minute(self, symbol, as_of, lookback_bars=390):
        raise DataNotAvailableError(
            f"OptionChainSource does not provide minute bars for {symbol!r}."
        )

    def get_ticks(self, symbol, as_of, lookback_seconds=300):
        raise DataNotAvailableError(
            f"OptionChainSource does not provide tick data for {symbol!r}.  "
            "Use UpstoxLiveSource for ticks."
        )

    def get_news(self, symbol, as_of, lookback_hours=24.0):
        raise DataNotAvailableError(
            f"OptionChainSource does not provide news for {symbol!r}.  "
            "Use NewsSource for news."
        )

    # ── Internal fetch logic ─────────────────────────────────────────────────

    def _get_fetch_ts(self) -> datetime:
        return self._fetch_ts_override if self._fetch_ts_override is not None else datetime.now(timezone.utc)

    def _do_fetch(self, symbol: str) -> dict:
        """Fetch raw JSON from Upstox or mock.  Raises on HTTP error."""
        if self._mock_fetch is not None:
            return self._mock_fetch(symbol, None)

        from urllib.parse import urlencode

        params = urlencode({"instrument_key": f"NSE_EQ|{symbol}"})
        url = f"{_OPTION_CHAIN_URL}?{params}"
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self._access_token}",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _fetch_with_backoff(self, symbol: str) -> pd.DataFrame | None:
        """Attempt to fetch the option chain with exponential backoff on failure."""
        last_exc: Exception | None = None

        for attempt in range(self._max_retries):
            try:
                fetch_ts = self._get_fetch_ts()
                raw = self._do_fetch(symbol)
                return self._parse_response(raw, fetch_ts)

            except urllib.error.HTTPError as exc:
                last_exc = exc
                if exc.code == 429:
                    wait = _BACKOFF_BASE * (2 ** attempt)
                    _log.warning(
                        "OptionChainSource: rate-limited (HTTP 429) for %r; "
                        "waiting %.1f s (attempt %d/%d).",
                        symbol, wait, attempt + 1, self._max_retries,
                    )
                    time.sleep(wait)
                else:
                    _log.warning(
                        "OptionChainSource: HTTP %d for %r: %s (attempt %d/%d).",
                        exc.code, symbol, exc, attempt + 1, self._max_retries,
                    )
                    if attempt < self._max_retries - 1:
                        wait = _BACKOFF_BASE * (2 ** attempt)
                        time.sleep(wait)

            except Exception as exc:
                last_exc = exc
                _log.warning(
                    "OptionChainSource: fetch error for %r: %s (attempt %d/%d).",
                    symbol, exc, attempt + 1, self._max_retries,
                )
                if attempt < self._max_retries - 1:
                    wait = _BACKOFF_BASE * (2 ** attempt)
                    time.sleep(wait)

        _log.error(
            "OptionChainSource: all %d fetch attempts failed for %r.  Last error: %s",
            self._max_retries, symbol, last_exc,
        )
        return None

    def _parse_response(self, raw: dict, fetch_ts: datetime) -> pd.DataFrame:
        """Convert Upstox JSON response to an OPTION_CHAIN_COLS DataFrame."""
        from datetime import date as _date

        if raw.get("status") != "success":
            raise ValueError(
                f"Unexpected Upstox API status: {raw.get('status')!r}.  "
                f"Full response keys: {list(raw.keys())}"
            )

        data = raw.get("data", [])
        if not data:
            return _empty(OPTION_CHAIN_COLS)

        rows = []
        for item in data:
            expiry_str = item.get("expiry", "")
            try:
                expiry = _date.fromisoformat(expiry_str)
            except (ValueError, TypeError):
                _log.warning("OptionChainSource: cannot parse expiry %r, skipping row.", expiry_str)
                continue

            strike = float(item.get("strike_price", 0) or 0)

            for opt_type, key in (("CE", "call_options"), ("PE", "put_options")):
                md = (item.get(key) or {}).get("market_data") or {}
                oi_val = int(md.get("oi", 0) or 0)
                prev_oi_val = int(md.get("prev_oi", 0) or 0)
                rows.append({
                    "expiry":      expiry,
                    "strike":      strike,
                    "option_type": opt_type,
                    "ltp":         float(md.get("ltp", 0) or 0),
                    "oi":          oi_val,
                    "oi_change":   oi_val - prev_oi_val,
                    "volume":      int(md.get("volume", 0) or 0),
                    "iv":          float(md.get("iv", 0) or 0),
                    "ts":          fetch_ts,
                })

        if not rows:
            return _empty(OPTION_CHAIN_COLS)

        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df[list(OPTION_CHAIN_COLS.keys())]
