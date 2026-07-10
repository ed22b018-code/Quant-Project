"""Price and volume features.

Daily features (gate=NONE — always available from ohlcv_daily):
    return_1d          1-day price return
    return_5d          5-day price return
    vol_20d            20-day realized volatility (annualized std of log returns)
    volume_ratio_20d   Latest volume / 20-day average volume
    atr_20d            20-day ATR as fraction of closing price

Intraday features (gate=INTRADAY — require ohlcv_intraday from Dhan bulk download):
    vwap_deviation          (close - VWAP) / VWAP using 1-min bars for the day
    intraday_momentum       (close_now - open_of_day) / open_of_day
    opening_range_position  Position within the 9:15–9:30 IST opening range

PIT guarantee (DESIGN.md §2 rule 1)
------------------------------------
Daily features:
    WHERE date < CAST(? AS DATE)
    A bar dated 2024-06-10 is NOT visible when as_of = 2024-06-10T10:00.
    EOD bar for date D is knowable only from midnight of D+1.

Intraday features:
    WHERE ts < ?  (TIMESTAMPTZ strict less-than)
    A bar timestamped exactly as_of is NOT visible.

Tests in test_features.py verify each feature by inserting a 'poison' row
exactly at the boundary and confirming it does not affect the computed value.

Data-availability note
----------------------
INTRADAY features are built and unit-tested against synthetic data now.
Full validation on real NSE minute bars runs after the Dhan bulk download.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

import duckdb

from src.features.base import DataGate, FeatureRegistry

UTC = timezone.utc


# ── Internal helpers ──────────────────────────────────────────────────────────

def _daily(db_path: str, symbol: str, as_of: datetime, cols: str, limit: int) -> list:
    """Fetch daily OHLCV rows strictly before as_of.date(), newest first."""
    with duckdb.connect(db_path, read_only=True, config={'threads': '1'}) as con:
        return con.execute(
            f"SELECT {cols} FROM ohlcv_daily "
            "WHERE symbol = ? AND date < CAST(? AS DATE) "
            "ORDER BY date DESC LIMIT ?",
            [symbol, as_of.date(), limit],
        ).fetchall()


def _intraday_today(db_path: str, symbol: str, as_of: datetime, cols: str) -> list:
    """Fetch 1-min bars for as_of's trading date where ts < as_of, oldest first.

    Uses CAST(ts AS DATE) to match the UTC trading date; NSE sessions
    run 03:45–10:00 UTC (09:15–15:30 IST), always within the same UTC date.
    """
    as_of_utc = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
    with duckdb.connect(db_path, read_only=True, config={'threads': '1'}) as con:
        return con.execute(
            f"SELECT {cols} FROM ohlcv_intraday "
            "WHERE symbol = ? AND timeframe = '1' "
            "  AND ts < ? "
            "  AND CAST(ts AS DATE) = CAST(? AS DATE) "
            "ORDER BY ts",
            [symbol, as_of_utc, as_of_utc.date()],
        ).fetchall()


# ── Daily features (gate=NONE) ────────────────────────────────────────────────


@FeatureRegistry.register("return_1d", DataGate.NONE,
                           "1-day price return: (close_t - close_t-1) / close_t-1")
def return_1d(symbol: str, as_of: datetime, db_path: str) -> float | None:
    rows = _daily(db_path, symbol, as_of, "close", 2)
    if len(rows) < 2:
        return None
    c1, c0 = rows[0][0], rows[1][0]
    return (c1 - c0) / c0 if c0 > 0 else None


@FeatureRegistry.register("return_5d", DataGate.NONE,
                           "5-day price return: (close_t - close_t-5) / close_t-5")
def return_5d(symbol: str, as_of: datetime, db_path: str) -> float | None:
    rows = _daily(db_path, symbol, as_of, "close", 6)
    if len(rows) < 6:
        return None
    c_now, c_5d = rows[0][0], rows[5][0]
    return (c_now - c_5d) / c_5d if c_5d > 0 else None


@FeatureRegistry.register("vol_20d", DataGate.NONE,
                           "20-day realized volatility: annualized std of daily log returns")
def vol_20d(symbol: str, as_of: datetime, db_path: str) -> float | None:
    rows = _daily(db_path, symbol, as_of, "close", 21)
    if len(rows) < 3:
        return None
    closes = [r[0] for r in rows]
    log_rets = [
        math.log(closes[i] / closes[i + 1])
        for i in range(len(closes) - 1)
        if closes[i] > 0 and closes[i + 1] > 0
    ]
    if len(log_rets) < 2:
        return None
    n = len(log_rets)
    mean = sum(log_rets) / n
    variance = sum((r - mean) ** 2 for r in log_rets) / (n - 1)
    return math.sqrt(variance) * math.sqrt(252)


@FeatureRegistry.register("volume_ratio_20d", DataGate.NONE,
                           "Latest day volume / 20-day average volume")
def volume_ratio_20d(symbol: str, as_of: datetime, db_path: str) -> float | None:
    rows = _daily(db_path, symbol, as_of, "volume", 21)
    if len(rows) < 2:
        return None
    latest = rows[0][0]
    prior = [r[0] for r in rows[1:]]
    avg = sum(prior) / len(prior)
    return latest / avg if avg > 0 else None


@FeatureRegistry.register("atr_20d", DataGate.NONE,
                           "20-day Average True Range as fraction of latest close")
def atr_20d(symbol: str, as_of: datetime, db_path: str) -> float | None:
    rows = _daily(db_path, symbol, as_of, "high, low, close", 21)
    if len(rows) < 2:
        return None
    trs = []
    for i in range(len(rows) - 1):
        h, l, _ = rows[i]
        _, _, prev_c = rows[i + 1]
        trs.append(max(h - l, abs(h - prev_c), abs(l - prev_c)))
    if not trs:
        return None
    atr = sum(trs) / len(trs)
    last_close = rows[0][2]
    return atr / last_close if last_close > 0 else None


# ── Intraday features (gate=INTRADAY) ─────────────────────────────────────────


@FeatureRegistry.register("vwap_deviation", DataGate.INTRADAY,
                           "Intraday VWAP deviation: (close - VWAP) / VWAP using 1-min bars")
def vwap_deviation(symbol: str, as_of: datetime, db_path: str) -> float | None:
    rows = _intraday_today(db_path, symbol, as_of, "close, volume")
    if not rows:
        return None
    total_vol = sum(r[1] for r in rows)
    if total_vol <= 0:
        return None
    vwap = sum(r[0] * r[1] for r in rows) / total_vol
    last_close = rows[-1][0]
    return (last_close - vwap) / vwap if vwap > 0 else None


@FeatureRegistry.register("intraday_momentum", DataGate.INTRADAY,
                           "Return from open-of-day to current close using 1-min bars")
def intraday_momentum(symbol: str, as_of: datetime, db_path: str) -> float | None:
    rows = _intraday_today(db_path, symbol, as_of, "open, close")
    if not rows:
        return None
    day_open = rows[0][0]
    last_close = rows[-1][1]
    return (last_close - day_open) / day_open if day_open > 0 else None


@FeatureRegistry.register("opening_range_position", DataGate.INTRADAY,
                           "Current price position within the 9:15–9:30 IST opening range")
def opening_range_position(symbol: str, as_of: datetime, db_path: str) -> float | None:
    """Position of current price relative to the first-15-minute opening range.

    Returns (close - ORL) / (ORH - ORL). Values outside [0, 1] indicate price
    has broken out of the opening range; returns None if the range is not yet
    complete at as_of (i.e., as_of < 9:30 IST = 04:00 UTC).

    Opening range: 9:15–9:30 IST = 03:45–04:00 UTC for NSE sessions.
    PIT: the OR bars (03:45–03:59) are all in the past once as_of >= 04:00 UTC.
    Current price uses WHERE ts < as_of (strict).
    """
    as_of_utc = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
    d = as_of_utc.date()
    or_start = datetime(d.year, d.month, d.day, 3, 45, tzinfo=UTC)
    or_end = datetime(d.year, d.month, d.day, 4, 0, tzinfo=UTC)

    if as_of_utc < or_end:
        return None

    with duckdb.connect(db_path, read_only=True, config={'threads': '1'}) as con:
        or_rows = con.execute(
            "SELECT MAX(high), MIN(low) FROM ohlcv_intraday "
            "WHERE symbol = ? AND timeframe = '1' AND ts >= ? AND ts < ?",
            [symbol, or_start, or_end],
        ).fetchall()
        if not or_rows or or_rows[0][0] is None:
            return None
        orh, orl = or_rows[0]

        cur_rows = con.execute(
            "SELECT close FROM ohlcv_intraday "
            "WHERE symbol = ? AND timeframe = '1' AND ts < ? "
            "ORDER BY ts DESC LIMIT 1",
            [symbol, as_of_utc],
        ).fetchall()
        if not cur_rows:
            return None
        current = cur_rows[0][0]

    if orh <= orl:
        return 0.5
    return (current - orl) / (orh - orl)
