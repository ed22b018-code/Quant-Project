"""F&O features — gate=FO (require fo_daily_metrics from Dhan bulk download).

Features:
    pcr_oi              Latest Put/Call ratio by open interest
    pcr_volume          Latest Put/Call ratio by volume
    net_oi_change       Net OI change as fraction of total (CE+PE) OI
    max_pain_distance   (spot - max_pain_strike) / spot
    spot_futures_basis  (futures_settle - spot) / spot

PIT guarantee
-------------
All queries use WHERE date < CAST(? AS DATE).
A row dated 2024-06-10 is NOT visible when as_of = 2024-06-10T10:00.
EOD F&O settlement for date D is knowable only after market close of D.

Data-availability note
----------------------
These features require fo_daily_metrics, which is populated by the Dhan bulk
historical downloader.  Until the download runs, all five features return None
(the FO gate is closed).  Build and test with synthetic data now; validate on
real F&O data after the Dhan download completes.
"""
from __future__ import annotations

from datetime import datetime

import duckdb

from src.features.base import DataGate, FeatureRegistry


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fetch_fo(db_path: str, symbol: str, as_of: datetime, cols: str) -> list:
    with duckdb.connect(db_path, read_only=True, config={'threads': '1'}) as con:
        return con.execute(
            f"SELECT {cols} FROM fo_daily_metrics "
            "WHERE symbol = ? AND date < CAST(? AS DATE) "
            "ORDER BY date DESC LIMIT 1",
            [symbol, as_of.date()],
        ).fetchall()


def _fetch_spot(db_path: str, symbol: str, as_of: datetime) -> float | None:
    with duckdb.connect(db_path, read_only=True, config={'threads': '1'}) as con:
        rows = con.execute(
            "SELECT close FROM ohlcv_daily "
            "WHERE symbol = ? AND date < CAST(? AS DATE) "
            "ORDER BY date DESC LIMIT 1",
            [symbol, as_of.date()],
        ).fetchall()
    return rows[0][0] if rows else None


# ── F&O features ──────────────────────────────────────────────────────────────


@FeatureRegistry.register("pcr_oi", DataGate.FO,
                           "Put/Call ratio by open interest (most recent F&O date)")
def pcr_oi(symbol: str, as_of: datetime, db_path: str) -> float | None:
    rows = _fetch_fo(db_path, symbol, as_of, "pcr_oi")
    return rows[0][0] if rows else None


@FeatureRegistry.register("pcr_volume", DataGate.FO,
                           "Put/Call ratio by volume (most recent F&O date)")
def pcr_volume(symbol: str, as_of: datetime, db_path: str) -> float | None:
    rows = _fetch_fo(db_path, symbol, as_of, "pcr_volume")
    return rows[0][0] if rows else None


@FeatureRegistry.register("net_oi_change", DataGate.FO,
                           "Net OI change / total (CE+PE) OI — signed, normalized")
def net_oi_change(symbol: str, as_of: datetime, db_path: str) -> float | None:
    rows = _fetch_fo(db_path, symbol, as_of, "net_oi_change, total_ce_oi, total_pe_oi")
    if not rows:
        return None
    oi_chg, ce_oi, pe_oi = rows[0]
    if oi_chg is None:
        return None
    total = (ce_oi or 0) + (pe_oi or 0)
    return oi_chg / total if total > 0 else None


@FeatureRegistry.register("max_pain_distance", DataGate.FO,
                           "Distance from spot to max-pain strike: (spot - max_pain) / spot")
def max_pain_distance(symbol: str, as_of: datetime, db_path: str) -> float | None:
    fo_rows = _fetch_fo(db_path, symbol, as_of, "max_pain_strike")
    if not fo_rows or fo_rows[0][0] is None:
        return None
    max_pain = fo_rows[0][0]
    spot = _fetch_spot(db_path, symbol, as_of)
    if spot is None or spot <= 0:
        return None
    return (spot - max_pain) / spot


@FeatureRegistry.register("spot_futures_basis", DataGate.FO,
                           "Futures premium vs spot: (futures_settle - spot) / spot")
def spot_futures_basis(symbol: str, as_of: datetime, db_path: str) -> float | None:
    fo_rows = _fetch_fo(db_path, symbol, as_of, "futures_settle")
    if not fo_rows or fo_rows[0][0] is None:
        return None
    futures = fo_rows[0][0]
    spot = _fetch_spot(db_path, symbol, as_of)
    if spot is None or spot <= 0:
        return None
    return (futures - spot) / spot
