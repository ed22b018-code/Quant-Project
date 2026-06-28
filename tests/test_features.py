"""Tests for src/features — Checkpoint 2.1 Feature Engineering.

Structure
---------
Each of the 15 registered features gets a dedicated PIT test proving that a row
inserted exactly AT as_of is excluded from the computation, plus a correctness
test on synthetic data.  Additional meta-tests cover the registry, gate filtering,
and the no-GPU-imports invariant.

PIT proof pattern
-----------------
  1. Seed the DB with 'correct' rows before as_of.
  2. Insert a 'poison' row timestamped EXACTLY AT as_of (boundary).
  3. Assert the computed value equals the expected value from correct rows only
     (or None when insufficient pre-as_of rows exist).

If the poison row were visible, the result would be wrong — that's the test.

Data-availability flags
-----------------------
INTRADAY and FO features are tested against synthetic data now.
Full validation on real NSE minute-bar / F&O data runs after Dhan download.
Tests that require INTRADAY/FO data are annotated with #[needs-real-data].
"""
from __future__ import annotations

import math
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pytest

from src.db.schema import DUCKDB_SCHEMA_SQL
from src.db.duckdb_client import DuckDBClient
from src.features import DataGate, FeatureMeta, FeatureRegistry, compute_features
from src.features.price_volume import (
    atr_20d,
    intraday_momentum,
    opening_range_position,
    return_1d,
    return_5d,
    vol_20d,
    volume_ratio_20d,
    vwap_deviation,
)
from src.features.fo import (
    max_pain_distance,
    net_oi_change,
    pcr_oi,
    pcr_volume,
    spot_futures_basis,
)
from src.features.news_count import news_count_1d, news_count_5d

UTC = timezone.utc

# ── Reference timestamps ──────────────────────────────────────────────────────
# Daily AS_OF: Jun 10 → visible dates are strictly < Jun 10 (bar for Jun 10
# is knowable only from midnight Jun 11).
AS_OF = datetime(2024, 6, 10, 10, 0, tzinfo=UTC)
AS_OF_DATE = AS_OF.date()  # 2024-06-10

# Intraday AS_OF: Jun 10, 05:00 UTC = 10:30 IST (mid-session)
AS_OF_INTRA = datetime(2024, 6, 10, 5, 0, tzinfo=UTC)

# Opening-range AS_OF: must be >= 04:00 UTC (9:30 IST — OR complete)
AS_OF_OR = datetime(2024, 6, 10, 4, 30, tzinfo=UTC)


# ── DB helpers ────────────────────────────────────────────────────────────────

def _setup(db_path: Path) -> None:
    with DuckDBClient(db_path) as conn:
        conn.execute_schema(DUCKDB_SCHEMA_SQL)


def _ins_daily(
    db_path: Path,
    symbol: str,
    day: date,
    *,
    close: float,
    high: float | None = None,
    low: float | None = None,
    open_: float | None = None,
    volume: int = 1_000_000,
) -> None:
    h = high if high is not None else close * 1.01
    lo = low if low is not None else close * 0.99
    o = open_ if open_ is not None else close
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            "INSERT INTO ohlcv_daily "
            "(symbol, date, open, high, low, close, volume, traded_value, "
            " series, delivery_qty, source, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            [symbol, day, o, h, lo, close, volume,
             float(volume * close), "EQ", None, "test",
             datetime.now(UTC)],
        )


def _ins_intraday(
    db_path: Path,
    symbol: str,
    ts: datetime,
    *,
    close: float,
    volume: int = 1000,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
) -> None:
    h = high if high is not None else close * 1.001
    lo = low if low is not None else close * 0.999
    o = open_ if open_ is not None else close
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            "INSERT INTO ohlcv_intraday "
            "(symbol, timeframe, ts, open, high, low, close, volume, source, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            [symbol, "1", ts, o, h, lo, close, volume, "test", datetime.now(UTC)],
        )


def _ins_fo(
    db_path: Path,
    symbol: str,
    day: date,
    *,
    pcr_oi_val: float = 1.0,
    pcr_vol_val: float = 1.0,
    total_ce_oi: int = 1_000_000,
    total_pe_oi: int = 1_000_000,
    net_oi_chg: int = 0,
    max_pain: float = 100.0,
    futures_settle: float | None = None,
) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            "INSERT INTO fo_daily_metrics "
            "(symbol, date, pcr_oi, pcr_volume, total_ce_oi, total_pe_oi, "
            " net_oi_change, max_pain_strike, futures_settle, source, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            [symbol, day, pcr_oi_val, pcr_vol_val, total_ce_oi, total_pe_oi,
             net_oi_chg, max_pain, futures_settle, "test", datetime.now(UTC)],
        )


def _ins_news(
    db_path: Path,
    symbol: str,
    ts: datetime,
    news_id: str,
) -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            "INSERT INTO news "
            "(news_id, headline, body, source, published_at, first_knowable_at, "
            " symbols, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            [news_id, "Test headline", None, "test",
             ts, ts, ["TEST"], datetime.now(UTC)],
        )


# ── return_1d ─────────────────────────────────────────────────────────────────

def test_return_1d_pit(tmp_duckdb_path):
    """Poison bar dated AS_OF_DATE is excluded; result uses Jun 8+9 only."""
    _setup(tmp_duckdb_path)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 8), close=100.0)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=102.0)
    _ins_daily(tmp_duckdb_path, "T", AS_OF_DATE, close=99_999.0)  # poison

    result = return_1d("T", AS_OF, str(tmp_duckdb_path))
    assert result == pytest.approx((102.0 - 100.0) / 100.0)


def test_return_1d_correct(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 8), close=100.0)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=105.0)
    assert return_1d("T", AS_OF, str(tmp_duckdb_path)) == pytest.approx(0.05)


def test_return_1d_insufficient_data_returns_none(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=100.0)  # only 1 row
    assert return_1d("T", AS_OF, str(tmp_duckdb_path)) is None


# ── return_5d ─────────────────────────────────────────────────────────────────

def test_return_5d_pit(tmp_duckdb_path):
    """Poison bar at AS_OF_DATE not seen; result spans Jun 4–9."""
    _setup(tmp_duckdb_path)
    for i, c in enumerate([100.0, 101.0, 102.0, 103.0, 104.0, 105.0], start=0):
        _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 4 + i), close=c)
    _ins_daily(tmp_duckdb_path, "T", AS_OF_DATE, close=99_999.0)  # poison

    result = return_5d("T", AS_OF, str(tmp_duckdb_path))
    # rows[0]=Jun9=105, rows[5]=Jun4=100 → (105-100)/100 = 0.05
    assert result == pytest.approx(0.05)


def test_return_5d_correct(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    closes = [100.0, 102.0, 101.0, 103.0, 104.0, 110.0]
    for i, c in enumerate(closes):
        _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 4 + i), close=c)
    result = return_5d("T", AS_OF, str(tmp_duckdb_path))
    assert result == pytest.approx((110.0 - 100.0) / 100.0)


def test_return_5d_insufficient_data_returns_none(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=100.0)
    assert return_5d("T", AS_OF, str(tmp_duckdb_path)) is None


# ── vol_20d ───────────────────────────────────────────────────────────────────

def test_vol_20d_pit(tmp_duckdb_path):
    """Poison bar at AS_OF_DATE excluded; vol computed from Jun 7-9 only."""
    _setup(tmp_duckdb_path)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 7), close=100.0)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 8), close=101.0)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=102.0)
    _ins_daily(tmp_duckdb_path, "T", AS_OF_DATE, close=99_999.0)  # poison

    result_with_poison = vol_20d("T", AS_OF, str(tmp_duckdb_path))

    # Without poison: same 3 bars → same result
    db2 = tmp_duckdb_path.parent / "test2.duckdb"
    _setup(db2)
    _ins_daily(db2, "T", date(2024, 6, 7), close=100.0)
    _ins_daily(db2, "T", date(2024, 6, 8), close=101.0)
    _ins_daily(db2, "T", date(2024, 6, 9), close=102.0)
    result_clean = vol_20d("T", AS_OF, str(db2))

    assert result_with_poison == pytest.approx(result_clean)
    assert result_with_poison is not None  # ensure real value returned


def test_vol_20d_correct(tmp_duckdb_path):
    """Volatility of constant returns should be 0."""
    _setup(tmp_duckdb_path)
    base = 100.0
    for i in range(10):
        _ins_daily(tmp_duckdb_path, "T", date(2024, 5, 1 + i), close=base)
    # All same price → 0 log returns → std = 0
    result = vol_20d("T", AS_OF, str(tmp_duckdb_path))
    assert result == pytest.approx(0.0, abs=1e-10)


def test_vol_20d_insufficient_data_returns_none(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=100.0)
    assert vol_20d("T", AS_OF, str(tmp_duckdb_path)) is None


# ── volume_ratio_20d ──────────────────────────────────────────────────────────

def test_volume_ratio_20d_pit(tmp_duckdb_path):
    """Poison bar at AS_OF_DATE excluded; ratio = latest_before_poison / avg_prior."""
    _setup(tmp_duckdb_path)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 8), close=100.0, volume=1_000_000)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=100.0, volume=2_000_000)
    _ins_daily(tmp_duckdb_path, "T", AS_OF_DATE, close=100.0, volume=99_999_999)  # poison

    result = volume_ratio_20d("T", AS_OF, str(tmp_duckdb_path))
    # Latest visible = Jun9 vol=2M, prior = [Jun8 vol=1M], ratio = 2.0
    assert result == pytest.approx(2.0)


def test_volume_ratio_20d_correct(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 8), close=100.0, volume=500_000)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=100.0, volume=1_000_000)
    result = volume_ratio_20d("T", AS_OF, str(tmp_duckdb_path))
    assert result == pytest.approx(2.0)


# ── atr_20d ───────────────────────────────────────────────────────────────────

def test_atr_20d_pit(tmp_duckdb_path):
    """Poison bar at AS_OF_DATE excluded; ATR computed from Jun 8-9 only."""
    _setup(tmp_duckdb_path)
    # Jun 8: h=105, l=95, c=100
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 8), close=100.0, high=105.0, low=95.0)
    # Jun 9: h=106, l=96, c=101  → TR = max(10, |106-100|, |96-100|) = 10
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=101.0, high=106.0, low=96.0)
    _ins_daily(tmp_duckdb_path, "T", AS_OF_DATE, close=99_999.0, high=100_100.0, low=0.1)  # poison

    result = atr_20d("T", AS_OF, str(tmp_duckdb_path))
    # ATR = 10 / 1 = 10; last_close = 101; fraction = 10/101
    assert result == pytest.approx(10.0 / 101.0, rel=1e-5)


def test_atr_20d_correct(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    # 3 bars; TR for row 0 vs row 1 is computed
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 7), close=100.0, high=102.0, low=98.0)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 8), close=100.0, high=105.0, low=95.0)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=101.0, high=106.0, low=96.0)
    result = atr_20d("T", AS_OF, str(tmp_duckdb_path))
    # TR[0] = max(106-96, |106-100|, |96-100|) = max(10, 6, 4) = 10  (row0 vs row1)
    # TR[1] = max(105-95, |105-100|, |95-100|) = max(10, 5, 5) = 10  (row1 vs row2)
    # ATR = 10; last_close = 101
    assert result == pytest.approx(10.0 / 101.0, rel=1e-5)


# ── vwap_deviation ────────────────────────────────────────────────────────────

def test_vwap_deviation_pit(tmp_duckdb_path):
    """Bar at exactly AS_OF_INTRA excluded; VWAP uses only prior 3 bars."""
    _setup(tmp_duckdb_path)
    base_ts = datetime(2024, 6, 10, 3, 45, tzinfo=UTC)
    for i in range(3):
        _ins_intraday(tmp_duckdb_path, "T", base_ts + timedelta(minutes=i),
                      close=100.0 + i, volume=1000)
    # Poison bar at exactly AS_OF_INTRA
    _ins_intraday(tmp_duckdb_path, "T", AS_OF_INTRA, close=99_999.0, volume=1)

    result = vwap_deviation("T", AS_OF_INTRA, str(tmp_duckdb_path))

    # Expected VWAP from 3 bars: (100+101+102)/3 = 101; last_close = 102
    expected_vwap = (100.0 * 1000 + 101.0 * 1000 + 102.0 * 1000) / 3000
    expected = (102.0 - expected_vwap) / expected_vwap
    assert result == pytest.approx(expected, rel=1e-5)


def test_vwap_deviation_correct(tmp_duckdb_path):
    """VWAP of constant-close bars == close; deviation == 0."""
    _setup(tmp_duckdb_path)
    base_ts = datetime(2024, 6, 10, 3, 45, tzinfo=UTC)
    for i in range(4):
        _ins_intraday(tmp_duckdb_path, "T", base_ts + timedelta(minutes=i),
                      close=100.0, volume=500)
    result = vwap_deviation("T", AS_OF_INTRA, str(tmp_duckdb_path))
    assert result == pytest.approx(0.0, abs=1e-10)


def test_vwap_deviation_empty_returns_none(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    assert vwap_deviation("T", AS_OF_INTRA, str(tmp_duckdb_path)) is None


# ── intraday_momentum ─────────────────────────────────────────────────────────

def test_intraday_momentum_pit(tmp_duckdb_path):
    """Poison bar at AS_OF_INTRA excluded; momentum from day_open to last visible."""
    _setup(tmp_duckdb_path)
    base_ts = datetime(2024, 6, 10, 3, 45, tzinfo=UTC)
    # Bar 0: open=100, close=100; bar 1: open=100, close=102
    _ins_intraday(tmp_duckdb_path, "T", base_ts, close=100.0, open_=100.0)
    _ins_intraday(tmp_duckdb_path, "T", base_ts + timedelta(minutes=1),
                  close=102.0, open_=100.0)
    _ins_intraday(tmp_duckdb_path, "T", AS_OF_INTRA, close=99_999.0)  # poison

    result = intraday_momentum("T", AS_OF_INTRA, str(tmp_duckdb_path))
    # day_open = rows[0].open = 100; last_close = rows[1].close = 102
    assert result == pytest.approx((102.0 - 100.0) / 100.0)


def test_intraday_momentum_correct(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    base_ts = datetime(2024, 6, 10, 3, 45, tzinfo=UTC)
    _ins_intraday(tmp_duckdb_path, "T", base_ts, close=100.0, open_=98.0)
    _ins_intraday(tmp_duckdb_path, "T", base_ts + timedelta(minutes=1),
                  close=105.0, open_=101.0)
    result = intraday_momentum("T", AS_OF_INTRA, str(tmp_duckdb_path))
    assert result == pytest.approx((105.0 - 98.0) / 98.0)


# ── opening_range_position ────────────────────────────────────────────────────

def _seed_or_bars(db_path: Path) -> None:
    """Seed 5 opening-range bars (03:45–03:49 UTC) and a post-OR bar."""
    for i in range(5):
        ts = datetime(2024, 6, 10, 3, 45 + i, tzinfo=UTC)
        # Simulate prices 95–105 across the OR window
        _ins_intraday(db_path, "T", ts, close=95.0 + i * 2.5,
                      high=96.0 + i * 2.5, low=94.0 + i * 2.5)
    # Post-OR bar at 04:10 UTC
    _ins_intraday(db_path, "T", datetime(2024, 6, 10, 4, 10, tzinfo=UTC), close=102.0)


def test_opening_range_position_pit(tmp_duckdb_path):
    """Poison bar exactly at AS_OF_OR does not affect the result."""
    _setup(tmp_duckdb_path)
    _seed_or_bars(tmp_duckdb_path)
    # as_of = 04:30 UTC; current price = last bar before 04:30 = 04:10 close = 102
    _ins_intraday(tmp_duckdb_path, "T", AS_OF_OR, close=99_999.0)  # poison (exactly at as_of)

    result = opening_range_position("T", AS_OF_OR, str(tmp_duckdb_path))

    # OR bars at 03:45–03:59 UTC; high of 03:49 = 96 + 4*2.5 = 106; low of 03:45 = 94
    # orh = 106, orl = 94; current = 102 (04:10 bar, before 04:30)
    orh, orl = 106.0, 94.0
    expected = (102.0 - orl) / (orh - orl)
    assert result == pytest.approx(expected, rel=1e-5)


def test_opening_range_position_before_or_complete_returns_none(tmp_duckdb_path):
    """as_of before 04:00 UTC (OR not yet complete) → None."""
    _setup(tmp_duckdb_path)
    _seed_or_bars(tmp_duckdb_path)
    as_of_early = datetime(2024, 6, 10, 3, 50, tzinfo=UTC)  # 9:20 IST — mid-OR
    assert opening_range_position("T", as_of_early, str(tmp_duckdb_path)) is None


def test_opening_range_position_correct(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    # Simple OR: orh=110, orl=90; current price=100 → position = 0.5
    _ins_intraday(tmp_duckdb_path, "T", datetime(2024, 6, 10, 3, 45, tzinfo=UTC),
                  close=90.0, high=110.0, low=90.0, volume=1000)
    _ins_intraday(tmp_duckdb_path, "T", datetime(2024, 6, 10, 4, 15, tzinfo=UTC), close=100.0)
    result = opening_range_position("T", AS_OF_OR, str(tmp_duckdb_path))
    assert result == pytest.approx(0.5)


# ── pcr_oi ────────────────────────────────────────────────────────────────────

def test_pcr_oi_pit(tmp_duckdb_path):
    """Poison FO row at AS_OF_DATE excluded; returns Jun 9 value."""
    _setup(tmp_duckdb_path)
    _ins_fo(tmp_duckdb_path, "T", date(2024, 6, 9), pcr_oi_val=1.25)
    _ins_fo(tmp_duckdb_path, "T", AS_OF_DATE, pcr_oi_val=99.0)  # poison

    result = pcr_oi("T", AS_OF, str(tmp_duckdb_path))
    assert result == pytest.approx(1.25)


def test_pcr_oi_correct(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    _ins_fo(tmp_duckdb_path, "T", date(2024, 6, 9), pcr_oi_val=0.85)
    assert pcr_oi("T", AS_OF, str(tmp_duckdb_path)) == pytest.approx(0.85)


def test_pcr_oi_empty_returns_none(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    assert pcr_oi("T", AS_OF, str(tmp_duckdb_path)) is None


# ── pcr_volume ────────────────────────────────────────────────────────────────

def test_pcr_volume_pit(tmp_duckdb_path):
    """Poison FO row at AS_OF_DATE excluded."""
    _setup(tmp_duckdb_path)
    _ins_fo(tmp_duckdb_path, "T", date(2024, 6, 9), pcr_vol_val=1.10)
    _ins_fo(tmp_duckdb_path, "T", AS_OF_DATE, pcr_vol_val=99.0)  # poison

    assert pcr_volume("T", AS_OF, str(tmp_duckdb_path)) == pytest.approx(1.10)


def test_pcr_volume_correct(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    _ins_fo(tmp_duckdb_path, "T", date(2024, 6, 9), pcr_vol_val=0.75)
    assert pcr_volume("T", AS_OF, str(tmp_duckdb_path)) == pytest.approx(0.75)


# ── net_oi_change ─────────────────────────────────────────────────────────────

def test_net_oi_change_pit(tmp_duckdb_path):
    """Poison FO row at AS_OF_DATE excluded; result from Jun 9 data only."""
    _setup(tmp_duckdb_path)
    # Jun 9: net_oi_change=+100_000, total=2_000_000 → ratio = +0.05
    _ins_fo(tmp_duckdb_path, "T", date(2024, 6, 9),
            net_oi_chg=100_000, total_ce_oi=1_000_000, total_pe_oi=1_000_000)
    _ins_fo(tmp_duckdb_path, "T", AS_OF_DATE,
            net_oi_chg=-99_999_999)  # poison

    result = net_oi_change("T", AS_OF, str(tmp_duckdb_path))
    assert result == pytest.approx(100_000 / 2_000_000)


def test_net_oi_change_correct(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    _ins_fo(tmp_duckdb_path, "T", date(2024, 6, 9),
            net_oi_chg=-200_000, total_ce_oi=2_000_000, total_pe_oi=2_000_000)
    result = net_oi_change("T", AS_OF, str(tmp_duckdb_path))
    assert result == pytest.approx(-200_000 / 4_000_000)


# ── max_pain_distance ─────────────────────────────────────────────────────────

def test_max_pain_distance_pit(tmp_duckdb_path):
    """Poison FO row at AS_OF_DATE excluded; spot from Jun 9 ohlcv, pain from Jun 9 FO."""
    _setup(tmp_duckdb_path)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=1000.0)
    _ins_fo(tmp_duckdb_path, "T", date(2024, 6, 9), max_pain=950.0)
    _ins_fo(tmp_duckdb_path, "T", AS_OF_DATE, max_pain=99_999.0)  # poison

    result = max_pain_distance("T", AS_OF, str(tmp_duckdb_path))
    # (1000 - 950) / 1000 = 0.05
    assert result == pytest.approx(0.05)


def test_max_pain_distance_correct(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=500.0)
    _ins_fo(tmp_duckdb_path, "T", date(2024, 6, 9), max_pain=510.0)
    result = max_pain_distance("T", AS_OF, str(tmp_duckdb_path))
    # (500 - 510) / 500 = -0.02
    assert result == pytest.approx(-0.02)


# ── spot_futures_basis ────────────────────────────────────────────────────────

def test_spot_futures_basis_pit(tmp_duckdb_path):
    """Poison FO row at AS_OF_DATE excluded; basis from Jun 9 data."""
    _setup(tmp_duckdb_path)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=1000.0)
    _ins_fo(tmp_duckdb_path, "T", date(2024, 6, 9), futures_settle=1010.0)
    _ins_fo(tmp_duckdb_path, "T", AS_OF_DATE, futures_settle=99_999.0)  # poison

    result = spot_futures_basis("T", AS_OF, str(tmp_duckdb_path))
    # (1010 - 1000) / 1000 = 0.01
    assert result == pytest.approx(0.01)


def test_spot_futures_basis_correct(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=200.0)
    _ins_fo(tmp_duckdb_path, "T", date(2024, 6, 9), futures_settle=198.0)
    result = spot_futures_basis("T", AS_OF, str(tmp_duckdb_path))
    assert result == pytest.approx((198.0 - 200.0) / 200.0)


# ── news_count_1d ─────────────────────────────────────────────────────────────

def test_news_count_1d_pit(tmp_duckdb_path):
    """News item at exactly AS_OF is NOT counted (strict < as_of)."""
    _setup(tmp_duckdb_path)
    # 2 items inside lookback window (< AS_OF, >= AS_OF - 24h)
    _ins_news(tmp_duckdb_path, "TEST", AS_OF - timedelta(hours=1), "n1")
    _ins_news(tmp_duckdb_path, "TEST", AS_OF - timedelta(hours=12), "n2")
    # Poison item exactly at AS_OF
    _ins_news(tmp_duckdb_path, "TEST", AS_OF, "n_poison")

    result = news_count_1d("TEST", AS_OF, str(tmp_duckdb_path))
    assert result == 2.0


def test_news_count_1d_lookback_boundary(tmp_duckdb_path):
    """Item exactly 24h before AS_OF IS included (>= cutoff)."""
    _setup(tmp_duckdb_path)
    _ins_news(tmp_duckdb_path, "TEST", AS_OF - timedelta(hours=24), "n_boundary")
    result = news_count_1d("TEST", AS_OF, str(tmp_duckdb_path))
    assert result == 1.0


def test_news_count_1d_empty_returns_zero(tmp_duckdb_path):
    _setup(tmp_duckdb_path)
    assert news_count_1d("TEST", AS_OF, str(tmp_duckdb_path)) == 0.0


# ── news_count_5d ─────────────────────────────────────────────────────────────

def test_news_count_5d_pit(tmp_duckdb_path):
    """News item at exactly AS_OF is NOT counted."""
    _setup(tmp_duckdb_path)
    _ins_news(tmp_duckdb_path, "TEST", AS_OF - timedelta(days=2), "n1")
    _ins_news(tmp_duckdb_path, "TEST", AS_OF - timedelta(days=4), "n2")
    _ins_news(tmp_duckdb_path, "TEST", AS_OF, "n_poison")  # exactly at AS_OF

    result = news_count_5d("TEST", AS_OF, str(tmp_duckdb_path))
    assert result == 2.0


def test_news_count_5d_older_item_excluded(tmp_duckdb_path):
    """Item older than 5 days is NOT included."""
    _setup(tmp_duckdb_path)
    _ins_news(tmp_duckdb_path, "TEST", AS_OF - timedelta(days=3), "n_in")
    _ins_news(tmp_duckdb_path, "TEST", AS_OF - timedelta(days=6), "n_out")

    result = news_count_5d("TEST", AS_OF, str(tmp_duckdb_path))
    assert result == 1.0


# ── FeatureRegistry meta-tests ────────────────────────────────────────────────

def test_registry_contains_all_15_features():
    """All 15 features are registered after importing src.features."""
    names = {m.name for m in FeatureRegistry.list_features()}
    expected = {
        "return_1d", "return_5d", "vol_20d", "volume_ratio_20d", "atr_20d",
        "vwap_deviation", "intraday_momentum", "opening_range_position",
        "pcr_oi", "pcr_volume", "net_oi_change", "max_pain_distance", "spot_futures_basis",
        "news_count_1d", "news_count_5d",
    }
    assert expected.issubset(names), f"Missing: {expected - names}"


def test_registry_gate_flags_correct():
    none_features = {m.name for m in FeatureRegistry.list_features() if m.gate == DataGate.NONE}
    intra_features = {m.name for m in FeatureRegistry.list_features() if m.gate == DataGate.INTRADAY}
    fo_features = {m.name for m in FeatureRegistry.list_features() if m.gate == DataGate.FO}
    news_features = {m.name for m in FeatureRegistry.list_features() if m.gate == DataGate.NEWS}

    assert none_features == {"return_1d", "return_5d", "vol_20d", "volume_ratio_20d", "atr_20d"}
    assert intra_features == {"vwap_deviation", "intraday_momentum", "opening_range_position"}
    assert fo_features == {"pcr_oi", "pcr_volume", "net_oi_change", "max_pain_distance", "spot_futures_basis"}
    assert news_features == {"news_count_1d", "news_count_5d"}


def test_compute_all_gates_none_always_run(tmp_duckdb_path):
    """Gate=NONE features always run regardless of gates_open."""
    _setup(tmp_duckdb_path)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 8), close=100.0)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=102.0)

    # Pass empty gates_open — only NONE features should run
    results = FeatureRegistry.compute_all(
        "T", AS_OF, str(tmp_duckdb_path), gates_open=set()
    )
    # NONE features run and have values
    assert results["return_1d"] == pytest.approx(0.02)
    # Gated features return None
    assert results["pcr_oi"] is None
    assert results["vwap_deviation"] is None
    assert results["news_count_1d"] is None


def test_compute_features_convenience_wrapper(tmp_duckdb_path):
    """compute_features() delegates to FeatureRegistry.compute_all correctly."""
    _setup(tmp_duckdb_path)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 8), close=100.0)
    _ins_daily(tmp_duckdb_path, "T", date(2024, 6, 9), close=103.0)

    results = compute_features("T", AS_OF, str(tmp_duckdb_path),
                               gates_open={DataGate.NONE})
    assert "return_1d" in results
    assert results["return_1d"] == pytest.approx(0.03)


# ── No-GPU invariant (extended to cover src/features/) ───────────────────────

def test_no_gpu_imports_in_features_layer():
    """Non-sentiment files under src/features/ must not import GPU packages.

    sentiment.py is excluded: it imports torch lazily (inside a function)
    with CUDA availability detection. Its device invariant is verified
    separately in test_sentiment.py::test_sentiment_never_hardcodes_cuda_device.
    All other feature files must be GPU-import-free.
    """
    gpu_packages = {"torch", "tensorflow", "cupy", "jax"}
    features_dir = Path(__file__).parent.parent / "src" / "features"
    violations = []
    for py_file in features_dir.glob("*.py"):
        if py_file.name == "sentiment.py":
            continue  # torch use is lazy + device-detected; see test_sentiment.py
        source = py_file.read_text(encoding="utf-8")
        for pkg in gpu_packages:
            if f"import {pkg}" in source or f"from {pkg}" in source:
                violations.append(f"{py_file.name}: imports {pkg}")
    assert not violations, "GPU imports found in src/features/: " + "; ".join(violations)
