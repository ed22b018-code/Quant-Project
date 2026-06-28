"""Tests for Checkpoint 2.3 — Triple-barrier labels, purge+embargo, DatasetBuilder.

KEY TESTS (explicitly requested)
---------------------------------
test_purge_removes_overlapping_label_window
    A training sample whose label window (t0=Jan 5, t1=Jan 10) overlaps the
    validation start (Jan 8) is purged. Structural proof: no t1 crosses the
    split boundary after purge_and_embargo().

test_embargo_removes_near_boundary
    After purge, samples too close to the val boundary are removed by the
    embargo rule. Proves the two-condition filter (t1 < val_start AND
    t0 < embargo_cutoff) works correctly.

test_dataset_row_no_future_data  ← FULL-PIPELINE PIT LEAKAGE TEST
    A DuckDB database is seeded with correct data before T and a poison row
    AT T (date = 2024-06-10, which equals as_of.date()). DatasetBuilder.build()
    assembles a row at T. The assembled feature value must match correct data
    only — proving the end-to-end pipeline respects strict PIT (<, not <=).

Other tests
-----------
Triple-barrier label correctness, WalkForwardSplitter time-ordering and
sealed-test-set enforcement, provenance columns.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.db.schema import DUCKDB_SCHEMA_SQL
from src.db.duckdb_client import DuckDBClient
from src.features import DataGate
from src.models.labels import Label, TripleBarrierLabeler, purge_and_embargo
from src.models.dataset import DatasetBuilder, WalkForwardSplitter

UTC = timezone.utc

# ── DB helpers (identical pattern to test_features.py) ───────────────────────

def _setup(db_path: Path) -> None:
    with DuckDBClient(db_path) as conn:
        conn.execute_schema(DUCKDB_SCHEMA_SQL)


def _ins_daily(
    db_path: Path,
    symbol: str,
    day: date,
    *,
    close: float,
    volume: int = 1_000_000,
) -> None:
    h = close * 1.01
    lo = close * 0.99
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            "INSERT INTO ohlcv_daily "
            "(symbol, date, open, high, low, close, volume, traded_value, "
            " series, delivery_qty, source, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            [symbol, day, close, h, lo, close, volume,
             float(volume * close), "EQ", None, "test", datetime.now(UTC)],
        )


# ── 1. Triple-barrier label correctness ──────────────────────────────────────

def test_upper_barrier_hit_first():
    """Price rises above profit target → BULLISH label."""
    prices = pd.Series([100.0, 101.0, 102.0, 103.0])
    ts = pd.Series(pd.date_range("2024-01-01", periods=4, freq="30min"))
    labeler = TripleBarrierLabeler(profit_pct=0.015, stop_pct=0.01, horizon_bars=5)
    df = labeler.label_series(prices, ts)

    assert df.iloc[0]["label"] == int(Label.BULLISH)
    assert df.iloc[0]["barrier"] == "profit"


def test_lower_barrier_hit_first():
    """Price falls below stop-loss → BEARISH label."""
    prices = pd.Series([100.0, 99.0, 98.0, 97.0])
    ts = pd.Series(pd.date_range("2024-01-01", periods=4, freq="30min"))
    labeler = TripleBarrierLabeler(profit_pct=0.05, stop_pct=0.02, horizon_bars=5)
    df = labeler.label_series(prices, ts)

    assert df.iloc[0]["label"] == int(Label.BEARISH)
    assert df.iloc[0]["barrier"] == "stop"


def test_time_barrier_no_trade():
    """Price stays flat — neither barrier hit → NO_TRADE label."""
    prices = pd.Series([100.0, 100.1, 99.9, 100.05])
    ts = pd.Series(pd.date_range("2024-01-01", periods=4, freq="30min"))
    # Very tight horizon so vertical fires before any meaningful move
    labeler = TripleBarrierLabeler(profit_pct=0.05, stop_pct=0.05, horizon_bars=2)
    df = labeler.label_series(prices, ts)

    assert df.iloc[0]["label"] == int(Label.NO_TRADE)
    assert df.iloc[0]["barrier"] == "time"


def test_no_trade_is_first_class_label():
    """NO_TRADE == 0, Label.NO_TRADE.label_str == 'no_trade'."""
    assert int(Label.NO_TRADE) == 0
    assert Label.NO_TRADE.label_str == "no_trade"
    assert Label.BULLISH.label_str == "bullish"
    assert Label.BEARISH.label_str == "bearish"


def test_upper_barrier_before_lower():
    """When price first crosses the upper barrier, label is BULLISH (not BEARISH)."""
    # Entry at 100; profit target = 102, stop = 98
    # Path: 100 → 95 (stop would fire) → 103 (but stop fires first)
    # Actually: path 100 → 103 → 97 — profit fires at bar 1
    prices = pd.Series([100.0, 103.0, 97.0, 90.0])
    ts = pd.Series(pd.date_range("2024-01-01", periods=4, freq="30min"))
    labeler = TripleBarrierLabeler(profit_pct=0.02, stop_pct=0.02, horizon_bars=5)
    df = labeler.label_series(prices, ts)

    assert df.iloc[0]["label"] == int(Label.BULLISH)
    assert df.iloc[0]["barrier"] == "profit"
    assert df.iloc[0]["t1"] == ts.iloc[1]


# ── 2. purge_and_embargo — KEY TESTS ─────────────────────────────────────────

def test_purge_removes_overlapping_label_window():
    """
    KEY TEST — label-window purge.

    Training samples:
        A: t0=Jan 5, t1=Jan 10 — label window OVERLAPS val_start (Jan 8) → PURGED
        B: t0=Jan 6, t1=Jan 7  — label window ends before val_start      → KEPT

    After purge_and_embargo(embargo_bars=0), only B remains.
    """
    train = pd.DataFrame({
        "t0": [date(2024, 1, 5), date(2024, 1, 6)],
        "t1": [date(2024, 1, 10), date(2024, 1, 7)],
        "label": [1, 0],
    })
    val = pd.DataFrame({"t0": [date(2024, 1, 8)], "t1": [date(2024, 1, 9)]})

    result = purge_and_embargo(train, val, embargo_bars=0)

    assert len(result) == 1, "Only the clean sample should survive purging"
    assert result.iloc[0]["t0"] == date(2024, 1, 6)


def test_embargo_removes_near_boundary():
    """
    KEY TEST — embargo removes samples too close to the validation boundary.

    val_start = Jan 10; embargo = 2 bars × 1 day = cutoff at Jan 8.

    Training samples:
        A: t0=Jan 5, t1=Jan 7  — t1<Jan10 ✓; t0=Jan5 < Jan8 ✓ → KEPT
        B: t0=Jan 8, t1=Jan 9  — t1<Jan10 ✓; t0=Jan8 NOT < Jan8 → EMBARGO-REMOVED
        C: t0=Jan 6, t1=Jan 11 — t1=Jan11 >= Jan10 → PURGED

    After purge_and_embargo(embargo_bars=2, bar=1 day), only A remains.
    """
    train = pd.DataFrame({
        "t0": [date(2024, 1, 5), date(2024, 1, 8), date(2024, 1, 6)],
        "t1": [date(2024, 1, 7), date(2024, 1, 9), date(2024, 1, 11)],
        "label": [1, 0, -1],
    })
    val = pd.DataFrame({"t0": [date(2024, 1, 10)]})

    result = purge_and_embargo(
        train, val, embargo_bars=2, bar_timedelta=timedelta(days=1)
    )

    assert len(result) == 1, "Only sample A should survive purge + embargo"
    assert result.iloc[0]["t0"] == date(2024, 1, 5)


def test_purge_keeps_clean_samples():
    """Samples with t1 well before val_start and t0 before embargo cutoff are kept."""
    train = pd.DataFrame({
        "t0": [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)],
        "t1": [date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 5)],
        "label": [1, 1, 0],
    })
    val = pd.DataFrame({"t0": [date(2024, 1, 15)]})  # val far away

    result = purge_and_embargo(train, val, embargo_bars=0)

    assert len(result) == 3  # all kept


def test_no_label_window_crosses_boundary():
    """
    Structural guarantee: after purge_and_embargo, ZERO training samples have
    t1 >= val_start. This is the invariant that prevents label-window leakage.
    """
    # Generate 20 training samples, some with overlapping label windows
    n = 20
    train = pd.DataFrame({
        "t0": [date(2024, 1, i + 1) for i in range(n)],
        "t1": [date(2024, 1, i + 5) for i in range(n)],  # 4-bar label windows
        "label": [1] * n,
    })
    val = pd.DataFrame({"t0": [date(2024, 1, 15)]})

    result = purge_and_embargo(train, val, embargo_bars=3, bar_timedelta=timedelta(days=1))

    if len(result) > 0:
        val_start = pd.Timestamp(val["t0"].min())
        t1_series = pd.to_datetime(result["t1"])
        assert (t1_series < val_start).all(), (
            "Label window leakage: some t1 >= val_start after purge+embargo"
        )


def test_purge_empty_val_returns_all_train():
    """Edge case: empty val DataFrame → all training samples returned unchanged."""
    train = pd.DataFrame({
        "t0": [date(2024, 1, 1)],
        "t1": [date(2024, 1, 5)],
        "label": [1],
    })
    val = pd.DataFrame({"t0": pd.Series([], dtype="object")})

    result = purge_and_embargo(train, val)
    assert len(result) == 1


# ── 3. DatasetBuilder — full-pipeline PIT leakage test ───────────────────────

def test_dataset_builder_builds_rows(tmp_path):
    """DatasetBuilder returns a DataFrame with expected columns for each timestamp."""
    db = tmp_path / "test.db"
    _setup(db)
    _ins_daily(db, "AAA", date(2024, 6, 8), close=100.0)
    _ins_daily(db, "AAA", date(2024, 6, 9), close=102.0)

    T = datetime(2024, 6, 10, 9, 30, tzinfo=UTC)
    builder = DatasetBuilder(feature_names=["return_1d"], gates_open={DataGate.NONE})
    df = builder.build("AAA", [T], str(db))

    assert "symbol" in df.columns
    assert "t0" in df.columns
    assert "return_1d" in df.columns
    assert df.iloc[0]["symbol"] == "AAA"
    assert df.iloc[0]["t0"] == T


def test_dataset_row_no_future_data(tmp_path):
    """
    FULL-PIPELINE PIT LEAKAGE TEST.

    Seed the DB with:
        Correct: 2024-06-08 close=100.0, 2024-06-09 close=102.0
        Poison:  2024-06-10 close=99999.0  ← exactly at as_of.date()

    Build a dataset row at T = 2024-06-10 09:30 UTC.

    The return_1d feature uses WHERE date < CAST(as_of AS DATE) = '2024-06-10',
    so only dates ≤ 2024-06-09 are visible. return_1d = (102-100)/100 = 0.02.

    If the poison row leaked, return_1d would be (99999-102)/102 ≈ 979 — the
    test would fail, exposing the leakage.
    """
    db = tmp_path / "pit_test.db"
    _setup(db)

    # Correct data — both strictly before 2024-06-10
    _ins_daily(db, "PIT", date(2024, 6, 8), close=100.0)
    _ins_daily(db, "PIT", date(2024, 6, 9), close=102.0)
    # Poison — exactly AT as_of.date(); must be excluded by PIT guard
    _ins_daily(db, "PIT", date(2024, 6, 10), close=99_999.0)

    T = datetime(2024, 6, 10, 9, 30, tzinfo=UTC)
    builder = DatasetBuilder(feature_names=["return_1d"], gates_open={DataGate.NONE})
    df = builder.build("PIT", [T], str(db))

    got = df.iloc[0]["return_1d"]
    assert abs(got - 0.02) < 1e-6, (
        f"PIT violation: return_1d = {got:.4f} but expected 0.0200. "
        "The poison row (date=2024-06-10) leaked into the feature computation."
    )


def test_dataset_provenance_gate_flags(tmp_path):
    """Gate-flag provenance columns reflect the gates_open setting."""
    db = tmp_path / "gf.db"
    _setup(db)
    _ins_daily(db, "GF", date(2024, 6, 9), close=100.0)
    _ins_daily(db, "GF", date(2024, 6, 8), close=99.0)

    T = datetime(2024, 6, 10, 9, 30, tzinfo=UTC)
    builder = DatasetBuilder(gates_open={DataGate.NONE})
    df = builder.build("GF", [T], str(db))

    assert not df.iloc[0]["gate_intraday_open"]
    assert not df.iloc[0]["gate_fo_open"]
    assert not df.iloc[0]["gate_news_open"]
    assert not df.iloc[0]["gate_news_nlp_open"]
    assert df.iloc[0]["forward_only_features"] == ""


# ── 4. WalkForwardSplitter ────────────────────────────────────────────────────

def _make_dataset(n: int = 100) -> pd.DataFrame:
    """Create a minimal synthetic dataset with t0 column."""
    return pd.DataFrame({
        "t0": pd.date_range("2024-01-01", periods=n, freq="30min"),
        "feat": range(n),
        "label": [0, 1, -1] * (n // 3) + [0] * (n % 3),
    })


def test_walk_forward_splits_time_ordered():
    """Every train fold ends before its paired val fold starts."""
    dataset = _make_dataset(90)
    splitter = WalkForwardSplitter(n_splits=5, final_test_fraction=0.10, t0_col="t0")
    splits = splitter.generate_splits(dataset)

    assert len(splits) > 0
    for train_fold, val_fold in splits:
        assert len(train_fold) > 0
        assert len(val_fold) > 0
        train_max_t0 = train_fold["t0"].max()
        val_min_t0 = val_fold["t0"].min()
        assert train_max_t0 < val_min_t0, (
            f"Train fold bleeds into val: train_max={train_max_t0}, val_min={val_min_t0}"
        )


def test_no_future_in_train_folds():
    """No train row has t0 >= the minimum t0 in its paired val fold."""
    dataset = _make_dataset(80)
    splitter = WalkForwardSplitter(n_splits=4, final_test_fraction=0.10)
    splits = splitter.generate_splits(dataset)

    for train_fold, val_fold in splits:
        val_start = val_fold["t0"].min()
        assert (train_fold["t0"] < val_start).all()


def test_final_test_set_sealed():
    """get_final_test_set() raises RuntimeError on a second call."""
    dataset = _make_dataset(50)
    splitter = WalkForwardSplitter(n_splits=3, final_test_fraction=0.10)

    test1 = splitter.get_final_test_set(dataset)
    assert len(test1) > 0

    with pytest.raises(RuntimeError, match="already been accessed"):
        splitter.get_final_test_set(dataset)


def test_final_test_set_not_in_splits():
    """No row from the final test set appears in any train or val fold."""
    dataset = _make_dataset(60)
    splitter = WalkForwardSplitter(n_splits=4, final_test_fraction=0.10)
    splits = splitter.generate_splits(dataset)

    # Reconstruct test set from a fresh splitter (same logic)
    splitter2 = WalkForwardSplitter(n_splits=4, final_test_fraction=0.10)
    test_set = splitter2.get_final_test_set(dataset)
    test_t0s = set(test_set["t0"].tolist())

    for train_fold, val_fold in splits:
        all_fold_t0s = set(train_fold["t0"].tolist()) | set(val_fold["t0"].tolist())
        overlap = test_t0s & all_fold_t0s
        assert len(overlap) == 0, f"Test rows leaked into folds: {overlap}"
