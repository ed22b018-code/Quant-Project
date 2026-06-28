"""Tests for Checkpoint 1.5.1 — corporate-action adjustment layer.

Covers:
  - Schema idempotency and columns
  - parse_action_description (split, bonus, dividend, unrecognized, edge cases)
  - compute_adjustment_factors (no actions, single split, compounding, PIT exclusion)
  - AdjustedBhavCopySource (returns continuity HEADLINE TEST, anchor invariant,
    pre/post split values, empty path, source label, PIT filtering)
  - BACKTEST_ONLY firewall (flag presence, live sources clean, structural proof)
  - CSV fallback round-trip
  - CorporateActionFetcher with mock response
"""
from __future__ import annotations

import csv
import logging
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from src.data.bhavcopy_source import BhavCopySource
from src.data.corporate_actions import (
    AdjustedBhavCopySource,
    CorporateActionFetcher,
    compute_adjustment_factors,
    load_corporate_actions_csv,
    parse_action_description,
)
from src.data.base import DataNotAvailableError, DataSource
from src.db.duckdb_client import DuckDBClient
from src.db.schema import DUCKDB_SCHEMA_SQL


# ── Test helpers ──────────────────────────────────────────────────────────────

_FIXED_TS = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)


def _insert_bars(db_path, symbol: str, bars: list[tuple]) -> None:
    """Insert (date, open, high, low, close, volume) tuples into ohlcv_daily."""
    with DuckDBClient(db_path) as db:
        db.execute_schema(DUCKDB_SCHEMA_SQL)
        for d, o, h, l, c, v in bars:
            db.conn.execute(
                """
                INSERT INTO ohlcv_daily
                    (symbol, date, open, high, low, close, volume,
                     traded_value, source, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, NULL, 'nse_bhavcopy', ?)
                ON CONFLICT DO NOTHING
                """,
                [symbol, d, o, h, l, c, v, _FIXED_TS],
            )


def _insert_action(
    db_path,
    symbol: str,
    ex_date: date,
    action_type: str,
    factor: float,
    description: str = "",
) -> None:
    """Insert one corporate-action row into the corporate_actions table."""
    with DuckDBClient(db_path) as db:
        db.execute_schema(DUCKDB_SCHEMA_SQL)
        db.conn.execute(
            """
            INSERT INTO corporate_actions
                (symbol, ex_date, action_type, adjustment_factor,
                 description, source, ingested_at)
            VALUES (?, ?, ?, ?, ?, 'test', ?)
            ON CONFLICT DO NOTHING
            """,
            [symbol, ex_date, action_type, factor, description or None, _FIXED_TS],
        )


# ── Schema tests ──────────────────────────────────────────────────────────────

def test_corporate_actions_table_created_idempotently(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    with DuckDBClient(db_path) as db:
        db.execute_schema(DUCKDB_SCHEMA_SQL)
        db.execute_schema(DUCKDB_SCHEMA_SQL)  # second call must not raise
        count = db.conn.execute(
            "SELECT COUNT(*) FROM corporate_actions"
        ).fetchone()[0]
    assert count == 0


def test_corporate_actions_columns_present(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    with DuckDBClient(db_path) as db:
        db.execute_schema(DUCKDB_SCHEMA_SQL)
        cols = {
            row[0]
            for row in db.conn.execute("DESCRIBE corporate_actions").fetchall()
        }
    required = {
        "symbol", "ex_date", "action_type", "adjustment_factor",
        "description", "source", "ingested_at",
    }
    assert required.issubset(cols)


# ── parse_action_description tests ───────────────────────────────────────────

def test_parse_split_10_to_2():
    action_type, factor = parse_action_description(
        "STOCK SPLIT FROM Rs 10/- TO Rs 2/- PER SHARE"
    )
    assert action_type == "SPLIT"
    assert factor == pytest.approx(0.2)


def test_parse_split_10_to_5():
    action_type, factor = parse_action_description("SPLIT FROM Rs 10 TO Rs 5")
    assert action_type == "SPLIT"
    assert factor == pytest.approx(0.5)


def test_parse_face_value_split():
    action_type, factor = parse_action_description(
        "FACE VALUE SPLIT FROM RS.10/- TO RS.1/-"
    )
    assert action_type == "SPLIT"
    assert factor == pytest.approx(0.1)


def test_parse_bonus_1_1():
    action_type, factor = parse_action_description("BONUS 1:1")
    assert action_type == "BONUS"
    assert factor == pytest.approx(0.5)


def test_parse_bonus_1_2():
    # 1 bonus share for every 2 existing → factor = 2/(2+1)
    action_type, factor = parse_action_description("BONUS 1:2")
    assert action_type == "BONUS"
    assert factor == pytest.approx(2.0 / 3.0)


def test_parse_dividend_returns_1_0():
    action_type, factor = parse_action_description(
        "INTERIM DIVIDEND Rs 5/- Per Share"
    )
    assert action_type == "DIVIDEND"
    assert factor == pytest.approx(1.0)


def test_parse_final_dividend_returns_1_0():
    action_type, factor = parse_action_description(
        "FINAL DIVIDEND - Rs.10/- Per Share"
    )
    assert action_type == "DIVIDEND"
    assert factor == pytest.approx(1.0)


def test_parse_unrecognized_returns_other_1_0_and_logs_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="src.data.corporate_actions"):
        action_type, factor = parse_action_description("AMALGAMATION")
    assert action_type == "OTHER"
    assert factor == pytest.approx(1.0)
    # A WARNING must have been emitted — caller is notified that no adjustment was applied
    assert any(
        "OTHER" in r.message or "unrecognized" in r.message.lower()
        for r in caplog.records
    )


def test_parse_invalid_split_ratio_returns_other():
    # new_FV > old_FV is invalid for a regular split; should not produce a factor > 1
    action_type, factor = parse_action_description("SPLIT FROM Rs 2 TO Rs 10")
    assert action_type == "OTHER"
    assert factor == pytest.approx(1.0)


def test_parse_empty_string_returns_other():
    action_type, factor = parse_action_description("")
    assert action_type == "OTHER"
    assert factor == pytest.approx(1.0)


def test_parse_buyback_returns_other():
    action_type, factor = parse_action_description("BUY BACK OF SHARES")
    assert action_type == "OTHER"
    assert factor == pytest.approx(1.0)


# ── compute_adjustment_factors tests ─────────────────────────────────────────

def test_no_actions_returns_all_ones():
    actions = pd.DataFrame(columns=["ex_date", "adjustment_factor"])
    price_dates = pd.Series([date(2024, 1, 1), date(2024, 1, 2)])
    factors = compute_adjustment_factors(
        actions, price_dates, datetime(2024, 2, 1, tzinfo=timezone.utc)
    )
    assert list(factors) == pytest.approx([1.0, 1.0])


def test_single_split_applied_before_ex_date_not_on():
    # Split on 2024-01-10 with factor=0.5
    actions = pd.DataFrame({
        "ex_date":           [date(2024, 1, 10)],
        "adjustment_factor": [0.5],
    })
    price_dates = pd.Series([
        date(2024, 1, 8),   # before ex_date → factor = 0.5
        date(2024, 1, 10),  # on ex_date     → factor = 1.0 (already post-split)
        date(2024, 1, 12),  # after ex_date  → factor = 1.0
    ])
    factors = compute_adjustment_factors(
        actions, price_dates, datetime(2024, 2, 1, tzinfo=timezone.utc)
    )
    assert factors.iloc[0] == pytest.approx(0.5)
    assert factors.iloc[1] == pytest.approx(1.0)
    assert factors.iloc[2] == pytest.approx(1.0)


def test_two_actions_compound():
    # Two 1:1 bonuses (each factor=0.5): 2024-01-10 and 2024-07-01
    actions = pd.DataFrame({
        "ex_date":           [date(2024, 1, 10), date(2024, 7, 1)],
        "adjustment_factor": [0.5,               0.5],
    })
    as_of = datetime(2025, 1, 1, tzinfo=timezone.utc)
    price_dates = pd.Series([
        date(2024, 1, 5),   # before first   → 0.5 × 0.5 = 0.25
        date(2024, 3, 1),   # between the two → only second applies → 0.5
        date(2024, 8, 1),   # after both      → 1.0
    ])
    factors = compute_adjustment_factors(actions, price_dates, as_of)
    assert factors.iloc[0] == pytest.approx(0.25)
    assert factors.iloc[1] == pytest.approx(0.5)
    assert factors.iloc[2] == pytest.approx(1.0)


def test_pit_future_action_excluded():
    # Action on 2025-06-01 is AFTER as_of 2024-12-31 → must NOT be applied
    actions = pd.DataFrame({
        "ex_date":           [date(2025, 6, 1)],
        "adjustment_factor": [0.5],
    })
    price_dates = pd.Series([date(2024, 1, 1)])
    factors = compute_adjustment_factors(
        actions, price_dates, datetime(2024, 12, 31, tzinfo=timezone.utc)
    )
    # Factor must be 1.0 — the future split is not known at this as_of
    assert factors.iloc[0] == pytest.approx(1.0)


# ── AdjustedBhavCopySource tests ─────────────────────────────────────────────

def test_adjusted_returns_continuous_across_split(tmp_path):
    """HEADLINE TEST — the central correctness invariant.

    A 2:1 split on 2024-01-05 causes raw prices to drop ~50% at the ex-date.
    After backward adjustment the same price series shows a normal daily move
    (~+1%) across that boundary.  The most-recent price (backward anchor) must
    equal the raw price.
    """
    db_path = str(tmp_path / "test.duckdb")
    symbol = "TSPLIT"

    # Pre-split prices ~1000 range; post-split prices ~500 range (halved by 2:1 split)
    bars = [
        (date(2024, 1, 2), 980.0,  1005.0, 975.0,  1000.0, 10000),
        (date(2024, 1, 3), 1000.0, 1025.0, 995.0,  1020.0, 10000),
        (date(2024, 1, 4), 1020.0, 1045.0, 1015.0, 1040.0, 10000),  # day before split
        (date(2024, 1, 5), 515.0,  525.0,  510.0,  525.0,  20000),  # ex-date
        (date(2024, 1, 8), 520.0,  530.0,  515.0,  530.0,  20000),
        (date(2024, 1, 9), 525.0,  535.0,  520.0,  535.0,  20000),  # most recent
    ]
    _insert_bars(db_path, symbol, bars)
    _insert_action(
        db_path, symbol, date(2024, 1, 5), "SPLIT", 0.5,
        "SPLIT FROM Rs 10 TO Rs 5",
    )

    as_of = datetime(2024, 1, 10, 9, 0, tzinfo=timezone.utc)

    df_raw = BhavCopySource(db_path).get_ohlcv_daily(symbol, as_of)
    df_raw = df_raw.sort_values("date").reset_index(drop=True)

    df_adj = AdjustedBhavCopySource(db_path).get_ohlcv_daily(symbol, as_of)
    df_adj = df_adj.sort_values("date").reset_index(drop=True)

    assert len(df_adj) == 6, f"Expected 6 bars, got {len(df_adj)}"

    # Raw: close drops ~50% at split boundary (artefact)
    raw_return_at_split = df_raw["close"].iloc[3] / df_raw["close"].iloc[2] - 1
    assert raw_return_at_split < -0.4, (
        f"Raw return at split should be ~-50%, got {raw_return_at_split:.1%}"
    )

    # Adjusted: no artefact — return reflects actual market movement (~+1%)
    adj_return_at_split = df_adj["close"].iloc[3] / df_adj["close"].iloc[2] - 1
    assert -0.05 < adj_return_at_split < 0.05, (
        f"Adjusted return at split should be ~+1%, got {adj_return_at_split:.1%}"
    )

    # Backward-anchor invariant: most-recent adjusted price = most-recent raw price
    assert df_adj["close"].iloc[-1] == pytest.approx(df_raw["close"].iloc[-1]), (
        "Most-recent adjusted price must equal the raw price (backward anchor)"
    )

    # Source label
    assert (df_adj["source"] == "nse_bhavcopy_adjusted").all()


def test_adjusted_pre_split_bar_scaled(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    symbol = "TADJ"
    bars = [
        (date(2024, 1, 2), 800.0, 810.0, 795.0, 1000.0, 10000),  # pre-split
        (date(2024, 1, 5), 490.0, 505.0, 485.0, 500.0,  20000),  # ex-date
    ]
    _insert_bars(db_path, symbol, bars)
    _insert_action(db_path, symbol, date(2024, 1, 5), "SPLIT", 0.5)

    as_of = datetime(2024, 1, 6, 0, 0, tzinfo=timezone.utc)
    df = AdjustedBhavCopySource(db_path).get_ohlcv_daily(symbol, as_of)
    df = df.sort_values("date").reset_index(drop=True)

    # Pre-split close 1000 × factor 0.5 = 500
    assert df.loc[df["date"] == date(2024, 1, 2), "close"].iloc[0] == pytest.approx(500.0)
    # Ex-date close 500 × factor 1.0 = 500 (bar is on/after ex_date)
    assert df.loc[df["date"] == date(2024, 1, 5), "close"].iloc[0] == pytest.approx(500.0)


def test_adjusted_most_recent_price_unchanged(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    symbol = "TANCH"
    bars = [
        (date(2024, 1, 2), 990.0, 1010.0, 985.0, 1000.0, 10000),
        (date(2024, 1, 3), 995.0, 1015.0, 990.0, 1010.0, 10000),  # most recent
    ]
    _insert_bars(db_path, symbol, bars)
    # Split ex-date is on 2024-01-03 — most-recent bar should not be adjusted
    _insert_action(db_path, symbol, date(2024, 1, 3), "SPLIT", 0.5)

    as_of = datetime(2024, 1, 4, 0, 0, tzinfo=timezone.utc)
    df_raw = BhavCopySource(db_path).get_ohlcv_daily(symbol, as_of)
    df_adj = AdjustedBhavCopySource(db_path).get_ohlcv_daily(symbol, as_of)

    df_raw = df_raw.sort_values("date").reset_index(drop=True)
    df_adj = df_adj.sort_values("date").reset_index(drop=True)

    # Most-recent bar = ex-date bar → on/after ex_date → factor 1.0 → unchanged
    assert df_adj["close"].iloc[-1] == pytest.approx(df_raw["close"].iloc[-1])


def test_adjusted_no_actions_prices_identical_to_raw(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    symbol = "TNOCA"
    bars = [
        (date(2024, 1, 2), 100.0, 105.0, 99.0, 102.0, 500),
        (date(2024, 1, 3), 102.0, 107.0, 101.0, 104.0, 500),
    ]
    _insert_bars(db_path, symbol, bars)
    # No corporate actions loaded

    as_of = datetime(2024, 1, 4, 0, 0, tzinfo=timezone.utc)
    df_raw = BhavCopySource(db_path).get_ohlcv_daily(symbol, as_of)
    df_adj = AdjustedBhavCopySource(db_path).get_ohlcv_daily(symbol, as_of)

    df_raw = df_raw.sort_values("date").reset_index(drop=True)
    df_adj = df_adj.sort_values("date").reset_index(drop=True)

    pd.testing.assert_series_equal(
        df_raw["close"].rename("close"),
        df_adj["close"].rename("close"),
        rtol=1e-4,
        check_names=False,
    )


def test_adjusted_empty_when_no_raw_data(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    as_of = datetime(2024, 1, 4, 0, 0, tzinfo=timezone.utc)
    df = AdjustedBhavCopySource(db_path).get_ohlcv_daily("NODATA", as_of)
    assert df.empty


def test_adjusted_only_returns_bars_before_as_of(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    symbol = "TPIT"
    bars = [
        (date(2024, 1, 2), 100.0, 105.0, 99.0, 102.0, 500),
        (date(2024, 1, 3), 102.0, 107.0, 101.0, 104.0, 500),
        (date(2024, 1, 4), 104.0, 109.0, 103.0, 106.0, 500),
    ]
    _insert_bars(db_path, symbol, bars)

    # as_of = 2024-01-03: only bar on 2024-01-02 should be returned (date < as_of.date())
    as_of = datetime(2024, 1, 3, 0, 0, tzinfo=timezone.utc)
    df = AdjustedBhavCopySource(db_path).get_ohlcv_daily(symbol, as_of)

    assert len(df) == 1
    assert df["date"].iloc[0] == date(2024, 1, 2)


def test_raw_source_prices_unchanged_after_loading_actions(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    symbol = "TRAW"
    bars = [
        (date(2024, 1, 2), 1000.0, 1010.0, 995.0, 1000.0, 10000),
    ]
    _insert_bars(db_path, symbol, bars)
    _insert_action(db_path, symbol, date(2024, 1, 5), "SPLIT", 0.5)

    as_of = datetime(2024, 1, 6, 0, 0, tzinfo=timezone.utc)
    df_raw = BhavCopySource(db_path).get_ohlcv_daily(symbol, as_of)

    # Raw source must return the original unadjusted price regardless of loaded actions
    assert df_raw["close"].iloc[0] == pytest.approx(1000.0)


def test_adjusted_pit_excludes_future_split(tmp_path):
    """A split that occurs after as_of must NOT be applied to historical bars."""
    db_path = str(tmp_path / "test.duckdb")
    symbol = "TFUTURE"
    bars = [
        (date(2023, 6, 1), 1000.0, 1010.0, 995.0, 1000.0, 10000),
    ]
    _insert_bars(db_path, symbol, bars)
    # Split in 2024 — future relative to as_of below
    _insert_action(db_path, symbol, date(2024, 6, 1), "SPLIT", 0.5)

    # as_of = end of 2023: the 2024 split is unknown here
    as_of = datetime(2023, 12, 31, tzinfo=timezone.utc)
    df_adj = AdjustedBhavCopySource(db_path).get_ohlcv_daily(symbol, as_of)

    # Bar should be at the raw price (split not applied)
    assert df_adj["close"].iloc[0] == pytest.approx(1000.0)

    # as_of = end of 2024: the 2024 split IS now known
    as_of_after = datetime(2024, 12, 31, tzinfo=timezone.utc)
    df_adj_after = AdjustedBhavCopySource(db_path).get_ohlcv_daily(symbol, as_of_after)

    # Bar should be adjusted (split now applies)
    assert df_adj_after["close"].iloc[0] == pytest.approx(500.0)


# ── BACKTEST_ONLY firewall tests ──────────────────────────────────────────────

def test_backtest_only_flag_on_adjusted_source():
    assert AdjustedBhavCopySource.BACKTEST_ONLY is True


def test_live_sources_not_backtest_only():
    from src.data.live_upstox import UpstoxLiveSource
    from src.data.news_ingest import NewsSource
    from src.data.option_chain import OptionChainSource

    for cls in [BhavCopySource, UpstoxLiveSource, NewsSource, OptionChainSource]:
        assert not getattr(cls, "BACKTEST_ONLY", False), (
            f"{cls.__name__} must not carry BACKTEST_ONLY=True"
        )


def test_live_feature_path_structural_enforcement():
    """Structural proof: a simulated live feature function can discriminate
    AdjustedBhavCopySource from live-safe sources via the BACKTEST_ONLY flag.

    This test proves the flag is present and correctly valued so that any live
    code checking it will refuse the adjusted source.
    """

    def simulate_live_feature_check(source: DataSource) -> None:
        if getattr(source, "BACKTEST_ONLY", False):
            raise RuntimeError(
                f"Source {source.data_source_name!r} is backtest-only and must not "
                "be used in live feature computation."
            )

    raw_src = BhavCopySource(":memory:")
    adj_src = AdjustedBhavCopySource(":memory:")

    # Live-safe source passes the check
    simulate_live_feature_check(raw_src)

    # Adjusted source is correctly rejected
    with pytest.raises(RuntimeError, match="backtest-only"):
        simulate_live_feature_check(adj_src)


def test_adjusted_data_source_name_distinct():
    assert AdjustedBhavCopySource.data_source_name != BhavCopySource.data_source_name
    assert AdjustedBhavCopySource.data_source_name == "nse_bhavcopy_adjusted"


def test_adjusted_limitations_contains_backtest_warning():
    lims = AdjustedBhavCopySource.data_source_limitations.upper()
    assert "BACKTEST" in lims


# ── CSV fallback test ─────────────────────────────────────────────────────────

def test_load_csv_fallback_round_trip(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    csv_path = str(tmp_path / "actions.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["symbol", "ex_date", "action_type", "adjustment_factor", "description"]
        )
        writer.writerow(["HDFC", "2019-09-17", "BONUS", "0.5", "BONUS 1:1"])
        writer.writerow(["TCS",  "2018-06-01", "SPLIT", "0.2", "SPLIT 10:2"])

    count = load_corporate_actions_csv(csv_path, db_path)
    assert count == 2

    # Second load: ON CONFLICT DO NOTHING → 0 new rows
    count2 = load_corporate_actions_csv(csv_path, db_path)
    assert count2 == 0

    with DuckDBClient(db_path) as db:
        db.execute_schema(DUCKDB_SCHEMA_SQL)
        rows = db.conn.execute(
            "SELECT symbol, action_type, adjustment_factor "
            "FROM corporate_actions ORDER BY symbol"
        ).fetchall()

    assert len(rows) == 2
    symbols = [r[0] for r in rows]
    assert "HDFC" in symbols
    assert "TCS" in symbols
    for row in rows:
        assert row[2] in (pytest.approx(0.5), pytest.approx(0.2))


# ── CorporateActionFetcher mock test ─────────────────────────────────────────

def test_fetcher_with_mock_response_inserts_action(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    mock = [
        {"symbol": "HDFC", "exDate": "17-09-2019", "subject": "BONUS 1:1"},
    ]
    fetcher = CorporateActionFetcher(_mock_response=mock)
    count = fetcher.fetch_and_store(
        "HDFC", date(2019, 1, 1), date(2019, 12, 31), db_path
    )
    assert count == 1

    with DuckDBClient(db_path) as db:
        db.execute_schema(DUCKDB_SCHEMA_SQL)
        rows = db.conn.execute(
            "SELECT symbol, action_type, adjustment_factor FROM corporate_actions"
        ).fetchall()

    assert len(rows) == 1
    assert rows[0][0] == "HDFC"
    assert rows[0][1] == "BONUS"
    assert rows[0][2] == pytest.approx(0.5)


def test_fetcher_ignores_wrong_symbol(tmp_path):
    db_path = str(tmp_path / "test.duckdb")
    mock = [
        {"symbol": "RELIANCE", "exDate": "17-09-2019", "subject": "BONUS 1:1"},
    ]
    fetcher = CorporateActionFetcher(_mock_response=mock)
    # Fetching for "HDFC" but mock only has "RELIANCE"
    count = fetcher.fetch_and_store(
        "HDFC", date(2019, 1, 1), date(2019, 12, 31), db_path
    )
    assert count == 0
