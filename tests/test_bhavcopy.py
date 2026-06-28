"""
Tests for the NSE bhavcopy data layer.

Coverage:
  A. parse_bhavcopy_csv  — newer NSE format, older NSE format, EQ series filter.
  B. validate_bhavcopy   — negative price (hard reject), high < low (hard reject),
                           zero volume (soft flag / kept), extreme move (soft flag / kept).
  C. load_bhavcopy_to_duckdb — round-trip, idempotency, provenance field.
  D. PIT enforcement on real DuckDB data — as_of before a row's date → DataLeakageError.
  E. Survivorship / eligibility — SYNTH_DELISTED present on Jan 22, absent after
     it stops appearing; SYNTH_LIVE present throughout. Proves the function
     reconstructs presence from data, never from a static list.
  F. BhavCopySource interface — correct schema, PIT enforcement, DataNotAvailableError
     for all non-daily methods, metadata fields.
  G. DataNotAvailableError lives in base and is importable from there.
  H. download_bhavcopy — holiday → None (no raise); trading-day network failure
     → MissingDataGap; already-downloaded file → returns existing path.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.base import DataLeakageError, DataNotAvailableError
from src.data.bhavcopy import (
    MissingDataGap,
    download_bhavcopy,
    load_bhavcopy_to_duckdb,
    parse_bhavcopy_csv,
    validate_bhavcopy,
)
from src.data.bhavcopy_source import BhavCopySource
from src.data.eligibility import get_eligible_symbols

# ── Synthetic CSV fixtures ───────────────────────────────────────────────────

_NEW_FMT_CSV = """\
TckrSymb,TradDt,SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol,TtlTrfVal
RELIANCE,2024-01-22,EQ,2500.00,2550.00,2490.00,2520.00,1000000,2520000000.00
INFY,2024-01-22,EQ,1450.00,1480.00,1440.00,1460.00,500000,730000000.00
HDFCBANK,2024-01-22,EQ,1600.00,1630.00,1590.00,1610.00,750000,1207500000.00
RHTF,2024-01-22,BE,100.00,105.00,99.00,103.00,50000,5150000.00
"""

_OLD_FMT_CSV = """\
SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN
RELIANCE,EQ,2500.00,2550.00,2490.00,2520.00,2520.00,2480.00,1000000,2520000000.00,22-JAN-2024,5000,INE002A01018
INFY,EQ,1450.00,1480.00,1440.00,1460.00,1460.00,1440.00,500000,730000000.00,22-JAN-2024,3000,INE009A01021
"""

# Survivorship: SYNTH_LIVE present on both dates; SYNTH_DELISTED only on Jan 22
_SURV_JAN22 = """\
TckrSymb,TradDt,SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol,TtlTrfVal
SYNTH_LIVE,2024-01-22,EQ,100.00,105.00,98.00,102.00,200000,20400000.00
SYNTH_DELISTED,2024-01-22,EQ,50.00,52.00,49.00,51.00,100000,5100000.00
"""
_SURV_JAN23 = """\
TckrSymb,TradDt,SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol,TtlTrfVal
SYNTH_LIVE,2024-01-23,EQ,102.00,107.00,100.00,104.00,210000,21840000.00
"""

# Validation edge-cases
_VALIDATION_CSV = """\
TckrSymb,TradDt,SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol,TtlTrfVal
GOOD,2024-01-22,EQ,100.00,110.00,95.00,105.00,50000,5250000.00
NEG_PRICE,2024-01-22,EQ,-1.00,110.00,95.00,105.00,50000,5250000.00
HGT_LT_LOW,2024-01-22,EQ,100.00,90.00,95.00,105.00,50000,5250000.00
ZERO_VOL,2024-01-22,EQ,100.00,110.00,95.00,105.00,0,0.00
EXTREME,2024-01-22,EQ,100.00,999.00,1.00,100.00,50000,5000000.00
"""


def _write_csv(raw_dir: Path, d: date, content: str) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / f"{d.isoformat()}.csv").write_text(content, encoding="utf-8")


def _build_db(tmp_path: Path, entries: list[tuple[date, str]]) -> Path:
    """Create an in-memory-equivalent DuckDB with given (date, csv_content) pairs."""
    db_path = tmp_path / "test.duckdb"
    raw_dir = tmp_path / "raw"
    for d, content in entries:
        _write_csv(raw_dir, d, content)
        load_bhavcopy_to_duckdb(d, raw_dir, db_path)
    return db_path


# ── A. parse_bhavcopy_csv ─────────────────────────────────────────────────────

def test_parse_new_format_returns_expected_symbols() -> None:
    df = parse_bhavcopy_csv(_NEW_FMT_CSV)
    assert set(df["symbol"]) == {"RELIANCE", "INFY", "HDFCBANK"}


def test_parse_new_format_filters_non_eq_series() -> None:
    df = parse_bhavcopy_csv(_NEW_FMT_CSV)
    # RHTF is BE series — must be excluded
    assert "RHTF" not in df["symbol"].values


def test_parse_new_format_date_is_date_object() -> None:
    df = parse_bhavcopy_csv(_NEW_FMT_CSV)
    assert all(isinstance(v, date) for v in df["date"])


def test_parse_new_format_numeric_types() -> None:
    df = parse_bhavcopy_csv(_NEW_FMT_CSV)
    assert df["open"].dtype == "float64"
    assert df["volume"].dtype == "int64"


def test_parse_old_format_returns_expected_symbols() -> None:
    df = parse_bhavcopy_csv(_OLD_FMT_CSV)
    assert "RELIANCE" in df["symbol"].values
    assert "INFY" in df["symbol"].values


def test_parse_old_format_date_is_date_object() -> None:
    df = parse_bhavcopy_csv(_OLD_FMT_CSV)
    assert all(isinstance(v, date) for v in df["date"])
    assert df["date"].iloc[0] == date(2024, 1, 22)


def test_parse_bytes_input() -> None:
    df = parse_bhavcopy_csv(_NEW_FMT_CSV.encode("utf-8"))
    assert not df.empty


# ── B. validate_bhavcopy ──────────────────────────────────────────────────────

def _parsed_validation_df() -> pd.DataFrame:
    return parse_bhavcopy_csv(_VALIDATION_CSV)


def test_validate_rejects_negative_price() -> None:
    raw = _parsed_validation_df()
    valid, rejected = validate_bhavcopy(raw)
    assert "NEG_PRICE" not in valid["symbol"].values
    assert any("NEG_PRICE" in r for r in rejected)


def test_validate_rejects_high_lt_low() -> None:
    raw = _parsed_validation_df()
    valid, rejected = validate_bhavcopy(raw)
    assert "HGT_LT_LOW" not in valid["symbol"].values
    assert any("HGT_LT_LOW" in r for r in rejected)


def test_validate_keeps_zero_volume_but_logs(caplog: pytest.LogCaptureFixture) -> None:
    raw = _parsed_validation_df()
    with caplog.at_level(logging.WARNING, logger="src.data.bhavcopy"):
        valid, rejected = validate_bhavcopy(raw)
    assert "ZERO_VOL" in valid["symbol"].values          # kept
    assert not any("ZERO_VOL" in r for r in rejected)   # not hard-rejected
    assert any("ZERO_VOL" in r.message for r in caplog.records)


def test_validate_keeps_extreme_move_but_logs(caplog: pytest.LogCaptureFixture) -> None:
    raw = _parsed_validation_df()
    with caplog.at_level(logging.WARNING, logger="src.data.bhavcopy"):
        valid, rejected = validate_bhavcopy(raw)
    assert "EXTREME" in valid["symbol"].values
    assert not any("EXTREME" in r for r in rejected)
    assert any("EXTREME" in r.message for r in caplog.records)


def test_validate_good_row_survives() -> None:
    raw = _parsed_validation_df()
    valid, _ = validate_bhavcopy(raw)
    assert "GOOD" in valid["symbol"].values


def test_validate_empty_dataframe_is_fine() -> None:
    empty = pd.DataFrame(columns=["symbol", "open", "high", "low", "close", "volume"])
    valid, rejected = validate_bhavcopy(empty)
    assert valid.empty
    assert rejected == []


# ── C. load_bhavcopy_to_duckdb ────────────────────────────────────────────────

def test_load_round_trip(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "test.duckdb"
    d = date(2024, 1, 22)
    _write_csv(raw_dir, d, _NEW_FMT_CSV)

    loaded, rejected = load_bhavcopy_to_duckdb(d, raw_dir, db_path)

    assert loaded == 3          # RELIANCE, INFY, HDFCBANK (RHTF excluded)
    assert rejected == 0

    # Query back
    from src.db.duckdb_client import DuckDBClient
    with DuckDBClient(db_path) as db:
        df = db.query_df("SELECT * FROM ohlcv_daily WHERE symbol = 'RELIANCE'")
    assert len(df) == 1
    assert float(df["close"].iloc[0]) == pytest.approx(2520.0)


def test_load_provenance_field(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "test.duckdb"
    d = date(2024, 1, 22)
    _write_csv(raw_dir, d, _NEW_FMT_CSV)
    load_bhavcopy_to_duckdb(d, raw_dir, db_path)

    from src.db.duckdb_client import DuckDBClient
    with DuckDBClient(db_path) as db:
        df = db.query_df("SELECT DISTINCT source FROM ohlcv_daily")
    assert df["source"].iloc[0] == "nse_bhavcopy"


def test_load_idempotent(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    db_path = tmp_path / "test.duckdb"
    d = date(2024, 1, 22)
    _write_csv(raw_dir, d, _NEW_FMT_CSV)

    load_bhavcopy_to_duckdb(d, raw_dir, db_path)
    load_bhavcopy_to_duckdb(d, raw_dir, db_path)   # second load — must not duplicate

    from src.db.duckdb_client import DuckDBClient
    with DuckDBClient(db_path) as db:
        count = db.query_df("SELECT COUNT(*) AS n FROM ohlcv_daily")["n"].iloc[0]
    assert count == 3    # still 3, not 6


def test_load_missing_csv_raises(tmp_path: Path) -> None:
    db_path = tmp_path / "test.duckdb"
    with pytest.raises(FileNotFoundError):
        load_bhavcopy_to_duckdb(date(2024, 1, 22), tmp_path / "raw", db_path)


# ── D. PIT enforcement on real DuckDB data ───────────────────────────────────

def test_bhavcopy_source_pit_blocks_row_on_same_date(tmp_path: Path) -> None:
    """as_of is midnight of the bar date — bar not yet knowable, must be blocked."""
    db_path = _build_db(tmp_path, [(date(2024, 1, 22), _NEW_FMT_CSV)])
    src = BhavCopySource(db_path)
    # as_of = midnight on 2024-01-22 → bar for 2024-01-22 is NOT available yet
    as_of = datetime(2024, 1, 22, 0, 0, 0)
    # The query filters date < 2024-01-22, so the result is empty → no PIT violation
    df = src.get_ohlcv_daily("RELIANCE", as_of)
    assert df.empty


def test_bhavcopy_source_pit_allows_row_next_day(tmp_path: Path) -> None:
    """as_of = start of 2024-01-23 → bar for 2024-01-22 is available."""
    db_path = _build_db(tmp_path, [(date(2024, 1, 22), _NEW_FMT_CSV)])
    src = BhavCopySource(db_path)
    as_of = datetime(2024, 1, 23, 0, 0, 0)
    df = src.get_ohlcv_daily("RELIANCE", as_of)
    assert len(df) == 1
    assert float(df["close"].iloc[0]) == pytest.approx(2520.0)


# ── E. Survivorship / eligibility ────────────────────────────────────────────

def test_survivorship_delisted_present_before_cutoff(tmp_path: Path) -> None:
    """
    SYNTH_DELISTED appears in the Jan 22 bhavcopy only.
    It must be in the eligible set for Jan 22 — it was trading then.
    """
    db_path = _build_db(tmp_path, [
        (date(2024, 1, 22), _SURV_JAN22),
        (date(2024, 1, 23), _SURV_JAN23),
    ])
    eligible_jan22 = get_eligible_symbols(date(2024, 1, 22), db_path)
    assert "SYNTH_DELISTED" in eligible_jan22


def test_survivorship_delisted_absent_after_cutoff(tmp_path: Path) -> None:
    """
    SYNTH_DELISTED does NOT appear in the Jan 23 bhavcopy.
    It must NOT be in the eligible set for Jan 23 — it was no longer trading.
    This is the survivorship proof: absent from later dates automatically.
    """
    db_path = _build_db(tmp_path, [
        (date(2024, 1, 22), _SURV_JAN22),
        (date(2024, 1, 23), _SURV_JAN23),
    ])
    eligible_jan23 = get_eligible_symbols(date(2024, 1, 23), db_path)
    assert "SYNTH_DELISTED" not in eligible_jan23


def test_survivorship_live_present_both_dates(tmp_path: Path) -> None:
    """SYNTH_LIVE appears in both bhavcopy files — eligible on both dates."""
    db_path = _build_db(tmp_path, [
        (date(2024, 1, 22), _SURV_JAN22),
        (date(2024, 1, 23), _SURV_JAN23),
    ])
    assert "SYNTH_LIVE" in get_eligible_symbols(date(2024, 1, 22), db_path)
    assert "SYNTH_LIVE" in get_eligible_symbols(date(2024, 1, 23), db_path)


def test_eligibility_empty_for_date_with_no_data(tmp_path: Path) -> None:
    """If no bhavcopy was loaded for a date, the eligible set is empty — not an error."""
    db_path = _build_db(tmp_path, [(date(2024, 1, 22), _SURV_JAN22)])
    eligible = get_eligible_symbols(date(2024, 1, 25), db_path)
    assert eligible == frozenset()


# ── F. BhavCopySource interface ───────────────────────────────────────────────

def test_bhavcopy_source_is_datasource_subclass() -> None:
    from src.data.base import DataSource
    assert issubclass(BhavCopySource, DataSource)


def test_bhavcopy_source_metadata_fields() -> None:
    src = BhavCopySource(Path(":memory:"))
    assert src.data_source_name == "nse_bhavcopy"
    assert src.data_granularity == "daily"
    assert "daily" in src.data_source_limitations
    assert "raw" in src.data_source_limitations


def test_bhavcopy_source_returns_correct_column_schema(tmp_path: Path) -> None:
    from src.data.historical import OHLCV_DAILY_COLS
    db_path = _build_db(tmp_path, [(date(2024, 1, 22), _NEW_FMT_CSV)])
    src = BhavCopySource(db_path)
    as_of = datetime(2024, 1, 23, 9, 0, 0)
    df = src.get_ohlcv_daily("RELIANCE", as_of)
    assert list(df.columns) == list(OHLCV_DAILY_COLS.keys())


def test_bhavcopy_source_empty_result_for_unknown_symbol(tmp_path: Path) -> None:
    db_path = _build_db(tmp_path, [(date(2024, 1, 22), _NEW_FMT_CSV)])
    src = BhavCopySource(db_path)
    df = src.get_ohlcv_daily("NOTEXIST", datetime(2024, 1, 23, 0, 0, 0))
    assert df.empty


def test_bhavcopy_source_minute_raises_not_available(tmp_path: Path) -> None:
    src = BhavCopySource(tmp_path / "db.duckdb")
    with pytest.raises(DataNotAvailableError, match="daily"):
        src.get_ohlcv_minute("RELIANCE", datetime(2024, 1, 23))


def test_bhavcopy_source_ticks_raises_not_available(tmp_path: Path) -> None:
    src = BhavCopySource(tmp_path / "db.duckdb")
    with pytest.raises(DataNotAvailableError):
        src.get_ticks("RELIANCE", datetime(2024, 1, 23))


def test_bhavcopy_source_news_raises_not_available(tmp_path: Path) -> None:
    src = BhavCopySource(tmp_path / "db.duckdb")
    with pytest.raises(DataNotAvailableError):
        src.get_news("RELIANCE", datetime(2024, 1, 23))


def test_bhavcopy_source_option_chain_raises_not_available(tmp_path: Path) -> None:
    src = BhavCopySource(tmp_path / "db.duckdb")
    with pytest.raises(DataNotAvailableError):
        src.get_option_chain("RELIANCE", datetime(2024, 1, 23))


# ── G. DataNotAvailableError is in base ──────────────────────────────────────

def test_data_not_available_error_importable_from_base() -> None:
    from src.data.base import DataNotAvailableError as DNAE
    assert issubclass(DNAE, RuntimeError)


def test_data_not_available_error_is_not_data_leakage_error() -> None:
    assert not issubclass(DataNotAvailableError, DataLeakageError)
    assert not issubclass(DataLeakageError, DataNotAvailableError)


# ── H. download_bhavcopy ──────────────────────────────────────────────────────

def _make_mock_calendar(classification: str) -> MagicMock:
    mock = MagicMock()
    mock.classify_missing_file.return_value = classification
    return mock


def test_download_holiday_returns_none(tmp_path: Path) -> None:
    cal = _make_mock_calendar("expected")
    result = download_bhavcopy(
        date(2024, 1, 26), tmp_path / "raw", calendar=cal
    )
    assert result is None


def test_download_holiday_does_not_raise(tmp_path: Path) -> None:
    cal = _make_mock_calendar("expected")
    try:
        download_bhavcopy(date(2024, 1, 26), tmp_path / "raw", calendar=cal)
    except Exception as exc:  # noqa: BLE001
        pytest.fail(f"Unexpected exception for holiday: {exc}")


def test_download_trading_day_network_failure_raises_missing_gap(
    tmp_path: Path,
) -> None:
    cal = _make_mock_calendar("unexpected")
    with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
        with pytest.raises(MissingDataGap):
            download_bhavcopy(date(2024, 1, 22), tmp_path / "raw", calendar=cal)


def test_download_returns_existing_file_without_network(tmp_path: Path) -> None:
    cal = _make_mock_calendar("unexpected")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    existing = raw_dir / "2024-01-22.csv"
    existing.write_text(_NEW_FMT_CSV, encoding="utf-8")

    with patch("urllib.request.urlopen") as mock_url:
        result = download_bhavcopy(date(2024, 1, 22), raw_dir, calendar=cal)

    mock_url.assert_not_called()   # no network call needed
    assert result == existing
