"""Small test-fetch — validates Dhan API connection and response format.

This is the FIRST real Dhan API call on each machine. Run it on BOTH
machines (dev laptop and RTX 3060) before launching the big download.
Surfaces auth and response-format surprises in seconds, not hours.

What it checks and prints
--------------------------
  - Authentication: 401 → key wrong / expired
  - Instrument master: security_id lookup for the symbol
  - Raw response: column names, row count, first 2 rows
  - Column validation: actual vs expected
  - Fix pointer: if columns differ, prints exactly where to fix the parser

If the response format differs from what the parser expects, the fix is in:
    src/data/dhan_downloader.py
    DhanBulkDownloader._parse_ohlcv_response()
    Line: timestamps = raw.get("timestamp", [])   ← key to verify

Usage
-----
    python -m scripts.test_fetch
    python -m scripts.test_fetch --symbol INFY --days 3 --timeframe 5

After verifying this works on both machines, proceed to:
    python -m scripts.run_orchestration
"""
from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from typing import Any, Callable, Optional

_LOG = logging.getLogger(__name__)

# Expected columns produced by _parse_ohlcv_response().
# If the real API returns different keys, the fix is in _parse_ohlcv_response().
EXPECTED_COLS: frozenset[str] = frozenset(["ts", "open", "high", "low", "close", "volume"])

# Where to fix a column mismatch. Printed in the output so the fix is found fast.
FIX_LOCATION = (
    "src/data/dhan_downloader.py\n"
    "    class DhanBulkDownloader\n"
    "    def _parse_ohlcv_response(self, raw, symbol, timeframe)\n"
    "    Look for:  timestamps = raw.get(\"timestamp\", [])  ← verify key name\n"
    "               opens      = raw.get(\"open\", [])       ← and all array keys"
)

MAX_PROBE_DAYS: int = 5  # test_fetch is a SMALL probe; reject large windows


def test_fetch(
    symbol: str = "RELIANCE",
    days: int = 3,
    timeframe: str = "1",
    db_path: Optional[str] = None,
    *,
    _mock_fetch: Optional[Callable] = None,
    _mock_instrument_master: Optional[dict] = None,
    _quiet: bool = False,
) -> dict:
    """Run a bounded test fetch and return a diagnostic result dict.

    Parameters
    ----------
    symbol : str
        NSE ticker to test with (default: RELIANCE).
    days : int
        Window size in calendar days (max 5 — this is a probe, not a download).
    timeframe : str
        Dhan intraday timeframe code (e.g. "1", "5", "15").
    db_path : str | None
        DuckDB path. Defaults to ":memory:" (no persistence).
    _mock_fetch : callable | None
        DhanBulkDownloader._mock_fetch hook for tests. When set, no real
        HTTP call is made.
    _mock_instrument_master : dict | None
        DhanInstrumentMaster._mock_master for tests.
    _quiet : bool
        Suppress stdout output (for tests).

    Returns
    -------
    dict with keys:
        status       : 'pass' | 'partial' | 'fail'
        symbol       : str
        days         : int
        rows         : int
        columns      : list[str]      — columns in the parsed DataFrame
        missing_cols : list[str]      — expected cols absent from parsed output
        fix_location : str | None     — where to fix if columns mismatch
        message      : str

    Raises
    ------
    ValueError if days > MAX_PROBE_DAYS (5).
    """
    if days > MAX_PROBE_DAYS:
        raise ValueError(
            f"test_fetch is a SMALL probe (max {MAX_PROBE_DAYS} days), got {days}. "
            f"Use scripts/run_bulk_download.py for large downloads."
        )

    from src.data.dhan_downloader import DhanBulkDownloader  # noqa: PLC0415

    # Dhan intraday data has a 2-4 calendar day publication lag.
    # Requesting today-3 always returns empty arrays even when auth is fine.
    # Shift the window back 5 days so the probe reliably finds published data.
    _LAG_BUFFER = 5
    end_date = date.today() - timedelta(days=_LAG_BUFFER)
    start_date = end_date - timedelta(days=days)

    _using_real_api = (_mock_fetch is None)

    if _using_real_api:
        # Real API: create downloader without mock master, then refresh to get
        # the real instrument master (security_id lookup via Dhan's public CSV).
        dl = DhanBulkDownloader(
            db_path=db_path or ":memory:",
            request_delay_seconds=0,
        )
        if not _quiet:
            print("  Refreshing instrument master (downloads Dhan CSV)...", end=" ", flush=True)
        dl._instrument_master.refresh()
        if not _quiet:
            print("done")
    else:
        # Mock / test path: use the provided mock instrument master
        dl = DhanBulkDownloader(
            db_path=db_path or ":memory:",
            request_delay_seconds=0,
            _mock_fetch=_mock_fetch,
            _mock_instrument_master=_mock_instrument_master or {(symbol, "EQUITY"): "TEST_ID"},
        )

    if not _quiet:
        print(f"\n===== Dhan Test Fetch =====")
        print(f"  Symbol    : {symbol}")
        print(f"  Timeframe : {timeframe} min")
        print(f"  Window    : {start_date} -> {end_date} ({days} days, shifted -{_LAG_BUFFER}d for publication lag)")
        print()

    df = dl._download_chunk(symbol, timeframe, start_date, end_date)

    if df is None or df.empty:
        result = {
            "status": "fail",
            "symbol": symbol,
            "days": days,
            "rows": 0,
            "columns": [],
            "missing_cols": sorted(EXPECTED_COLS),
            "fix_location": FIX_LOCATION,
            "message": (
                "API returned empty or None data. "
                "Check credentials (DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN) "
                "and that the symbol is valid."
            ),
        }
        if not _quiet:
            print(f"  [FAIL] {result['message']}")
        return result

    got_cols = frozenset(df.columns)
    missing_cols = EXPECTED_COLS - got_cols
    extra_cols = got_cols - EXPECTED_COLS

    if not _quiet:
        print("  Column check:")
        for col in sorted(EXPECTED_COLS):
            icon = "OK" if col in got_cols else "MISSING"
            print(f"    [{icon}] {col}")
        if extra_cols:
            print(f"  New columns (not in expected set): {sorted(extra_cols)}")
            print("    These may need handling in _parse_ohlcv_response().")
        print()
        print(f"  Row count : {len(df)}")
        print(f"  First row : {df.iloc[0].to_dict() if len(df) > 0 else 'N/A'}")
        if len(df) > 1:
            print(f"  Last row  : {df.iloc[-1].to_dict()}")

    if missing_cols:
        result = {
            "status": "partial",
            "symbol": symbol,
            "days": days,
            "rows": len(df),
            "columns": sorted(df.columns),
            "missing_cols": sorted(missing_cols),
            "fix_location": FIX_LOCATION,
            "message": (
                f"Column mismatch — parsed DataFrame is missing: {sorted(missing_cols)}. "
                f"The real API may use different key names than expected. "
                f"Fix in: {FIX_LOCATION.splitlines()[0]}"
            ),
        }
        if not _quiet:
            print()
            print(f"  [PARTIAL] Column mismatch detected.")
            print(f"  Missing expected columns: {sorted(missing_cols)}")
            print(f"  Fix location:")
            for line in FIX_LOCATION.splitlines():
                print(f"    {line}")
    else:
        result = {
            "status": "pass",
            "symbol": symbol,
            "days": days,
            "rows": len(df),
            "columns": sorted(df.columns),
            "missing_cols": [],
            "fix_location": None,
            "message": "All expected columns present.",
        }
        if not _quiet:
            print()
            print("  [PASS] Response format OK — all expected columns present.")
            print(f"  Ready to run: python -m scripts.run_orchestration")

    return result


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    import argparse  # noqa: PLC0415
    from dotenv import load_dotenv  # type: ignore[import-untyped]  # noqa: PLC0415

    try:
        load_dotenv()
    except ImportError:
        pass  # python-dotenv not required

    parser = argparse.ArgumentParser(
        description="Small Dhan test-fetch — validates connection and response format."
    )
    parser.add_argument("--symbol", default="RELIANCE", help="NSE ticker (default: RELIANCE)")
    parser.add_argument("--days", type=int, default=3, help="Window size in days (max 5)")
    parser.add_argument("--timeframe", default="1", help="Timeframe code (default: 1 min)")
    parser.add_argument("--db-path", default=None, help="DuckDB path (default: in-memory)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    result = test_fetch(
        symbol=args.symbol,
        days=args.days,
        timeframe=args.timeframe,
        db_path=args.db_path,
    )
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
