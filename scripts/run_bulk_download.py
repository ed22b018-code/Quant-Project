"""CLI wrapper — run the Dhan bulk download independently of the full orchestration.

Use this when you want to (re-)download data without re-running training.
The download is resumable: already-stored date ranges are skipped.

Usage
-----
    python -m scripts.run_bulk_download
    python -m scripts.run_bulk_download --symbols RELIANCE TCS INFY --start 2019-01-01
    python -m scripts.run_bulk_download --fo-only          # only F&O history
    python -m scripts.run_bulk_download --coverage-only   # just print coverage report
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta

try:
    from dotenv import load_dotenv  # type: ignore[import-untyped]
    load_dotenv()
except ImportError:
    pass


_DEFAULT_SYMBOLS = [
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
    "SBIN", "WIPRO", "AXISBANK", "LT", "HINDUNILVR",
]

_DEFAULT_TIMEFRAMES = ["1"]


def main(argv: list[str] | None = None) -> int:
    from src.config import load_config
    config = load_config()
    
    parser = argparse.ArgumentParser(
        description="Dhan bulk download — intraday OHLCV + F&O history."
    )
    parser.add_argument(
        "--db-path", default="data/market.duckdb",
        help="DuckDB path (default: data/market.duckdb)",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=config.universe.symbols,
        help="NSE tickers to download (space-separated, defaults to all from config.toml)",
    )
    parser.add_argument(
        "--timeframes", nargs="+", default=_DEFAULT_TIMEFRAMES,
        help="Intraday timeframe codes (default: 1 5 15 25 60)",
    )
    parser.add_argument(
        "--start", default=None,
        help="Start date YYYY-MM-DD (default: 3 years ago)",
    )
    parser.add_argument(
        "--end", default=None,
        help="End date YYYY-MM-DD (default: today)",
    )
    parser.add_argument(
        "--fo-only", action="store_true",
        help="Only download F&O history (skip intraday OHLCV)",
    )
    parser.add_argument(
        "--intraday-only", action="store_true",
        help="Only download intraday OHLCV (skip F&O)",
    )
    parser.add_argument(
        "--coverage-only", action="store_true",
        help="Print coverage report and exit without downloading",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    end_date = date.fromisoformat(args.end) if args.end else date.today()
    start_date = (
        date.fromisoformat(args.start)
        if args.start
        else end_date - timedelta(days=365 * 3)
    )

    from src.data.dhan_downloader import DhanBulkDownloader  # noqa: PLC0415
    dl = DhanBulkDownloader(db_path=args.db_path)

    if args.coverage_only:
        print(dl.coverage_report())
        return 0

    if not args.fo_only:
        print(f"Downloading intraday OHLCV: {args.symbols} × {args.timeframes}")
        print(f"  Date range: {start_date} -> {end_date}")
        dl.run(
            symbols=args.symbols,
            timeframes=args.timeframes,
            start_date=start_date,
            end_date=end_date,
        )

    if not args.intraday_only:
        print(f"Downloading F&O history: {args.symbols}")
        dl.run_fo(
            symbols=args.symbols,
            start_date=start_date,
            end_date=end_date,
        )
        print("Computing F&O daily metrics...")
        dl.compute_fo_metrics(symbols=args.symbols)

    print("\nCoverage report:")
    print(dl.coverage_report())
    return 0


if __name__ == "__main__":
    sys.exit(main())
