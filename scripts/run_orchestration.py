"""CLI entry point — run the full end-to-end pipeline (fetch → train → save).

This is the SINGLE COMMAND for Step 5 in the Dhan deployment runbook:
    python -m scripts.run_orchestration

The pipeline is RESUMABLE: completed stages are recorded in
    {output_dir}/.orchestration_state.json
If a stage fails, fix the issue and re-run — completed stages are skipped.

To force a re-run from a specific stage:
    python -m scripts.run_orchestration --start-from bulk_download

Stage list (in order)
---------------------
  1. preflight       — env vars, disk space, connectivity probe
  2. test_fetch      — tiny API probe (1 symbol, 3 days)
  3. bulk_download   — full 5yr intraday + F&O history
  4. coverage_check  — print coverage matrix
  5. build_dataset   — assemble features + labels (PIT)
  6. train_models    — LightGBM quantile cone + scenario classifier
  7. calibrate       — isotonic calibration on last validation fold
  8. save_summary    — write provenance.json + print metrics

Credential note
---------------
Credentials are read from .env — NEVER committed to git.
Each machine (dev laptop AND RTX 3060) needs its own copy of .env.
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


from src.config import load_config
_config = load_config()
_DEFAULT_SYMBOLS = _config.universe.symbols


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: preflight → fetch → train → save.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db-path", default="data/market.duckdb",
        help="DuckDB path (default: data/market.duckdb)",
    )
    parser.add_argument(
        "--output-dir", default="models/v1",
        help="Output directory for models and provenance (default: models/v1)",
    )
    parser.add_argument(
        "--symbols", nargs="+", default=_DEFAULT_SYMBOLS,
        help="NSE tickers to include (space-separated)",
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
        "--dry-run", action="store_true",
        help="Validate wiring without real API calls or training",
    )
    parser.add_argument(
        "--start-from",
        default=None,
        metavar="STAGE",
        help=(
            "Ignore checkpoint from this stage onwards. "
            "Valid values: preflight, test_fetch, bulk_download, coverage_check, "
            "build_dataset, train_models, calibrate, save_summary"
        ),
    )
    parser.add_argument(
        "--skip-test-fetch", action="store_true",
        help="Skip the test_fetch stage (when connection was already verified)",
    )
    parser.add_argument(
        "--skip-fo", action="store_true",
        help=(
            "Skip F&O download (recommended for first run). "
            "The Dhan /charts/historical endpoint returns OHLCV only — "
            "OI data needed for PCR/max-pain is absent. "
            "F&O gate features will be NaN."
        ),
    )
    parser.add_argument(
        "--skip-news", action="store_true",
        help="Skip News features. Will prevent the dataset builder from removing all rows due to NaNs.",
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

    from src.orchestration.orchestrator import Orchestrator, Stage  # noqa: PLC0415

    start_from: Stage | None = None
    if args.start_from:
        try:
            start_from = Stage(args.start_from.lower())
        except ValueError:
            valid = ", ".join(s.value for s in Stage)
            print(f"Error: unknown stage '{args.start_from}'. Valid stages: {valid}")
            return 2

    orch = Orchestrator(
        db_path=args.db_path,
        output_dir=args.output_dir,
        universe=args.symbols,
        start_date=start_date,
        end_date=end_date,
        skip_fo=args.skip_fo,
        skip_news=args.skip_news,
    )

    try:
        summary = orch.run(
            dry_run=args.dry_run,
            start_from=start_from,
            skip_test_fetch=args.skip_test_fetch,
        )
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        return 1

    print(f"\n  Stages run     : {summary.get('stages_run', [])}")
    print(f"  Stages skipped : {summary.get('stages_skipped', [])}")
    print(f"  Output dir     : {summary.get('output_dir')}")
    if summary.get("provenance_path"):
        print(f"  Provenance     : {summary['provenance_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
