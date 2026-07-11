"""tools/build_universe.py — derive the training universe from the database.

Applies three principled eligibility rules to ohlcv_intraday and reports
symbols in three buckets:

  INCLUDED                  — passes all rules; ready for training
  ELIGIBLE-PENDING-REDOWNLOAD — known NSE F&O constituent; excluded only due to
                               download failure on the 3060; re-download to include
  EXCLUDED-BY-RULE          — does not meet data-coverage criteria

Eligibility rules
-----------------
Rule 1  symbol must have rows in ohlcv_intraday with timeframe='1'
Rule 2  MAX(ts)::DATE >= '2026-01-01'   (data covers the sealed test period)
Rule 3  COUNT(DISTINCT CAST(ts AS DATE)) >= 600   (~80% of the 3-year window)

Usage (from project root, with venv active)
-------------------------------------------
    python tools/build_universe.py
    python tools/build_universe.py --db path/to/other.duckdb
    python tools/build_universe.py --output-toml   # print config.toml-ready list

Re-run this script whenever:
  - New symbols are downloaded (to pick them up automatically)
  - A re-download fixes a failed symbol (to promote it from PENDING to INCLUDED)
  - The evaluation window changes (adjust MIN_LAST_DATE / MIN_TRADING_DAYS below)
Then paste the --output-toml block into config.toml and commit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Eligibility thresholds (edit here if the window changes) ─────────────────
MIN_LAST_DATE   = "2026-01-01"   # Rule 2: symbol must have data covering test period
MIN_TRADING_DAYS = 600           # Rule 3: ~80% of the 3-year window

# ── Known NSE F&O constituents that failed download on the 3060 ──────────────
# These are legitimate eligible stocks; excluded only due to API/instrument-master
# issues during the bulk download. Re-download each on the 3060 to include them.
# Target universe size = len(INCLUDED) + len(ELIGIBLE_PENDING_REDOWNLOAD) = ~114.
ELIGIBLE_PENDING_REDOWNLOAD: dict[str, str] = {
    "TATAMOTORS": (
        "all retries exhausted on 3060 — instrument-master security_id issue; "
        "re-download after refreshing instrument master CSV"
    ),
    "ZOMATO": (
        "all retries exhausted on 3060 — instrument-master security_id issue; "
        "re-download after refreshing instrument master CSV"
    ),
    "LTIM": (
        "all retries exhausted on 3060 — instrument-master security_id issue; "
        "LTIMindtree (listed 2022, formerly L&T Infotech + Mindtree merger); "
        "re-download after refreshing instrument master CSV"
    ),
    "CHOLAFIN": (
        "API returned empty data on 3060 — security_id lookup mismatch "
        "(Cholamandalam Financial Holdings); check instrument master for correct entry"
    ),
    "MCDOWELL-N": (
        "all retries exhausted — hyphenated ticker 'MCDOWELL-N' likely causes "
        "instrument-master lookup failure; verify Dhan instrument master CSV entry "
        "before re-downloading"
    ),
}


def _query_db(db_path: str) -> list[dict]:
    """Return per-symbol 1-min stats from ohlcv_intraday."""
    try:
        import duckdb
    except ImportError:
        sys.exit("duckdb not installed — activate the project venv first")

    con = duckdb.connect(db_path, read_only=True)
    rows = con.execute("""
        SELECT
            symbol,
            COUNT(*)                             AS rows_1m,
            CAST(MIN(ts) AS DATE)                AS first_date,
            CAST(MAX(ts) AS DATE)                AS last_date,
            COUNT(DISTINCT CAST(ts AS DATE))     AS trading_days
        FROM ohlcv_intraday
        WHERE timeframe = '1'
        GROUP BY symbol
        ORDER BY symbol
    """).fetchall()
    con.close()
    return [
        {
            "symbol":       r[0],
            "rows":         r[1],
            "first":        str(r[2]),
            "last":         str(r[3]),
            "trading_days": r[4],
        }
        for r in rows
    ]


def _classify(stats: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into (included, excluded_by_rule) based on eligibility rules."""
    included, excluded = [], []
    for s in stats:
        passes_r2 = s["last"] >= MIN_LAST_DATE
        passes_r3 = s["trading_days"] >= MIN_TRADING_DAYS
        if passes_r2 and passes_r3:
            included.append(s)
        else:
            reasons = []
            if not passes_r2:
                reasons.append(
                    f"last bar {s['last']} < {MIN_LAST_DATE} (missing test-period data)"
                )
            if not passes_r3:
                reasons.append(
                    f"only {s['trading_days']} trading days < {MIN_TRADING_DAYS} threshold"
                )
            excluded.append({**s, "exclusion_reason": "; ".join(reasons)})
    return included, excluded


def _print_report(
    included: list[dict],
    excluded: list[dict],
    ref_days: int,
) -> None:
    W = 90

    def header(title: str) -> str:
        return f"\n{'=' * W}\n  {title}\n{'=' * W}"

    print(header(f"BUCKET 1 — INCLUDED  ({len(included)} symbols)"))
    print(f"  All pass Rule 1 (has data), Rule 2 (last ≥ {MIN_LAST_DATE}), "
          f"Rule 3 (days ≥ {MIN_TRADING_DAYS})")
    print(f"\n  {'Symbol':<18} {'rows':>10} {'first':>12} {'last':>12} "
          f"{'days':>6} {'cov%':>7}")
    print(f"  {'-' * 72}")
    for s in included:
        cov = s["trading_days"] / ref_days * 100
        print(f"  {s['symbol']:<18} {s['rows']:>10,} {s['first']:>12} "
              f"{s['last']:>12} {s['trading_days']:>6,} {cov:>6.1f}%")

    print(header(
        f"BUCKET 2 — ELIGIBLE, PENDING RE-DOWNLOAD  "
        f"({len(ELIGIBLE_PENDING_REDOWNLOAD)} symbols)"
    ))
    print("  Valid NSE F&O constituents. Excluded ONLY due to download failure on 3060.")
    print("  Re-download each to expand to the ~114-symbol target universe.\n")
    for sym, note in ELIGIBLE_PENDING_REDOWNLOAD.items():
        print(f"  {sym:<18}  {note}")

    print(header(f"BUCKET 3 — EXCLUDED BY RULE  ({len(excluded)} symbols)"))
    print("  Do not pass the eligibility rules with current data.\n")
    for s in excluded:
        print(f"  {s['symbol']:<18}  rows={s['rows']:,}  days={s['trading_days']}  "
              f"last={s['last']}")
        print(f"    Reason: {s['exclusion_reason']}")

    print(f"\n{'=' * W}")
    print(f"  Summary:  INCLUDED={len(included)}  "
          f"ELIGIBLE-PENDING={len(ELIGIBLE_PENDING_REDOWNLOAD)}  "
          f"EXCLUDED-BY-RULE={len(excluded)}")
    print(f"  Interim universe: {len(included)} symbols  "
          f"|  Target (after re-downloads): "
          f"~{len(included) + len(ELIGIBLE_PENDING_REDOWNLOAD)} symbols")
    print("=" * W)


def _print_toml(included: list[dict]) -> None:
    """Print a config.toml-ready symbols array."""
    syms = sorted(s["symbol"] for s in included)
    print("\n# --- paste into config.toml [universe] section ---")
    print("symbols = [")
    for i in range(0, len(syms), 5):
        chunk = syms[i : i + 5]
        line = "  " + ", ".join(f'"{s}"' for s in chunk) + ","
        print(line)
    print("]")
    print(f"# {len(syms)} symbols derived on {_today()} "
          f"(Rule 2 ≥ {MIN_LAST_DATE}, Rule 3 ≥ {MIN_TRADING_DAYS} days)")


def _today() -> str:
    from datetime import date
    return str(date.today())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--db", default="data/market.duckdb",
        help="Path to DuckDB database (default: data/market.duckdb)",
    )
    parser.add_argument(
        "--output-toml", action="store_true",
        help="Print a config.toml-ready symbols list in addition to the report",
    )
    args = parser.parse_args()

    db_path = str(Path(args.db))
    if not Path(db_path).exists():
        sys.exit(f"Database not found: {db_path}")

    stats = _query_db(db_path)
    if not stats:
        sys.exit("No 1-min rows found in ohlcv_intraday — is this the right database?")

    # Reference trading days: use the symbol with the most distinct dates
    ref_days = max(s["trading_days"] for s in stats)

    included, excluded = _classify(stats)
    _print_report(included, excluded, ref_days)

    if args.output_toml:
        _print_toml(included)


if __name__ == "__main__":
    main()
