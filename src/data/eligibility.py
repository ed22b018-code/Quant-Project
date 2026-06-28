"""PIT symbol-eligibility reconstruction from NSE bhavcopy.

A symbol is eligible on date D iff it appeared in the bhavcopy loaded for D.
This is the only correct way to reconstruct the tradeable universe without
look-ahead bias — never use a static symbol list for evaluation.

Survivorship is documentable: the dated bhavcopy files permanently record
which symbols traded on each date, including stocks later delisted.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

from src.db.duckdb_client import DuckDBClient


def get_eligible_symbols(as_of: date, db_path: Path) -> frozenset[str]:
    """Return the set of symbols that appeared in the NSE bhavcopy for `as_of`.

    A symbol is eligible on date D iff at least one bhavcopy row for that
    exact date was loaded into DuckDB.  If no data was loaded for `as_of`
    (e.g. a holiday whose file was not fetched), the result is an empty set —
    which is correct: you couldn't have traded on that day.

    Parameters
    ----------
    as_of   : The date for which to reconstruct the universe.
    db_path : Path to the DuckDB database containing ohlcv_daily.

    Returns
    -------
    frozenset[str] — symbols eligible on `as_of`.  Empty set if no data.

    Notes
    -----
    Do NOT query with ``date <= as_of`` — that would include symbols from
    earlier dates and would NOT exclude stocks that were already delisted by
    `as_of`.  The correct universe is exactly the set that traded on that date.
    """
    with DuckDBClient(db_path) as db:
        df = db.query_df(
            "SELECT DISTINCT symbol FROM ohlcv_daily WHERE date = ?",
            (as_of,),
        )
    return frozenset(df["symbol"].tolist())
