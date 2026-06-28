"""News count features — gate=NEWS.

Features:
    news_count_1d   Count of news items tagged with symbol in the last 24 hours
    news_count_5d   Count of news items tagged with symbol in the last 5 days

PIT guarantee
-------------
Queries use WHERE first_knowable_at < ? (TIMESTAMPTZ strict less-than).
A news item with first_knowable_at == as_of is NOT counted.

The news table uses first_knowable_at (not published_at) as the PIT column —
it is the most conservative timestamp at which the story could realistically
have been seen (published_at + ingestion-delay buffer per DESIGN.md §3).

Symbol matching
---------------
The news.symbols column is a VARCHAR[] (DuckDB array). list_contains() checks
whether the target symbol appears in the tagged list.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import duckdb

from src.features.base import DataGate, FeatureRegistry

UTC = timezone.utc


def _count_news(db_path: str, symbol: str, as_of: datetime, lookback_hours: float) -> float:
    as_of_utc = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
    cutoff = as_of_utc - timedelta(hours=lookback_hours)
    with duckdb.connect(db_path) as con:
        rows = con.execute(
            "SELECT COUNT(*) FROM news "
            "WHERE first_knowable_at < ? "
            "  AND first_knowable_at >= ? "
            "  AND list_contains(symbols, ?)",
            [as_of_utc, cutoff, symbol],
        ).fetchall()
    return float(rows[0][0]) if rows else 0.0


@FeatureRegistry.register("news_count_1d", DataGate.NEWS,
                           "Count of tagged news items in the 24 hours before as_of")
def news_count_1d(symbol: str, as_of: datetime, db_path: str) -> float | None:
    return _count_news(db_path, symbol, as_of, 24.0)


@FeatureRegistry.register("news_count_5d", DataGate.NEWS,
                           "Count of tagged news items in the 5 days before as_of")
def news_count_5d(symbol: str, as_of: datetime, db_path: str) -> float | None:
    return _count_news(db_path, symbol, as_of, 5 * 24.0)
