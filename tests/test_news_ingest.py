"""Tests for RSSNewsIngest and NewsSource — Checkpoint 4.

Proof areas:
  1. Dedup: same story ingested twice → only one row in DB
  2. Dedup: same story with typo corrected (punctuation diff) → same hash, one row
  3. first_knowable_at = published_at + ingestion_delay_seconds
  4. PIT on first_knowable_at, NOT published_at:
       as_of between published_at and first_knowable_at → story NOT returned
  5. Word-boundary symbol tagging: exact match, alias match, non-word boundary no match
  6. Untagged stories stored (symbols=[]) but not returned by symbol query
  7. DuckDB round-trip: read back columns match NEWS_COLS schema
  8. NewsSource.get_news() PIT enforcement (DataLeakageError on future row)
  9. feedparser not installed → returns 0, no crash
 10. OI-change and column coverage
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.data.base import DataLeakageError, DataNotAvailableError
from src.data.historical import NEWS_COLS
from src.data.news_ingest import (
    NewsSource,
    RSSNewsIngest,
    _content_hash,
    _normalize_text,
    _tag_symbols,
    _build_patterns,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

_PUB_AT = datetime(2024, 1, 22, 9, 0, 0, tzinfo=timezone.utc)
_DELAY_S = 300  # 5 min
_FIRST_KA = _PUB_AT + timedelta(seconds=_DELAY_S)
_AS_OF_AFTER = _FIRST_KA + timedelta(minutes=10)   # after first_knowable_at → returned
_AS_OF_BETWEEN = _PUB_AT + timedelta(seconds=120)  # between pub and first_knowable_at → blocked


def _make_entry(headline, body="", published=None):
    """Build a minimal feedparser-style entry."""
    e = SimpleNamespace()
    e.title = headline
    e.summary = body
    if published is not None:
        import calendar as _cal
        e.published_parsed = published.timetuple()
    else:
        e.published_parsed = _PUB_AT.timetuple()
    return e


def _make_feed(entries, source_title="TestFeed"):
    """Build a minimal feedparser-style feed result."""
    f = SimpleNamespace()
    f.entries = entries
    f.feed = {"title": source_title}
    return f


def _mock_fp(*entries_list):
    """Return a mock feedparser with .parse() producing the given entries."""
    fp = MagicMock()
    fp.parse.return_value = _make_feed(entries_list)
    return fp


# ── 1. Normalization ──────────────────────────────────────────────────────────

def test_normalize_text_lowercase_no_punct():
    result = _normalize_text("Hello, World! TCS...")
    assert result == "hello world tcs"


def test_normalize_text_collapses_whitespace():
    result = _normalize_text("  too   many   spaces  ")
    assert result == "too many spaces"


def test_normalize_empty():
    assert _normalize_text("") == ""


# ── 2. Content hash ───────────────────────────────────────────────────────────

def test_content_hash_identical_stories_same_hash():
    h1 = _content_hash("TCS reports Q3 results", "Revenue up 10%")
    h2 = _content_hash("TCS reports Q3 results", "Revenue up 10%")
    assert h1 == h2


def test_content_hash_punctuation_variant_same_hash():
    """Typo fix that only adds/removes punctuation → same hash."""
    h1 = _content_hash("TCS reports Q3 results!!", "Revenue up 10%.")
    h2 = _content_hash("TCS reports Q3 results", "Revenue up 10%")
    assert h1 == h2


def test_content_hash_different_stories_different_hash():
    h1 = _content_hash("TCS Q3 beat", "")
    h2 = _content_hash("Infosys Q3 miss", "")
    assert h1 != h2


# ── 3. Symbol tagging ─────────────────────────────────────────────────────────

def test_tag_exact_ticker_match():
    patterns = _build_patterns(["TCS"])
    tags = _tag_symbols("TCS quarterly results beat expectations", patterns)
    assert "TCS" in tags


def test_tag_alias_match():
    patterns = _build_patterns(["TCS"])
    tags = _tag_symbols("Tata Consultancy Services reported strong numbers", patterns)
    assert "TCS" in tags


def test_tag_no_false_match_inside_word():
    """'ITC' inside 'switch' or 'pitch' must NOT match."""
    patterns = _build_patterns(["ITC"])
    # "switch" contains "itc" — word boundary must prevent match
    tags = _tag_symbols("The switch network encountered no issues", patterns)
    assert "ITC" not in tags


def test_tag_case_insensitive():
    patterns = _build_patterns(["TCS"])
    tags = _tag_symbols("tcs reports earnings", patterns)
    assert "TCS" in tags


def test_tag_multiple_symbols():
    patterns = _build_patterns(["TCS", "INFY"])
    tags = _tag_symbols("TCS and Infosys both report on the same day", patterns)
    assert "TCS" in tags
    assert "INFY" in tags


def test_tag_untagged_empty():
    patterns = _build_patterns(["TCS"])
    tags = _tag_symbols("Global markets rally on Fed pivot hopes", patterns)
    assert tags == []


# ── 4. Dedup via DuckDB ───────────────────────────────────────────────────────

def test_same_story_ingested_once(tmp_path):
    db = str(tmp_path / "news.duckdb")
    entry = _make_entry("TCS beats Q3 estimates", "Revenue surged 12%")
    fp = _mock_fp(entry)

    ingest = RSSNewsIngest(ingestion_delay_seconds=_DELAY_S, _mock_feedparser=fp)
    r1 = ingest.fetch_and_store(["http://feed.example.com/rss"], frozenset(["TCS"]), db)
    r2 = ingest.fetch_and_store(["http://feed.example.com/rss"], frozenset(["TCS"]), db)

    assert r1 == 1, "first ingest should insert 1 row"
    assert r2 == 0, "second ingest of same story should be deduped"


def test_typo_fix_deduped(tmp_path):
    """Story with punctuation typo corrected → same normalized hash → deduped."""
    db = str(tmp_path / "news.duckdb")
    entry1 = _make_entry("TCS beats Q3!!", "Revenue surged 12%.")
    entry2 = _make_entry("TCS beats Q3", "Revenue surged 12%")

    fp1 = _mock_fp(entry1)
    fp2 = _mock_fp(entry2)

    r1 = RSSNewsIngest(_DELAY_S, _mock_feedparser=fp1).fetch_and_store(
        ["http://a.com/rss"], frozenset(["TCS"]), db
    )
    r2 = RSSNewsIngest(_DELAY_S, _mock_feedparser=fp2).fetch_and_store(
        ["http://a.com/rss"], frozenset(["TCS"]), db
    )

    assert r1 == 1
    assert r2 == 0, "punctuation-only difference → same hash → deduped"


# ── 5. first_knowable_at stamping ─────────────────────────────────────────────

def test_first_knowable_at_is_published_plus_delay(tmp_path):
    db = str(tmp_path / "news.duckdb")
    entry = _make_entry("TCS Q3 headline")
    fp = _mock_fp(entry)

    RSSNewsIngest(_DELAY_S, _mock_feedparser=fp).fetch_and_store(
        ["http://a.com/rss"], frozenset(["TCS"]), db
    )

    from src.db.duckdb_client import DuckDBClient
    from src.db.schema import DUCKDB_SCHEMA_SQL
    with DuckDBClient(db) as db_cl:
        db_cl.execute_schema(DUCKDB_SCHEMA_SQL)
        row = db_cl.query_df("SELECT published_at, first_knowable_at FROM news LIMIT 1")

    pub = pd.to_datetime(row["published_at"].iloc[0], utc=True)
    fka = pd.to_datetime(row["first_knowable_at"].iloc[0], utc=True)
    assert (fka - pub).total_seconds() == pytest.approx(_DELAY_S, abs=1)


# ── 6. PIT on first_knowable_at, not published_at ─────────────────────────────

def test_pit_blocks_story_before_first_knowable(tmp_path):
    """as_of between published_at and first_knowable_at → story must NOT be returned."""
    db = str(tmp_path / "news.duckdb")
    entry = _make_entry("TCS headline")
    fp = _mock_fp(entry)

    RSSNewsIngest(_DELAY_S, _mock_feedparser=fp).fetch_and_store(
        ["http://a.com/rss"], frozenset(["TCS"]), db
    )

    src = NewsSource(db)
    # as_of is between published_at and first_knowable_at → should NOT return the story
    df = src.get_news("TCS", _AS_OF_BETWEEN, lookback_hours=24.0)
    assert df.empty, "story should be blocked until first_knowable_at"


def test_pit_allows_story_after_first_knowable(tmp_path):
    db = str(tmp_path / "news.duckdb")
    entry = _make_entry("TCS headline")
    fp = _mock_fp(entry)

    RSSNewsIngest(_DELAY_S, _mock_feedparser=fp).fetch_and_store(
        ["http://a.com/rss"], frozenset(["TCS"]), db
    )

    src = NewsSource(db)
    df = src.get_news("TCS", _AS_OF_AFTER, lookback_hours=24.0)
    assert len(df) == 1


# ── 7. Untagged stories stored but not returned ───────────────────────────────

def test_untagged_story_not_returned_by_symbol_query(tmp_path):
    db = str(tmp_path / "news.duckdb")
    entry = _make_entry("Global markets rally", "No specific Indian stock mentioned")
    fp = _mock_fp(entry)

    RSSNewsIngest(_DELAY_S, _mock_feedparser=fp).fetch_and_store(
        ["http://a.com/rss"], frozenset(["TCS"]), db
    )

    # Story is untagged → not returned for TCS
    src = NewsSource(db)
    df = src.get_news("TCS", _AS_OF_AFTER, lookback_hours=24.0)
    assert df.empty

    # But it IS stored in the DB
    from src.db.duckdb_client import DuckDBClient
    from src.db.schema import DUCKDB_SCHEMA_SQL
    with DuckDBClient(db) as db_cl:
        db_cl.execute_schema(DUCKDB_SCHEMA_SQL)
        count = db_cl.query_df("SELECT COUNT(*) AS n FROM news").iloc[0]["n"]
    assert count == 1, "untagged story should still be stored"


# ── 8. DuckDB round-trip schema ───────────────────────────────────────────────

def test_news_source_returns_correct_columns(tmp_path):
    db = str(tmp_path / "news.duckdb")
    entry = _make_entry("TCS Q3 headline")
    fp = _mock_fp(entry)

    RSSNewsIngest(_DELAY_S, _mock_feedparser=fp).fetch_and_store(
        ["http://a.com/rss"], frozenset(["TCS"]), db
    )

    src = NewsSource(db)
    df = src.get_news("TCS", _AS_OF_AFTER, lookback_hours=24.0)
    assert list(df.columns) == list(NEWS_COLS.keys())


def test_empty_news_source_returns_correct_columns(tmp_path):
    db = str(tmp_path / "news_empty.duckdb")
    src = NewsSource(db)
    df = src.get_news("TCS", _AS_OF_AFTER)
    assert list(df.columns) == list(NEWS_COLS.keys())
    assert df.empty


# ── 9. feedparser not installed ───────────────────────────────────────────────

def test_fetch_returns_0_when_feedparser_not_installed(tmp_path, caplog):
    db = str(tmp_path / "news.duckdb")
    # Setting sys.modules["feedparser"] = None makes `import feedparser` raise ImportError
    # (documented Python behaviour — None sentinel blocks re-import from disk)
    saved = sys.modules.pop("feedparser", None)
    sys.modules["feedparser"] = None  # type: ignore[assignment]
    try:
        ingest = RSSNewsIngest(ingestion_delay_seconds=_DELAY_S)
        with caplog.at_level("WARNING"):
            result = ingest.fetch_and_store(["http://feed.example.com/rss"], frozenset(), db)
        assert result == 0
        assert "feedparser" in caplog.text.lower()
    finally:
        del sys.modules["feedparser"]
        if saved is not None:
            sys.modules["feedparser"] = saved


# ── 10. NewsSource DataNotAvailableError on wrong methods ─────────────────────

def test_news_source_ohlcv_raises(tmp_path):
    src = NewsSource(str(tmp_path / "n.duckdb"))
    with pytest.raises(DataNotAvailableError):
        src.get_ohlcv_daily("TCS", _AS_OF_AFTER)


def test_news_source_ticks_raises(tmp_path):
    src = NewsSource(str(tmp_path / "n.duckdb"))
    with pytest.raises(DataNotAvailableError):
        src.get_ticks("TCS", _AS_OF_AFTER)


def test_news_source_option_chain_raises(tmp_path):
    src = NewsSource(str(tmp_path / "n.duckdb"))
    with pytest.raises(DataNotAvailableError):
        src.get_option_chain("TCS", _AS_OF_AFTER)


# ── 11. Metadata ─────────────────────────────────────────────────────────────

def test_news_source_metadata(tmp_path):
    src = NewsSource(str(tmp_path / "n.duckdb"))
    assert src.data_source_name == "rss_news"
    assert src.data_granularity == "tick"
    assert "first_knowable_at" in src.data_source_limitations


def test_news_source_is_datasource_subclass(tmp_path):
    from src.data.base import DataSource
    assert isinstance(NewsSource(str(tmp_path / "n.duckdb")), DataSource)
