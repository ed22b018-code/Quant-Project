"""RSS news ingest + NewsSource DataSource adapter.

Ingest design
─────────────
RSSNewsIngest.fetch_and_store():
  1. Parse feed URLs with feedparser (deferred import — module importable without it).
  2. For each entry, build news_id = SHA-256 of (normalized headline + normalized body).
     Normalization: lowercase, strip all punctuation, collapse whitespace.
     Same story with a typo corrected produces the same hash because punctuation and
     case differences vanish.
  3. Tag symbols using word-boundary regex against ticker strings AND their aliases.
     Untagged stories (no symbol matched) are stored with symbols=[] and logged at DEBUG.
  4. Set first_knowable_at = published_at + ingestion_delay_seconds (default 300 s).
  5. Upsert into DuckDB `news` table with ON CONFLICT (news_id) DO NOTHING.

NewsSource.get_news():
  Queries WHERE first_knowable_at < as_of AND list_contains(symbols, ?).
  PIT column is first_knowable_at — never published_at.  The ingestion delay buffer
  ensures the model never acts on a story before it could realistically have been read.

CPU-only — no GPU library imports.
"""
from __future__ import annotations

import hashlib
import logging
import re
import string
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable

import pandas as pd

from src.data.base import DataNotAvailableError, DataSource
from src.data.historical import NEWS_COLS, _empty
from src.db.duckdb_client import DuckDBClient
from src.db.schema import DUCKDB_SCHEMA_SQL

_log = logging.getLogger(__name__)

# ── Symbol alias map ─────────────────────────────────────────────────────────
# Maps ticker → list of common names/aliases used in news text.
# Word-boundary regex is built at ingest time; entries are case-insensitive.
SYMBOL_ALIASES: dict[str, list[str]] = {
    # ── Nifty 50 ──────────────────────────────────────────────────────────────
    # Collision notes:
    #   SBIN/SBILIFE: "SBI" inside "SBI Life" — resolved by span-based disambiguation
    #   in _tag_symbols (SBI's shorter span subsumed by SBI Life's longer span).
    #   BAJAJ family: three tickers; none share a common alias — always use full form.
    #   M&M/TECHM: "Mahindra" standalone maps to M&M; "Tech Mahindra" maps to TECHM.
    #   span-based suppression handles: "Tech Mahindra" headline → TECHM only (M&M suppressed).
    #   HDFCBANK/HDFCLIFE: no bare "HDFC" alias — post-merger ambiguity; full form only.
    "ADANIENT":   ["Adani Enterprises"],
    "ADANIPORTS": ["Adani Ports"],
    "APOLLOHOSP": ["Apollo Hospitals"],
    "ASIANPAINT": ["Asian Paints"],
    "AXISBANK":   ["Axis Bank"],
    "BAJAJ-AUTO": ["Bajaj Auto"],
    "BAJAJFINSV": ["Bajaj Finserv"],
    "BAJFINANCE": ["Bajaj Finance"],
    "BHARTIARTL": ["Bharti Airtel", "Airtel"],
    "BPCL":       ["Bharat Petroleum"],
    "BRITANNIA":  ["Britannia"],
    "CIPLA":      ["Cipla"],
    "COALINDIA":  ["Coal India"],
    "DIVISLAB":   ["Divi's Laboratories", "Divi's Labs"],
    "DRREDDY":    ["Dr Reddy's Laboratories", "Dr Reddy's", "Dr Reddy"],
    "EICHERMOT":  ["Eicher Motors", "Royal Enfield"],
    "GRASIM":     ["Grasim Industries"],
    "HCLTECH":    ["HCL Technologies", "HCL Tech", "HCL"],
    "HDFCBANK":   ["HDFC Bank"],
    "HDFCLIFE":   ["HDFC Life Insurance", "HDFC Life"],
    "HEROMOTOCO": ["Hero MotoCorp", "Hero Moto"],
    "HINDALCO":   ["Hindalco"],
    "HINDUNILVR": ["Hindustan Unilever", "HUL"],
    "ICICIBANK":  ["ICICI Bank"],
    "INDUSINDBK": ["IndusInd Bank"],
    "INFY":       ["Infosys"],
    "ITC":        ["ITC Limited"],
    "JSWSTEEL":   ["JSW Steel"],
    "KOTAKBANK":  ["Kotak Mahindra Bank", "Kotak Bank", "Kotak"],
    "LT":         ["Larsen & Toubro", "L&T"],
    "M&M":        ["Mahindra & Mahindra", "M&M", "Mahindra"],
    "MARUTI":     ["Maruti Suzuki", "Maruti"],
    "NESTLEIND":  ["Nestle India"],
    "NTPC":       ["NTPC"],
    "ONGC":       ["ONGC"],
    "POWERGRID":  ["Power Grid", "Power Grid Corporation"],
    "RELIANCE":   ["Reliance Industries", "RIL", "Reliance"],
    "SBILIFE":    ["SBI Life Insurance", "SBI Life"],
    "SBIN":       ["State Bank of India", "SBI"],
    "SHREECEM":   ["Shree Cement"],
    "SUNPHARMA":  ["Sun Pharma", "Sun Pharmaceutical"],
    "TATAMOTORS": ["Tata Motors"],
    "TATASTEEL":  ["Tata Steel"],
    "TCS":        ["Tata Consultancy Services", "TCS"],
    "TATACONSUM": ["Tata Consumer Products", "Tata Consumer"],
    "TECHM":      ["Tech Mahindra"],
    "TITAN":      ["Titan Company", "Titan"],
    "ULTRACEMCO": ["UltraTech Cement", "UltraTech"],
    "UPL":        ["UPL"],
    "WIPRO":      ["Wipro"],
    # ── Additional F&O universe ────────────────────────────────────────────────
    "ABB":        ["ABB India"],
    "ACC":        ["ACC Cement", "ACC Limited"],
    "AMBUJACEM":  ["Ambuja Cements", "Ambuja"],
    "AUROPHARMA": ["Aurobindo Pharma", "Aurobindo"],
    "BALKRISIND": ["Balkrishna Industries", "BKT Tyres"],
    "BANDHANBNK": ["Bandhan Bank"],
    "BANKBARODA": ["Bank of Baroda"],
    "BATAINDIA":  ["Bata India", "Bata"],
    "BEL":        ["Bharat Electronics"],
    "BHEL":       ["Bharat Heavy Electricals"],
    "BIOCON":     ["Biocon"],
    "BOSCHLTD":   ["Bosch India"],
    "CANBK":      ["Canara Bank"],
    "CHOLAFIN":   ["Cholamandalam Finance", "Chola Finance"],
    "CONCOR":     ["Container Corporation", "Concor"],
    "CUMMINSIND": ["Cummins India"],
    "DMART":      ["DMart", "Avenue Supermarts"],
    "ESCORTS":    ["Escorts Kubota", "Escorts"],
    "EXIDEIND":   ["Exide Industries", "Exide"],
    "FEDERALBNK": ["Federal Bank"],
    "GAIL":       ["GAIL India"],
    "GODREJCP":   ["Godrej Consumer Products", "Godrej Consumer"],
    "GRANULES":   ["Granules India"],
    "HAL":        ["Hindustan Aeronautics"],
    "HAVELLS":    ["Havells India", "Havells"],
    "IDFCFIRSTB": ["IDFC First Bank"],
    "IGL":        ["Indraprastha Gas"],
    "INDHOTEL":   ["Indian Hotels", "Taj Hotels", "IHCL"],
    "INDIGO":     ["IndiGo Airlines", "IndiGo"],
    "IRCTC":      ["IRCTC"],
    "JUBLFOOD":   ["Jubilant FoodWorks", "Domino's India"],
    "KPITTECH":   ["KPIT Technologies", "KPIT"],
    "LICHSGFIN":  ["LIC Housing Finance"],
    "LTIM":       ["LTIMindtree"],
    "LUPIN":      ["Lupin"],
    "MANAPPURAM": ["Manappuram Finance", "Manappuram"],
    "MCDOWELL-N": ["United Spirits", "McDowell's"],
    "MGL":        ["Mahanagar Gas"],
    "MOTHERSON":  ["Samvardhana Motherson", "Motherson Sumi"],
    "MPHASIS":    ["Mphasis"],
    "MUTHOOTFIN": ["Muthoot Finance", "Muthoot"],
    "NATIONALUM": ["NALCO", "National Aluminium"],
    "NAUKRI":     ["Naukri", "Info Edge"],
    "NMDC":       ["NMDC"],
    "OFSS":       ["Oracle Financial Services"],
    "PAGEIND":    ["Page Industries", "Jockey India"],
    "PERSISTENT": ["Persistent Systems"],
    "PETRONET":   ["Petronet LNG"],
    "PFC":        ["Power Finance Corporation"],
    "PIIND":      ["PI Industries"],
    "PNB":        ["Punjab National Bank"],
    "POLYCAB":    ["Polycab India", "Polycab"],
    "RBLBANK":    ["RBL Bank"],
    "RECLTD":     ["REC Limited", "Rural Electrification Corporation"],
    "RVNL":       ["Rail Vikas Nigam"],
    "SAIL":       ["Steel Authority of India"],
    "SIEMENS":    ["Siemens India"],
    "TATAELXSI":  ["Tata Elxsi"],
    "TATAPOWER":  ["Tata Power"],
    "TORNTPHARM": ["Torrent Pharmaceuticals", "Torrent Pharma"],
    "TRENT":      ["Trent", "Westside", "Zudio"],
    "TVSMOTOR":   ["TVS Motor"],
    "VEDL":       ["Vedanta"],
    "VOLTAS":     ["Voltas"],
    "ZOMATO":     ["Zomato"],
}

_PUNCT_TRANS = str.maketrans("", "", string.punctuation)


def _normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not text:
        return ""
    lowered = text.lower()
    no_punct = lowered.translate(_PUNCT_TRANS)
    return " ".join(no_punct.split())


def _content_hash(headline: str, body: str) -> str:
    """SHA-256 of normalized headline + body — stable dedup key."""
    payload = _normalize_text(headline) + "\n" + _normalize_text(body)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_patterns(universe: Iterable[str]) -> list[tuple[str, re.Pattern]]:
    """Build (ticker, compiled_regex) pairs for all tickers in universe."""
    patterns: list[tuple[str, re.Pattern]] = []
    seen_tickers: set[str] = set()
    for ticker in universe:
        if ticker in seen_tickers:
            continue
        seen_tickers.add(ticker)
        terms = [ticker] + SYMBOL_ALIASES.get(ticker, [])
        # One combined pattern per ticker: all aliases with word boundaries
        combined = "|".join(
            r"\b" + re.escape(term) + r"\b" for term in terms
        )
        patterns.append((ticker, re.compile(combined, re.IGNORECASE)))
    return patterns


def _tag_symbols(text: str, patterns: list[tuple[str, re.Pattern]]) -> list[str]:
    """Return sorted list of tickers whose alias matches `text`, with span-based disambiguation.

    When ticker A's match span is strictly contained within ticker B's longer match span,
    ticker A is suppressed — the more specific (longer) match wins.  Example: 'SBI' at
    [0,3] is subsumed by 'SBI Life' at [0,8] in 'SBI Life beats estimates', so SBIN is
    suppressed and only SBILIFE is tagged.

    A ticker with ANY non-subsumed match remains tagged even if it also has subsumed matches.
    """
    all_hits: list[tuple[str, int, int]] = []
    for ticker, pat in patterns:
        for m in pat.finditer(text):
            all_hits.append((ticker, m.start(), m.end()))
    if not all_hits:
        return []

    def _subsumed(ticker: str, s: int, e: int) -> bool:
        return any(
            t2 != ticker and s2 <= s and e2 >= e and (e2 - s2) > (e - s)
            for t2, s2, e2 in all_hits
        )

    tagged: set[str] = set()
    suppressed: set[str] = set()
    for ticker, s, e in all_hits:
        if _subsumed(ticker, s, e):
            suppressed.add(ticker)
        else:
            tagged.add(ticker)
    suppressed -= tagged  # ticker with any non-subsumed match stays tagged

    if suppressed:
        _log.debug(
            "_tag_symbols: %r suppressed by more-specific match in: %.100r",
            sorted(suppressed), text,
        )
    if not tagged:
        _log.debug("_tag_symbols: no symbols tagged in: %.100r", text)
    elif len(tagged) > 1:
        _log.debug(
            "_tag_symbols: %d symbols tagged %r in: %.100r",
            len(tagged), sorted(tagged), text,
        )
    return sorted(tagged)


# ── Ingester ─────────────────────────────────────────────────────────────────

class RSSNewsIngest:
    """Fetch RSS feeds, dedup, tag symbols, and store in DuckDB.

    Parameters
    ----------
    ingestion_delay_seconds:
        Buffer added to published_at to compute first_knowable_at.  Default 300 s
        (5 min) — conservative estimate of the time between a story going live and
        the system noticing it.
    _mock_feedparser:
        Test hook — object with a `.parse(url)` method returning a feedparser-style
        result dict.  When set, bypasses the real feedparser import.
    """

    def __init__(
        self,
        ingestion_delay_seconds: int = 300,
        _mock_feedparser: object | None = None,
    ) -> None:
        self._delay = timedelta(seconds=ingestion_delay_seconds)
        self._mock_feedparser = _mock_feedparser

    def fetch_and_store(
        self,
        feed_urls: list[str],
        universe: Iterable[str],
        db_path: str,
    ) -> int:
        """Fetch feeds, dedup, tag, store.  Returns number of new rows inserted.

        If feedparser is not installed and no mock is provided, logs a WARNING
        and returns 0 rather than crashing.
        """
        if self._mock_feedparser is not None:
            fp = self._mock_feedparser
        else:
            try:
                import feedparser as fp  # type: ignore[import]
            except ImportError:
                _log.warning(
                    "feedparser is not installed; RSS ingest unavailable.  "
                    "pip install feedparser>=6.0.11"
                )
                return 0

        patterns = _build_patterns(universe)
        now_utc = datetime.now(timezone.utc)

        rows: list[dict] = []
        for url in feed_urls:
            try:
                feed = fp.parse(url)
            except Exception as exc:
                _log.warning("RSSNewsIngest: error parsing feed %r: %s", url, exc)
                continue

            for entry in getattr(feed, "entries", []):
                headline = getattr(entry, "title", "") or ""
                body = getattr(entry, "summary", "") or ""
                source = getattr(feed, "feed", {}).get("title", "") if hasattr(feed, "feed") else ""
                if not source:
                    source = url

                # published_at
                published_struct = getattr(entry, "published_parsed", None)
                if published_struct is not None:
                    import calendar as _cal
                    published_at = datetime.fromtimestamp(
                        _cal.timegm(published_struct), tz=timezone.utc
                    )
                else:
                    published_at = now_utc

                # Dedup hash
                news_id = _content_hash(headline, body)

                # Symbol tagging
                full_text = f"{headline} {body}"
                symbols = _tag_symbols(full_text, patterns)
                if not symbols:
                    _log.debug(
                        "RSSNewsIngest: story untagged (no symbol matched): %r",
                        headline[:80],
                    )

                first_knowable_at = published_at + self._delay

                rows.append({
                    "news_id":           news_id,
                    "headline":          headline,
                    "body":              body,
                    "source":            source,
                    "published_at":      published_at,
                    "first_knowable_at": first_knowable_at,
                    "symbols":           symbols,
                    "ingested_at":       now_utc,
                })

        if not rows:
            return 0

        df = pd.DataFrame(rows)
        inserted = _upsert_news(df, db_path)
        _log.info("RSSNewsIngest: %d new stories inserted (of %d fetched).", inserted, len(rows))
        return inserted


def _upsert_news(df: pd.DataFrame, db_path: str) -> int:
    """Upsert news rows into DuckDB, skipping duplicates by news_id."""
    with DuckDBClient(db_path) as db:
        db.execute_schema(DUCKDB_SCHEMA_SQL)

        # Register the DataFrame so we can SELECT from it (direct conn for array support)
        db.conn.register("_news_staging", df)

        before = db.conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]

        db.conn.execute("""
            INSERT INTO news (news_id, headline, body, source,
                              published_at, first_knowable_at, symbols, ingested_at)
            SELECT news_id, headline, body, source,
                   published_at, first_knowable_at, symbols, ingested_at
            FROM _news_staging
            ON CONFLICT (news_id) DO NOTHING
        """)

        after = db.conn.execute("SELECT COUNT(*) FROM news").fetchone()[0]
        db.conn.unregister("_news_staging")

    return after - before


# ── DataSource adapter ────────────────────────────────────────────────────────

class NewsSource(DataSource):
    """DataSource adapter that reads news from DuckDB, PIT on first_knowable_at.

    Returns only stories tagged with the requested symbol where
    first_knowable_at < as_of.  Untagged stories (symbols=[]) are never returned
    by a symbol query — they are stored for future analysis only.
    """

    data_source_name = "rss_news"
    data_source_limitations = (
        "RSS-sourced news only; first_knowable_at = published_at + ingestion delay; "
        "symbol tagging via word-boundary regex on aliases — novel ticker names may be "
        "missed; dedup by content hash (normalized) so minor edits create a new story; "
        "historical backfill requires re-fetching and re-ingesting archived feeds"
    )
    data_granularity = "tick"

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path

    def get_news(
        self,
        symbol: str,
        as_of: datetime,
        lookback_hours: float = 24.0,
    ) -> pd.DataFrame:
        """News for `symbol` with first_knowable_at in [as_of - lookback, as_of).

        PIT column: first_knowable_at (not published_at).

        Returns an empty DataFrame (NEWS_COLS schema) if no stories are found.
        """
        lookback_start = as_of - timedelta(hours=lookback_hours)

        sql = """
            SELECT news_id, headline, body, source,
                   published_at, first_knowable_at, symbols
            FROM news
            WHERE first_knowable_at < ?
              AND first_knowable_at >= ?
              AND list_contains(symbols, ?)
            ORDER BY first_knowable_at DESC
        """
        with DuckDBClient(self._db_path) as db:
            db.execute_schema(DUCKDB_SCHEMA_SQL)
            df = db.query_df(sql, (as_of, lookback_start, symbol))

        if df.empty:
            return self._enforce_pit(_empty(NEWS_COLS), "first_knowable_at", as_of)

        # Normalize datetime columns — DuckDB may return TIMESTAMPTZ as UTC-aware
        for col in ("published_at", "first_knowable_at"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], utc=True)

        return self._enforce_pit(df, "first_knowable_at", as_of)

    # ── DataSource stubs ─────────────────────────────────────────────────────

    def get_ohlcv_daily(self, symbol, as_of, lookback_days=252):
        raise DataNotAvailableError(
            f"NewsSource does not provide OHLCV data for {symbol!r}."
        )

    def get_ohlcv_minute(self, symbol, as_of, lookback_bars=390):
        raise DataNotAvailableError(
            f"NewsSource does not provide minute bars for {symbol!r}."
        )

    def get_ticks(self, symbol, as_of, lookback_seconds=300):
        raise DataNotAvailableError(
            f"NewsSource does not provide tick data for {symbol!r}."
        )

    def get_option_chain(self, symbol, as_of):
        raise DataNotAvailableError(
            f"NewsSource does not provide option chain data for {symbol!r}."
        )
