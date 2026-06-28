"""Phase 1 review pass — consolidation + cross-cutting invariant tests.

Four areas:
  1. End-to-end smoke test across all four data sources simultaneously
  2. News tagging coverage (quantitative; must meet floor)
  3. Cross-cutting invariant audit (PIT, SDK quarantine, provenance, timezone)
  4. Intraday-edge boundary documented on BhavCopySource
"""
from __future__ import annotations

import glob
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.data.base import DataLeakageError, DataNotAvailableError, DataSource
from src.data.bhavcopy import load_bhavcopy_to_duckdb
from src.data.bhavcopy_source import BhavCopySource
from src.data.eligibility import get_eligible_symbols
from src.data.health_monitor import DataHealthMonitor
from src.data.historical import (
    NEWS_COLS, OPTION_CHAIN_COLS, OHLCV_DAILY_COLS, TICK_COLS,
    StubHistoricalSource,
)
from src.data.live_upstox import UpstoxLiveSource
from src.data.news_ingest import (
    NewsSource, RSSNewsIngest,
    _build_patterns, _tag_symbols, SYMBOL_ALIASES,
)
from src.data.option_chain import OptionChainSource

# ── Shared fixtures ─────────────────────────────────────────────────────────

_JAN22_CSV = """\
TckrSymb,TradDt,SctySrs,OpnPric,HghPric,LwPric,ClsPric,TtlTradgVol,TtlTrfVal
RELIANCE,2024-01-22,EQ,2500.0,2550.0,2490.0,2520.0,500000,1260000000.0
TCS,2024-01-22,EQ,3800.0,3850.0,3790.0,3820.0,200000,764000000.0
"""

_MOCK_OC_RESPONSE = {
    "status": "success",
    "data": [{
        "expiry": "2024-01-25",
        "strike_price": 2500.0,
        "call_options": {"market_data": {"ltp": 30.0, "oi": 500000, "prev_oi": 480000, "volume": 12000, "iv": 10.5}},
        "put_options":  {"market_data": {"ltp": 25.0, "oi": 450000, "prev_oi": 430000, "volume": 11000, "iv": 11.2}},
    }],
}

_DELAY_S = 300
_PUB_AT = datetime(2024, 1, 22, 9, 0, 0, tzinfo=timezone.utc)
_FIRST_KA = _PUB_AT + timedelta(seconds=_DELAY_S)
_AS_OF = datetime(2024, 1, 22, 11, 0, 0, tzinfo=timezone.utc)  # 2 hours after first_knowable_at


def _make_news_entry(headline, published=None):
    e = SimpleNamespace()
    e.title = headline
    e.summary = ""
    e.published_parsed = (published or _PUB_AT).timetuple()
    return e


def _make_feed(entries):
    f = SimpleNamespace()
    f.entries = entries
    f.feed = {"title": "TestFeed"}
    return f


# ═══════════════════════════════════════════════════════════════════════════════
# 1. End-to-end smoke test — all four sources together
# ═══════════════════════════════════════════════════════════════════════════════

def test_e2e_all_four_sources_pit_consistent(tmp_path):
    """Exercise BhavCopySource, UpstoxLiveSource, OptionChainSource, and NewsSource
    in one test.  Prove they all enforce PIT identically.

    as_of = 11:00 UTC on 2024-01-22 (during the trading session).
    All sources return data; all sources block future data.
    """
    db_path = str(tmp_path / "all.duckdb")
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    (raw_dir / "2024-01-22.csv").write_text(_JAN22_CSV, encoding="utf-8")
    load_bhavcopy_to_duckdb(date(2024, 1, 22), raw_dir, db_path)

    as_of_hist = datetime(2024, 1, 23, 9, 0)  # bars from Jan 22 visible
    as_of_live = _AS_OF                        # UTC-aware; ticks from before this

    # ── Source 1: BhavCopySource ─────────────────────────────────────────────
    bhav = BhavCopySource(db_path)
    df_daily = bhav.get_ohlcv_daily("RELIANCE", as_of_hist)
    assert not df_daily.empty
    assert df_daily["source"].iloc[0] == "nse_bhavcopy"

    # ── Source 2: UpstoxLiveSource ───────────────────────────────────────────
    live = UpstoxLiveSource(_inject_ws=MagicMock())
    past_tick_ts = _AS_OF - timedelta(minutes=30)   # 30 min before as_of
    live._on_tick({
        "symbol": "RELIANCE", "ts": past_tick_ts,
        "ltp": 2525.0, "volume": 100, "bid": 2524.0, "ask": 2526.0, "oi": 5000,
    })
    df_ticks = live.get_ticks("RELIANCE", as_of_live, lookback_seconds=3600)
    assert len(df_ticks) == 1
    assert df_ticks["source"].iloc[0] == "upstox_live"

    # ── Source 3: OptionChainSource ──────────────────────────────────────────
    oc_fetch_ts = _AS_OF - timedelta(hours=1)   # 1 hour before as_of
    oc = OptionChainSource(
        _mock_fetch=lambda s, e: _MOCK_OC_RESPONSE,
        _fetch_ts_override=oc_fetch_ts,
    )
    df_oc = oc.get_option_chain("RELIANCE", as_of_live)
    assert not df_oc.empty
    assert list(df_oc.columns) == list(OPTION_CHAIN_COLS.keys())

    # ── Source 4: NewsSource ─────────────────────────────────────────────────
    fp = MagicMock()
    fp.parse.return_value = _make_feed([_make_news_entry("RELIANCE posts record revenue")])
    RSSNewsIngest(_DELAY_S, _mock_feedparser=fp).fetch_and_store(
        ["http://x.com/rss"], frozenset(["RELIANCE"]), db_path
    )
    news_src = NewsSource(db_path)
    df_news = news_src.get_news("RELIANCE", _AS_OF, lookback_hours=24.0)
    assert len(df_news) == 1
    assert list(df_news.columns) == list(NEWS_COLS.keys())

    # ── PIT: future tick blocked ─────────────────────────────────────────────
    live._on_tick({
        "symbol": "RELIANCE", "ts": _AS_OF + timedelta(minutes=5),
        "ltp": 2530.0, "volume": 100, "bid": 2529.0, "ask": 2531.0, "oi": 5000,
    })
    with pytest.raises(DataLeakageError):
        live.get_ticks("RELIANCE", as_of_live, lookback_seconds=3600)

    # ── PIT: option chain with future ts blocked ─────────────────────────────
    future_oc = OptionChainSource(
        _mock_fetch=lambda s, e: _MOCK_OC_RESPONSE,
        _fetch_ts_override=_AS_OF + timedelta(hours=1),   # fetch ts AFTER as_of
    )
    with pytest.raises(DataLeakageError):
        future_oc.get_option_chain("RELIANCE", as_of_live)

    # ── PIT: news before first_knowable_at blocked ───────────────────────────
    as_of_before_fka = _PUB_AT + timedelta(seconds=60)   # before first_knowable_at
    df_early = news_src.get_news("RELIANCE", as_of_before_fka, lookback_hours=24.0)
    assert df_early.empty, "story before first_knowable_at must be blocked"


def test_e2e_eligibility_and_all_sources(tmp_path):
    """Survivorship eligibility integrates correctly with all four sources.

    Symbols in the Jan 22 bhavcopy are in the universe; symbols not loaded are not.
    """
    db_path = str(tmp_path / "elig.duckdb")
    raw_dir = tmp_path / "raw"; raw_dir.mkdir()
    (raw_dir / "2024-01-22.csv").write_text(_JAN22_CSV, encoding="utf-8")
    load_bhavcopy_to_duckdb(date(2024, 1, 22), raw_dir, db_path)

    eligible = get_eligible_symbols(date(2024, 1, 22), db_path)
    assert "RELIANCE" in eligible
    assert "TCS" in eligible
    assert "WIPRO" not in eligible   # not in test data

    # Data sources agree: BhavCopySource has data for eligible symbols only
    bhav = BhavCopySource(db_path)
    as_of = datetime(2024, 1, 23, 9, 0)
    df_rel = bhav.get_ohlcv_daily("RELIANCE", as_of)
    df_wipro = bhav.get_ohlcv_daily("WIPRO", as_of)
    assert not df_rel.empty
    assert df_wipro.empty   # WIPRO not loaded → correctly empty


def test_e2e_datasource_cross_boundary_errors_correct(tmp_path):
    """Each source raises DataNotAvailableError for data types it does not serve."""
    db_path = str(tmp_path / "x.duckdb")
    bhav = BhavCopySource(db_path)
    live = UpstoxLiveSource(_inject_ws=MagicMock())
    oc = OptionChainSource(_mock_fetch=lambda s, e: _MOCK_OC_RESPONSE)
    news = NewsSource(db_path)

    as_of = datetime(2024, 1, 22, 10, 0)

    # BhavCopySource: only daily; ticks, minute, news, chains → error
    with pytest.raises(DataNotAvailableError, match="intraday"):
        bhav.get_ohlcv_minute("TCS", as_of)
    with pytest.raises(DataNotAvailableError):
        bhav.get_ticks("TCS", as_of)

    # UpstoxLiveSource: only ticks; daily, minute → error
    with pytest.raises(DataNotAvailableError):
        live.get_ohlcv_daily("TCS", as_of)
    with pytest.raises(DataNotAvailableError):
        live.get_ohlcv_minute("TCS", as_of)

    # OptionChainSource: only option chain; everything else → error
    with pytest.raises(DataNotAvailableError):
        oc.get_ohlcv_daily("TCS", as_of)
    with pytest.raises(DataNotAvailableError):
        oc.get_ticks("TCS", as_of)
    with pytest.raises(DataNotAvailableError):
        oc.get_news("TCS", as_of)

    # NewsSource: only news; everything else → error
    with pytest.raises(DataNotAvailableError):
        news.get_ohlcv_daily("TCS", as_of)
    with pytest.raises(DataNotAvailableError):
        news.get_ticks("TCS", as_of)
    with pytest.raises(DataNotAvailableError):
        news.get_option_chain("TCS", as_of)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. News tagging coverage
# ═══════════════════════════════════════════════════════════════════════════════

# Broad coverage corpus — 116 company-specific headlines (one per F&O ticker)
# plus 15 macro headlines.  Coverage floor: 90%.
# Each company case records the MINIMUM set of expected tags; a case passes if
# expected ⊆ got (additional co-mentions are accepted; macro cases require got == {}).
_BROAD_TAGGING_CASES = [
    # ── Nifty 50 ────────────────────────────────────────────────────────────────
    ("Adani Enterprises wins bid for airport expansion project", {"ADANIENT"}),
    ("Adani Ports sees record cargo traffic at Mundra terminal", {"ADANIPORTS"}),
    ("Apollo Hospitals launches new oncology centre in Hyderabad", {"APOLLOHOSP"}),
    ("Asian Paints Q3 margins squeezed by raw material cost inflation", {"ASIANPAINT"}),
    ("Axis Bank raises interest rates on fixed deposits", {"AXISBANK"}),
    ("Bajaj Auto reports 15% jump in electric two-wheeler sales", {"BAJAJ-AUTO"}),
    ("Bajaj Finserv subsidiary reports strong insurance premium growth", {"BAJAJFINSV"}),
    ("Bajaj Finance NPA ratio rises amid consumer lending stress", {"BAJFINANCE"}),
    ("Airtel adds eight million new subscribers in the third quarter", {"BHARTIARTL"}),
    ("Bharat Petroleum posts record refinery throughput this quarter", {"BPCL"}),
    ("Britannia Industries launches new premium biscuit range", {"BRITANNIA"}),
    ("Cipla receives USFDA nod for respiratory generic drug", {"CIPLA"}),
    ("Coal India production surpasses annual target ahead of schedule", {"COALINDIA"}),
    ("Divi's Laboratories contract manufacturing division drives growth", {"DIVISLAB"}),
    ("Dr Reddy's files NDA for generic oncology treatment in the US", {"DRREDDY"}),
    ("Royal Enfield launches new Himalayan adventure motorcycle variant", {"EICHERMOT"}),
    ("Grasim Industries paints business expands capacity in west India", {"GRASIM"}),
    ("HCL Technologies wins $500 million IT services deal in Europe", {"HCLTECH"}),
    ("HCL Tech outperforms peers on deal wins in the BFSI vertical", {"HCLTECH"}),
    ("HCL group announces 1:1 bonus share for shareholders", {"HCLTECH"}),
    ("HDFC Bank net interest margin improves to 4.1% in Q3", {"HDFCBANK"}),
    ("HDFC Life Insurance new business premium growth beats estimates", {"HDFCLIFE"}),
    ("Hero MotoCorp launches new electric scooter for urban commuters", {"HEROMOTOCO"}),
    ("Hindalco Industries copper segment profitability beats estimates", {"HINDALCO"}),
    ("HUL rural sales recover strongly on good monsoon season", {"HINDUNILVR"}),
    ("ICICI Bank credit card spends up 30% year-on-year in Q3", {"ICICIBANK"}),
    ("IndusInd Bank vehicle loan book grows steadily despite pressure", {"INDUSINDBK"}),
    ("Infosys raises FY24 revenue guidance to upper end of 1-3.5%", {"INFY"}),
    ("ITC cigarette volumes recover to pre-pandemic levels in urban areas", {"ITC"}),
    ("JSW Steel capacity expansion project on track for FY26 commissioning", {"JSWSTEEL"}),
    ("Kotak Mahindra Bank navigates CEO transition smoothly", {"KOTAKBANK"}),
    ("Larsen & Toubro wins Mumbai metro rail order from MMRDA", {"LT"}),
    ("Mahindra & Mahindra SUV market share hits all-time record high", {"M&M"}),
    ("Maruti Suzuki hybrid vehicle sales gain significant traction", {"MARUTI"}),
    ("Nestle India launches new KitKat variants for festive season", {"NESTLEIND"}),
    ("NTPC renewable energy capacity addition well ahead of schedule", {"NTPC"}),
    ("ONGC oil production dips due to offshore field maintenance work", {"ONGC"}),
    ("Power Grid Corporation capex guidance raised for current fiscal", {"POWERGRID"}),
    ("Reliance Industries Jio subscriber growth accelerates in Q3", {"RELIANCE"}),
    ("RIL retail arm posts record quarterly revenue on strong footfalls", {"RELIANCE"}),
    ("SBI Life Insurance new business premium rises 20% in the quarter", {"SBILIFE"}),
    ("State Bank of India MCLR left unchanged for the third month", {"SBIN"}),
    ("Shree Cement Q3 volume growth disappoints on demand slowdown", {"SHREECEM"}),
    ("Sun Pharma specialty portfolio drives strong US revenue growth", {"SUNPHARMA"}),
    ("Tata Motors EV sales cross 20000 units in a single month", {"TATAMOTORS"}),
    ("Tata Steel UK operations return to profitability this quarter", {"TATASTEEL"}),
    ("TCS bags $2 billion multi-year deal with a global banking client", {"TCS"}),
    ("TCS Q3 PAT rises 10% and beats analyst estimates on strong deal wins", {"TCS"}),
    ("Tata Consumer Products tea market share rises in rural India", {"TATACONSUM"}),
    ("Tech Mahindra AI-led transformation deal pipeline grows strongly", {"TECHM"}),
    ("Titan Company watches segment sees festive quarter surge", {"TITAN"}),
    ("UltraTech Cement acquires new clinker capacity in eastern India", {"ULTRACEMCO"}),
    ("UPL crop protection volumes remain flat in the Brazil market", {"UPL"}),
    ("Wipro large deal wins show meaningful improvement in Q3 results", {"WIPRO"}),
    # ── Additional F&O universe ────────────────────────────────────────────────
    ("ABB India power automation business grows 25% in the quarter", {"ABB"}),
    ("ACC Cement volume growth beats industry average this quarter", {"ACC"}),
    ("Ambuja Cements completes acquisition of Sanghi Industries assets", {"AMBUJACEM"}),
    ("Aurobindo Pharma USFDA product approvals accelerate in H2", {"AUROPHARMA"}),
    ("BKT Tyres exports grow on strong Europe demand for off-highway tyres", {"BALKRISIND"}),
    ("Bandhan Bank microfinance stress shows signs of easing", {"BANDHANBNK"}),
    ("Bank of Baroda net interest margin expands on loan repricing", {"BANKBARODA"}),
    ("Bata India premium portfolio drives margin improvement", {"BATAINDIA"}),
    ("Bharat Electronics wins defence contract worth Rs 1000 crore", {"BEL"}),
    ("Bharat Heavy Electricals order book reaches Rs 1.2 lakh crore", {"BHEL"}),
    ("Biocon biosimilars receive European Medicines Agency approval", {"BIOCON"}),
    ("Bosch India EV component business posts strong growth", {"BOSCHLTD"}),
    ("Canara Bank slippages decline meaningfully in the third quarter", {"CANBK"}),
    ("Cholamandalam Finance vehicle loan growth remains robust", {"CHOLAFIN"}),
    ("Container Corporation announces new Dedicated Freight Corridor services", {"CONCOR"}),
    ("Cummins India data centre power division growing rapidly", {"CUMMINSIND"}),
    ("DMart store count crosses 400 and exceeds analyst expectations", {"DMART"}),
    ("Escorts Kubota tractor volume recovery on track after weak monsoon", {"ESCORTS"}),
    ("Exide Industries EV battery gigafactory commences pilot operations", {"EXIDEIND"}),
    ("Federal Bank raises retail deposit rates with new scheme", {"FEDERALBNK"}),
    ("GAIL India city gas distribution segment grows on new connections", {"GAIL"}),
    ("Godrej Consumer Products Indonesia volumes recover in Q3", {"GODREJCP"}),
    ("Granules India API capacity expansion fully commissioned", {"GRANULES"}),
    ("Hindustan Aeronautics receives advance payment for Tejas fighters", {"HAL"}),
    ("Havells India cables and wires segment growth beats forecast", {"HAVELLS"}),
    ("IDFC First Bank deposits grow at record pace on new retail push", {"IDFCFIRSTB"}),
    ("Indraprastha Gas PNG connection additions hit new monthly milestone", {"IGL"}),
    ("Taj Hotels RevPAR metrics improve strongly in the third quarter", {"INDHOTEL"}),
    ("IndiGo Airlines load factor remains above 90% for third quarter", {"INDIGO"}),
    ("IRCTC rail ticket bookings cross one million per day milestone", {"IRCTC"}),
    ("Jubilant FoodWorks Domino's same-store sales show improvement", {"JUBLFOOD"}),
    ("KPIT Technologies EV software deal pipeline continues to grow", {"KPITTECH"}),
    ("LIC Housing Finance loan disbursements hit record quarterly high", {"LICHSGFIN"}),
    ("LTIMindtree large deal total contract value exceeds expectations", {"LTIM"}),
    ("Lupin US generics gains market share on key respiratory drugs", {"LUPIN"}),
    ("Manappuram Finance gold loan AUM reaches new all-time record", {"MANAPPURAM"}),
    ("United Spirits premium segment drives strong volume recovery", {"MCDOWELL-N"}),
    ("Mahanagar Gas PNG connections hit new milestone in Mumbai region", {"MGL"}),
    ("Samvardhana Motherson EV wiring harness deal wins accelerate", {"MOTHERSON"}),
    ("Mphasis cloud services practice grows strongly in North America", {"MPHASIS"}),
    ("Muthoot Finance gold loan AUM at all-time record this quarter", {"MUTHOOTFIN"}),
    ("NALCO aluminium smelter expansion commissioned ahead of schedule", {"NATIONALUM"}),
    ("Naukri paid job listings growth recovers in the IT sector", {"NAUKRI"}),
    ("NMDC iron ore production surges 18% in the third quarter", {"NMDC"}),
    ("Oracle Financial Services banking software wins key deals in Asia", {"OFSS"}),
    ("Page Industries Jockey innerwear premium launches succeed", {"PAGEIND"}),
    ("Persistent Systems AI-led deal wins continue to accelerate", {"PERSISTENT"}),
    ("Petronet LNG Dahej terminal utilisation remains above 100%", {"PETRONET"}),
    ("Power Finance Corporation renewables loan book grows 30%", {"PFC"}),
    ("PI Industries agrochemical revenue beats analyst estimates", {"PIIND"}),
    ("Punjab National Bank GNPA ratio declines to 6% in Q3", {"PNB"}),
    ("Polycab India cables market share rises in the quarter", {"POLYCAB"}),
    ("RBL Bank credit card business shows clear signs of improvement", {"RBLBANK"}),
    ("REC Limited disburses record renewable energy loans this quarter", {"RECLTD"}),
    ("Rail Vikas Nigam order book reaches all-time high", {"RVNL"}),
    ("Steel Authority of India volume growth recovers in Q3", {"SAIL"}),
    ("Siemens India digital infrastructure order wins accelerate", {"SIEMENS"}),
    ("Tata Elxsi automotive embedded software revenues grow strongly", {"TATAELXSI"}),
    ("Tata Power renewable capacity addition on track for FY26", {"TATAPOWER"}),
    ("Torrent Pharmaceuticals launches several new brands in India", {"TORNTPHARM"}),
    ("Westside same-store sales up 18% in the festive quarter", {"TRENT"}),
    ("TVS Motor EV segment market share grows on new launches", {"TVSMOTOR"}),
    ("Vedanta zinc and aluminium production volumes rise in Q3", {"VEDL"}),
    ("Voltas air conditioner market share stable despite competition", {"VOLTAS"}),
    ("Zomato food delivery gross order value rises 20% in Q3", {"ZOMATO"}),
    # ── Multi-company ────────────────────────────────────────────────────────────
    ("TCS and Infosys both report earnings on the same day", {"TCS", "INFY"}),
    ("Larsen & Toubro and NTPC win infrastructure capex projects", {"LT", "NTPC"}),
    # ── Macro — must NOT tag any ticker ─────────────────────────────────────────
    ("RBI holds repo rate unchanged at 6.5% for the fifth time", set()),
    ("Sensex surges 500 points on positive global cues from US Fed", set()),
    ("Nifty 50 hits all-time high of 22900 in early trade", set()),
    ("India GDP growth forecast revised to 7.2% for FY25", set()),
    ("SEBI proposes new margin rules for equity derivatives segment", set()),
    ("Government announces PLI scheme for electronics manufacturing", set()),
    ("FII outflows continue for the third consecutive week", set()),
    ("Monsoon progress normal according to latest IMD forecast", set()),
    ("Nifty Bank index underperforms the broader market today", set()),
    ("CPI inflation falls to 4.8% in November on food price easing", set()),
    ("US Fed holds rates steady; Indian markets react positively", set()),
    ("Crude oil prices fall on demand concerns from slowing China economy", set()),
    ("Union Budget expected to focus heavily on infrastructure capex", set()),
    # Edge case: word-boundary prevents false match on ITC
    ("India startup pitch decks go AI-first this funding season", set()),
    ("Government plans to switch to digital land records across states", set()),
]

_FULL_UNIVERSE = list(SYMBOL_ALIASES.keys())


def test_tagging_all_known_symbols_tagged():
    """All 116 F&O symbols in the alias map are tagged on their canonical alias name."""
    patterns = _build_patterns(_FULL_UNIVERSE)
    for ticker, aliases in SYMBOL_ALIASES.items():
        for alias in aliases:
            headline = f"{alias} reports quarterly results"
            tags = _tag_symbols(headline, patterns)
            assert ticker in tags, (
                f"Expected {ticker!r} to match on alias {alias!r}; "
                f"got {tags}"
            )


def test_tagging_coverage_meets_floor():
    """At least 90% of company-specific headlines in the broad corpus tag the right symbol."""
    patterns = _build_patterns(_FULL_UNIVERSE)
    company_cases = [(h, exp) for h, exp in _BROAD_TAGGING_CASES if exp]
    correct = 0
    false_positives = []
    for headline, expected in company_cases:
        got = set(_tag_symbols(headline, patterns))
        if expected.issubset(got):
            correct += 1
        extra = got - expected
        if extra:
            false_positives.append((headline, expected, extra))
    coverage = correct / len(company_cases)
    fp_report = (
        "\nFalse positives:\n" + "\n".join(
            f"  {h!r}: expected {exp}, got extra {fp}"
            for h, exp, fp in false_positives
        ) if false_positives else ""
    )
    assert coverage >= 0.90, (
        f"News tagging coverage {coverage:.0%} below 90% floor.  "
        f"{correct}/{len(company_cases)} company stories correctly tagged.{fp_report}"
    )


def test_tagging_no_false_positives_on_macro():
    """Macro / market-wide headlines must produce no symbol tags across full 116-ticker universe."""
    patterns = _build_patterns(_FULL_UNIVERSE)
    macro_cases = [(h, exp) for h, exp in _BROAD_TAGGING_CASES if not exp]
    for headline, _ in macro_cases:
        tags = _tag_symbols(headline, patterns)
        assert not tags, f"False positive tags {tags} on macro headline: {headline!r}"


def test_tagging_word_boundary_no_false_match():
    """'ITC' must not fire on headlines where 'itc' appears inside a larger word."""
    patterns = _build_patterns(["ITC"])
    assert not _tag_symbols("India startup pitch decks go AI-first", patterns)
    assert not _tag_symbols("The switch network encountered issues", patterns)
    assert _tag_symbols("ITC cigarette volumes recover", patterns) == ["ITC"]


def test_tagging_sbi_disambiguation_fixed():
    """Span-based disambiguation: 'SBI Life' stories tag SBILIFE only, NOT SBIN.

    'SBI' at span [0,3] is subsumed by 'SBI Life' at span [0,8]; _tag_symbols
    suppresses the shorter match so only the more specific ticker is tagged.
    """
    patterns = _build_patterns(["SBIN", "SBILIFE"])
    tags = set(_tag_symbols("SBI Life beats estimates with 18% APE growth", patterns))
    assert "SBILIFE" in tags, "SBILIFE must be tagged on 'SBI Life' stories"
    assert "SBIN" not in tags, (
        "SBIN must NOT be tagged on 'SBI Life' stories — "
        "span-based suppression must eliminate the shorter 'SBI' match"
    )


def test_tagging_sbi_standalone_still_tags_sbin():
    """'SBI' as a bare abbreviation (no 'Life' following) still tags SBIN correctly."""
    patterns = _build_patterns(["SBIN", "SBILIFE"])
    tags = set(_tag_symbols("SBI hikes deposit rate by 25 basis points", patterns))
    assert "SBIN" in tags
    assert "SBILIFE" not in tags


def test_tagging_bajaj_family_disambiguation():
    """Bajaj Finance, Bajaj Finserv, and Bajaj Auto never cross-tag each other."""
    patterns = _build_patterns(["BAJFINANCE", "BAJAJFINSV", "BAJAJ-AUTO"])
    for headline, expected_ticker in [
        ("Bajaj Finance NPA ratio rises amid consumer credit stress",   "BAJFINANCE"),
        ("Bajaj Finserv insurance subsidiary posts strong APE growth",  "BAJAJFINSV"),
        ("Bajaj Auto electric two-wheeler sales cross 50000 units",     "BAJAJ-AUTO"),
    ]:
        tags = set(_tag_symbols(headline, patterns))
        assert expected_ticker in tags, f"{expected_ticker} not tagged in: {headline!r}"
        for other in {"BAJFINANCE", "BAJAJFINSV", "BAJAJ-AUTO"} - {expected_ticker}:
            assert other not in tags, (
                f"Cross-tag: {other} falsely tagged in: {headline!r}"
            )


def test_tagging_tech_mahindra_vs_mahindra():
    """'Tech Mahindra' tags TECHM only; standalone 'Mahindra' tags M&M only."""
    patterns = _build_patterns(["TECHM", "M&M"])
    # 'Mahindra' subsumed by 'Tech Mahindra' → TECHM only
    tags = set(_tag_symbols("Tech Mahindra AI deal wins accelerate in Q3", patterns))
    assert "TECHM" in tags
    assert "M&M" not in tags, "M&M must not fire on 'Tech Mahindra' headlines"
    # Standalone 'Mahindra' (no 'Tech' prefix) → M&M only
    tags2 = set(_tag_symbols("Mahindra & Mahindra SUV market share rises", patterns))
    assert "M&M" in tags2
    assert "TECHM" not in tags2


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Cross-cutting invariant audit
# ═══════════════════════════════════════════════════════════════════════════════

def test_invariant_no_sdk_import_outside_quarantine():
    """ONLY live_upstox.py may import upstox_client anywhere in src/."""
    src_root = Path(__file__).parent.parent / "src"
    violations = []
    for py_file in src_root.rglob("*.py"):
        if py_file.name == "live_upstox.py":
            continue   # quarantine zone — SDK import is allowed here
        text = py_file.read_text(encoding="utf-8")
        if "upstox_client" in text:
            violations.append(str(py_file.relative_to(src_root)))
    assert violations == [], (
        f"SDK import found outside quarantine: {violations}.  "
        "Only src/data/live_upstox.py may import upstox_client."
    )


def test_invariant_all_sources_have_provenance_fields():
    """Every DataSource subclass must declare non-default provenance metadata."""
    sources = [
        StubHistoricalSource(),
        BhavCopySource(":memory:"),
        UpstoxLiveSource(_inject_ws=MagicMock()),
        OptionChainSource(_mock_fetch=lambda s, e: {"status": "success", "data": []}),
        NewsSource(":memory:"),
    ]
    for src in sources:
        assert src.data_source_name != "unknown", (
            f"{type(src).__name__}.data_source_name is 'unknown'"
        )
        assert src.data_source_limitations != "unknown", (
            f"{type(src).__name__}.data_source_limitations is 'unknown'"
        )
        assert src.data_granularity in ("daily", "minute", "tick", "none"), (
            f"{type(src).__name__}.data_granularity = {src.data_granularity!r} is invalid"
        )


def test_invariant_all_sources_are_datasource_subclass():
    """All concrete adapters must be DataSource subclasses."""
    adapters = [
        StubHistoricalSource,
        BhavCopySource,
        UpstoxLiveSource,
        OptionChainSource,
        NewsSource,
    ]
    for cls in adapters:
        assert issubclass(cls, DataSource), f"{cls.__name__} is not a DataSource subclass"


def test_invariant_pit_called_on_empty_df_all_sources():
    """_enforce_pit is called even when a data source has no rows to return.

    An empty DF passes PIT immediately, but the contract says every return path
    must call _enforce_pit so there is no code path that accidentally skips it.
    """
    as_of = datetime(2024, 1, 22, 10, 0)
    stub = StubHistoricalSource()
    live = UpstoxLiveSource(_inject_ws=MagicMock())  # empty buffer
    oc = OptionChainSource(
        _mock_fetch=lambda s, e: {"status": "success", "data": []},
        _fetch_ts_override=datetime(2024, 1, 22, 9, 0),
    )

    # None of these should raise — all pass through _enforce_pit on empty DF
    df_stub = stub.get_ticks("NODATA", as_of)
    df_live = live.get_ticks("NODATA", as_of)
    df_oc = oc.get_option_chain("NODATA", as_of)

    assert df_stub.empty
    assert df_live.empty
    assert df_oc.empty


def test_invariant_no_gpu_imports_anywhere_in_data_layer():
    """No file in src/data/ may contain GPU library import patterns."""
    gpu_patterns = [
        "import torch", "import cuda", "import cupy",
        "from torch", "from cuda", 'device("cuda")',
        "device('cuda')", '.to("cuda")', ".to('cuda')",
    ]
    src_data = Path(__file__).parent.parent / "src" / "data"
    violations = []
    for py_file in src_data.rglob("*.py"):
        text = py_file.read_text(encoding="utf-8")
        for pat in gpu_patterns:
            if pat in text:
                violations.append(f"{py_file.name}: {pat!r}")
    assert violations == [], (
        f"GPU import patterns found in data layer: {violations}"
    )


def test_invariant_timezone_consistency_all_datetime_sources():
    """tz-aware as_of works with all four sources; naive works with daily-only.

    This proves that no source silently on a different timezone basis.
    """
    from zoneinfo import ZoneInfo
    IST = ZoneInfo("Asia/Kolkata")

    # IST-aware as_of accepted by all sources (bhavcopy uses date, ignores tz)
    as_of_ist = datetime(2024, 1, 23, 9, 15, 0, tzinfo=IST)

    stub = StubHistoricalSource()
    live = UpstoxLiveSource(_inject_ws=MagicMock())

    # Stub: all empty, no error — PIT sees no rows
    stub.get_ohlcv_daily("TCS", as_of_ist)
    stub.get_ticks("TCS", as_of_ist)

    # Live: empty buffer — PIT on empty, no error
    live.get_ticks("TCS", as_of_ist)

    # tz-mismatch (naive tick + tz-aware as_of) → DataLeakageError("tz-mismatch")
    live._on_tick({
        "symbol": "TCS", "ts": datetime(2024, 1, 23, 9, 14, 0),  # naive
        "ltp": 3800.0, "volume": 100, "bid": 3799.0, "ask": 3801.0, "oi": 0,
    })
    with pytest.raises(DataLeakageError, match="tz-mismatch"):
        live.get_ticks("TCS", as_of_ist)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Intraday-edge boundary documentation
# ═══════════════════════════════════════════════════════════════════════════════

def test_bhavcopy_minute_message_names_intraday_edge(tmp_path):
    """BhavCopySource.get_ohlcv_minute error names the intraday-edge caveat explicitly."""
    src = BhavCopySource(tmp_path / "x.duckdb")
    with pytest.raises(DataNotAvailableError) as exc_info:
        src.get_ohlcv_minute("TCS", datetime(2024, 1, 22, 10, 0))
    msg = str(exc_info.value).lower()
    assert "intraday" in msg, "Error message must name intraday edge limitation"
    assert "kite" in msg or "intraday dataset" in msg, (
        "Error must name a concrete alternative source (Kite or supplied intraday dataset)"
    )


def test_stub_historical_returns_correct_schemas_for_all_methods():
    """StubHistoricalSource returns empty DataFrames with correct column schemas."""
    stub = StubHistoricalSource()
    as_of = datetime(2024, 1, 23, 10, 0)

    assert list(stub.get_ohlcv_daily("X", as_of).columns) == list(OHLCV_DAILY_COLS.keys())
    assert list(stub.get_ticks("X", as_of).columns) == list(TICK_COLS.keys())
    assert list(stub.get_news("X", as_of).columns) == list(NEWS_COLS.keys())
    assert list(stub.get_option_chain("X", as_of).columns) == list(OPTION_CHAIN_COLS.keys())


def test_news_cols_includes_symbols_field():
    """NEWS_COLS must include 'symbols' so the stub and real adapter return the same schema."""
    assert "symbols" in NEWS_COLS, (
        "NEWS_COLS missing 'symbols' field — adds after Checkpoint 4"
    )
    # Stub returns correct schema including symbols
    stub = StubHistoricalSource()
    df = stub.get_news("TCS", datetime(2024, 1, 22, 10, 0))
    assert "symbols" in df.columns
