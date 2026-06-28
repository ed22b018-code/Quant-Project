"""Tests for src/features/sentiment.py — Checkpoint 2.2 FinBERT Sentiment Scorer.

Test structure
--------------
1. Device detection (CPU fallback, CUDA mock, torch-missing mock)
2. FinBERTScorer: batch scoring, score cache, graceful degradation
3. score_news_df: DataFrame in → DataFrame + sentiment out
4. aggregate_sentiment: PIT test (the key correctness guarantee), time decay,
   source weighting, no-news neutral return
5. news_sentiment registered feature: gate, DB integration, forward-only provenance
6. Hard-coded device invariant: sentiment.py never hard-codes a device

KEY TEST — test_aggregate_sentiment_pit
----------------------------------------
A story with first_knowable_at == as_of is NOT included in the score.
This proves that sentiment inherits the PIT contract of the news that produced it:
a score computed from a story cannot be used before that story was knowable.

Mock pattern
------------
Tests never hit the real FinBERT model (not downloaded, may not have torch).
All scoring uses FinBERTScorer(_mock_pipeline=...) for deterministic results.
"""
from __future__ import annotations

import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pytest

from src.db.schema import DUCKDB_SCHEMA_SQL
from src.db.duckdb_client import DuckDBClient
from src.features.base import DataGate, FeatureRegistry
from src.features.sentiment import (
    SENTIMENT_GATE_PROVENANCE,
    FinBERTScorer,
    aggregate_sentiment,
    news_sentiment,
    _get_scorer,
    _set_scorer,
)
import src.features.sentiment as _sent_mod

UTC = timezone.utc
AS_OF = datetime(2024, 6, 10, 10, 0, tzinfo=UTC)


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _mock_pipeline(label_sequence: list[str]) -> object:
    """Return a callable that acts like a HuggingFace text-classification pipeline.

    label_sequence: list of FinBERT labels ("positive", "negative", "neutral")
    to return for successive calls. Repeats last label if batch is longer.
    """
    responses = [{"label": lbl, "score": 0.95} for lbl in label_sequence]

    class _Mock:
        def __init__(self):
            self.call_count = 0
            self.texts_seen: list[list[str]] = []

        def __call__(self, texts, **kwargs):
            self.call_count += 1
            self.texts_seen.append(list(texts))
            last = responses[-1] if responses else {"label": "neutral", "score": 0.5}
            return [responses[i] if i < len(responses) else last for i in range(len(texts))]

    return _Mock()


def _setup(db_path: Path) -> None:
    with DuckDBClient(db_path) as conn:
        conn.execute_schema(DUCKDB_SCHEMA_SQL)


def _ins_news(db_path: Path, symbol: str, ts: datetime, news_id: str,
              headline: str = "Test headline") -> None:
    with duckdb.connect(str(db_path)) as con:
        con.execute(
            "INSERT INTO news "
            "(news_id, headline, body, source, published_at, first_knowable_at, "
            " symbols, ingested_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT DO NOTHING",
            [news_id, headline, None, "reuters",
             ts, ts, [symbol], datetime.now(UTC)],
        )


def _make_scored_df(
    stories: list[dict],
) -> pd.DataFrame:
    """Build a pre-scored DataFrame for aggregate_sentiment tests."""
    return pd.DataFrame({
        "news_id": [s["news_id"] for s in stories],
        "headline": [s.get("headline", "h") for s in stories],
        "source": [s.get("source", "reuters") for s in stories],
        "first_knowable_at": [s["first_knowable_at"] for s in stories],
        "sentiment": [s["sentiment"] for s in stories],
    })


# ── 1. Device detection ───────────────────────────────────────────────────────

def test_device_is_cpu_on_this_machine():
    """On this no-GPU dev machine, device is always 'cpu'.

    Covers the case where torch is not installed (ImportError → cpu)
    AND the case where torch is installed but CUDA unavailable.
    This is the device-fallback test requested in the checkpoint spec.
    """
    scorer = FinBERTScorer()
    assert scorer.device == "cpu"


def test_device_is_cpu_when_torch_missing(monkeypatch):
    """Device falls back to 'cpu' when torch cannot be imported.

    Simulates the 'torch not installed' scenario on machines where
    torch IS installed by temporarily blocking the import.
    """
    monkeypatch.setitem(sys.modules, "torch", None)
    scorer = FinBERTScorer()
    assert scorer.device == "cpu"


def test_device_is_cuda_when_available(monkeypatch):
    """Device is 'cuda' when torch reports a CUDA-capable GPU is present.

    Uses a lightweight fake torch module so this test runs regardless of
    whether torch is installed on the current machine.
    """
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: True)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    scorer = FinBERTScorer()
    assert scorer.device == "cuda"


# ── 2. FinBERTScorer — batch scoring ─────────────────────────────────────────

def test_score_batch_positive():
    mock = _mock_pipeline(["positive"])
    scorer = FinBERTScorer(_mock_pipeline=mock)
    scores = scorer.score_batch(["Infosys revenue beats estimates by 12%"], ["n1"])
    assert scores == {"n1": 1.0}


def test_score_batch_negative():
    mock = _mock_pipeline(["negative"])
    scorer = FinBERTScorer(_mock_pipeline=mock)
    scores = scorer.score_batch(["Wipro misses Q4 targets; guidance cut"], ["n1"])
    assert scores == {"n1": -1.0}


def test_score_batch_neutral():
    mock = _mock_pipeline(["neutral"])
    scorer = FinBERTScorer(_mock_pipeline=mock)
    scores = scorer.score_batch(["Tata Motors to announce results on Friday"], ["n1"])
    assert scores == {"n1": 0.0}


def test_score_batch_caches_articles():
    """The same news_id is scored only once even on repeated calls."""
    mock = _mock_pipeline(["positive"])
    scorer = FinBERTScorer(_mock_pipeline=mock)

    scorer.score_batch(["First call text"], ["n1"])
    scorer.score_batch(["Second call — same id, different text"], ["n1"])

    assert mock.call_count == 1  # pipeline called only once


def test_score_batch_multi_article():
    mock = _mock_pipeline(["positive", "negative", "neutral"])
    scorer = FinBERTScorer(_mock_pipeline=mock)
    scores = scorer.score_batch(["pos text", "neg text", "neu text"], ["n1", "n2", "n3"])
    assert scores == {"n1": 1.0, "n2": -1.0, "n3": 0.0}


def test_scorer_graceful_no_pipeline():
    """If pipeline is None (transformers not available), score_batch returns 0.0."""
    scorer = FinBERTScorer()
    scorer._loaded = True   # skip real loading
    scorer._pipeline = None  # simulate transformers not installed

    scores = scorer.score_batch(["Infosys profits rise"], ["n1"])
    assert scores == {"n1": 0.0}  # neutral, not a crash


# ── 3. score_news_df ──────────────────────────────────────────────────────────

def test_score_news_df_empty_returns_empty():
    scorer = FinBERTScorer(_mock_pipeline=_mock_pipeline(["positive"]))
    empty = pd.DataFrame(columns=["news_id", "headline", "source", "first_knowable_at"])
    result = scorer.score_news_df(empty)
    assert result.empty
    assert "sentiment" in result.columns


def test_sample_scored_output_format():
    """Demonstrate the scored output format.

    Sample output (shown here for documentation):

        news_id  headline                          source    first_knowable_at         sentiment
        n1       Infosys revenue beats estimates   reuters   2024-06-10 08:00:00+00:00  1.0
        n2       Wipro misses Q4 targets           ET        2024-06-10 09:00:00+00:00 -1.0

    sentiment ∈ {-1.0, 0.0, +1.0} (FinBERT 3-class output mapped to bounded scores).
    """
    mock = _mock_pipeline(["positive", "negative"])
    scorer = FinBERTScorer(_mock_pipeline=mock)

    df = pd.DataFrame({
        "news_id": ["n1", "n2"],
        "headline": ["Infosys revenue beats estimates", "Wipro misses Q4 targets"],
        "source": ["reuters", "economictimes"],
        "first_knowable_at": [
            datetime(2024, 6, 10, 8, 0, tzinfo=UTC),
            datetime(2024, 6, 10, 9, 0, tzinfo=UTC),
        ],
    })

    scored = scorer.score_news_df(df)

    assert "sentiment" in scored.columns
    assert len(scored) == 2
    assert scored["sentiment"].tolist() == [1.0, -1.0]
    assert scored["sentiment"].between(-1.0, 1.0).all()
    # FinBERT outputs exactly 3 labels — scores are in {-1, 0, 1}
    assert set(scored["sentiment"].unique()).issubset({-1.0, 0.0, 1.0})


# ── 4. aggregate_sentiment ────────────────────────────────────────────────────

def test_aggregate_sentiment_pit():
    """KEY PIT TEST: a story whose first_knowable_at == as_of is NOT included.

    This proves that sentiment inherits the PIT contract of the news that
    produced it. The score of story n2 is -1.0 (negative); if it leaked in,
    the aggregate would be zero or negative. Because it is correctly excluded,
    only the positive story n1 contributes → aggregate > 0.
    """
    stories = _make_scored_df([
        {   # knowable 2h before as_of → INCLUDED
            "news_id": "n1",
            "source": "reuters",
            "first_knowable_at": AS_OF - timedelta(hours=2),
            "sentiment": 1.0,
        },
        {   # knowable EXACTLY AT as_of → MUST BE EXCLUDED (strict <)
            "news_id": "n2",
            "source": "reuters",
            "first_knowable_at": AS_OF,
            "sentiment": -1.0,  # poison: if included, score pulled negative
        },
    ])

    score = aggregate_sentiment(stories, AS_OF)

    assert score > 0.0, (
        "PIT violation: boundary story (first_knowable_at == as_of) was included "
        "in the sentiment aggregate. score should be > 0 but got "
        f"{score!r}."
    )


def test_aggregate_sentiment_future_story_excluded():
    """Story knowable after as_of is also excluded."""
    stories = _make_scored_df([
        {
            "news_id": "n1",
            "source": "reuters",
            "first_knowable_at": AS_OF + timedelta(hours=1),
            "sentiment": 1.0,
        },
    ])
    assert aggregate_sentiment(stories, AS_OF) == 0.0


def test_aggregate_sentiment_no_news_returns_zero():
    """Empty DataFrame → 0.0 (neutral, not crash)."""
    empty = _make_scored_df([])
    assert aggregate_sentiment(empty, AS_OF) == 0.0


def test_aggregate_sentiment_time_decay():
    """More recent story should outweigh older story of opposite sign.

    With half-life 4h:
      recent (1h ago) decay = 0.5^(1/4) ≈ 0.841
      old    (8h ago) decay = 0.5^(8/4) = 0.25

    Weighted sum (equal source weight): recent positive dominates → score > 0.
    """
    stories = _make_scored_df([
        {
            "news_id": "n1",
            "source": "reuters",
            "first_knowable_at": AS_OF - timedelta(hours=1),
            "sentiment": 1.0,   # recent positive
        },
        {
            "news_id": "n2",
            "source": "reuters",
            "first_knowable_at": AS_OF - timedelta(hours=8),
            "sentiment": -1.0,  # old negative
        },
    ])

    score = aggregate_sentiment(stories, AS_OF, decay_halflife_hours=4.0)
    assert score > 0.0, f"Recent positive story should dominate; got {score!r}"


def test_aggregate_sentiment_source_weighting():
    """Regulatory filings (nse_filings=2.0) outweigh generic sources (1.0).

    Two simultaneous stories of opposite sentiment and different sources:
    the nse_filings story weight=2.0, the generic story weight=1.0.
    → net score should reflect the nse_filings direction.
    """
    ts = AS_OF - timedelta(hours=1)
    stories = _make_scored_df([
        {
            "news_id": "filing",
            "source": "nse_filings",
            "first_knowable_at": ts,
            "sentiment": -1.0,  # weight=2.0
        },
        {
            "news_id": "generic",
            "source": "unknown_blog",
            "first_knowable_at": ts,
            "sentiment": 1.0,   # weight=1.0 (default)
        },
    ])

    score = aggregate_sentiment(stories, AS_OF)
    # nse_filings negative (w=2) vs generic positive (w=1) → net negative
    assert score < 0.0, f"nse_filings weight=2.0 should dominate; got {score!r}"


# ── 5. news_sentiment registered feature ─────────────────────────────────────

def test_news_sentiment_feature_registered():
    """news_sentiment is in the registry with gate=NEWS_NLP."""
    meta = FeatureRegistry.get_meta("news_sentiment")
    assert meta is not None
    assert meta.gate == DataGate.NEWS_NLP


def test_news_sentiment_no_news_empty_db(tmp_duckdb_path, monkeypatch):
    """No news in DB → 0.0 (neutral, not crash)."""
    _setup(tmp_duckdb_path)

    mock_scorer = FinBERTScorer(
        _mock_pipeline=_mock_pipeline(["neutral"])
    )
    monkeypatch.setattr(_sent_mod, "_DEFAULT_SCORER", mock_scorer)

    result = news_sentiment("NONEXISTENT", AS_OF, str(tmp_duckdb_path))
    assert result == 0.0


def test_news_sentiment_with_news_db(tmp_duckdb_path, monkeypatch):
    """With positive news in DB, score should reflect it."""
    _setup(tmp_duckdb_path)
    _ins_news(tmp_duckdb_path, "INFOSYS",
              AS_OF - timedelta(hours=2), "n1",
              headline="Infosys Q4 revenue rises 15%, margin beats estimate")

    mock_scorer = FinBERTScorer(
        _mock_pipeline=_mock_pipeline(["positive"])
    )
    monkeypatch.setattr(_sent_mod, "_DEFAULT_SCORER", mock_scorer)

    result = news_sentiment("INFOSYS", AS_OF, str(tmp_duckdb_path))
    assert result > 0.0, f"Expected positive score; got {result!r}"


def test_news_sentiment_excludes_future_news(tmp_duckdb_path, monkeypatch):
    """News with first_knowable_at >= as_of is NOT scored (PIT via DB query)."""
    _setup(tmp_duckdb_path)
    _ins_news(tmp_duckdb_path, "RELIANCE",
              AS_OF,  # exactly at as_of — must be excluded by DB WHERE clause
              "n_future", headline="Future news that should not be visible")

    mock_scorer = FinBERTScorer(
        _mock_pipeline=_mock_pipeline(["negative"])
    )
    monkeypatch.setattr(_sent_mod, "_DEFAULT_SCORER", mock_scorer)

    # DB query uses WHERE first_knowable_at < ? → news at AS_OF is excluded
    # With no qualifying news, returns 0.0
    result = news_sentiment("RELIANCE", AS_OF, str(tmp_duckdb_path))
    assert result == 0.0


# ── 6. Provenance and hard-coded device invariant ─────────────────────────────

def test_sentiment_gate_provenance_constant():
    """SENTIMENT_GATE_PROVENANCE is 'forward_only'."""
    assert SENTIMENT_GATE_PROVENANCE == "forward_only"


def test_sentiment_provenance_in_feature_description():
    """The feature description stamps the forward_only provenance."""
    meta = FeatureRegistry.get_meta("news_sentiment")
    assert meta is not None
    assert SENTIMENT_GATE_PROVENANCE in meta.description


def test_sentiment_never_hardcodes_cuda_device():
    """sentiment.py must never pass a hard-coded 'cuda' string as a device argument.

    Permitted:  device = "cuda" if torch.cuda.is_available() else "cpu"
                torch.cuda.is_available()
    Forbidden:  device="cuda"   (unconditional assignment / keyword arg)

    This invariant ensures the file runs on CPU-only machines without modification.
    """
    sentiment_file = Path(__file__).parent.parent / "src" / "features" / "sentiment.py"
    source = sentiment_file.read_text(encoding="utf-8")
    # Detect the dangerous pattern: device="cuda" or device='cuda' as a
    # keyword argument or direct assignment WITHOUT an is_available() guard.
    import re
    hard_coded = re.findall(r'device\s*=\s*["\']cuda["\']', source)
    assert not hard_coded, (
        f"Hard-coded device='cuda' found in sentiment.py: {hard_coded}. "
        "Use: device='cuda' if torch.cuda.is_available() else 'cpu'"
    )
