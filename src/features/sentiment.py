"""FinBERT sentiment scorer for financial news (Checkpoint 2.2).

Architecture
------------
- FinBERTScorer  — batch scorer; lazy-import transformers; score cache by news_id
- aggregate_sentiment — PIT-safe aggregation; source-weighted + time-decayed
- news_sentiment  — registered feature (gate=NEWS_NLP)

Device selection
----------------
CUDA if available, else CPU. Detected at runtime via torch.cuda.is_available().
Never hard-coded. Runs correctly (just slower) on this no-GPU dev machine;
runs faster on the RTX 3060.

Graceful degradation
--------------------
If transformers or torch is not installed, scoring returns 0.0 (neutral) with
a WARNING log. The rest of the pipeline is unaffected. Install when ready:
    pip install transformers torch

Forward-validity (DESIGN.md §3)
--------------------------------
SENTIMENT_GATE_PROVENANCE = "forward_only"

The news_sentiment feature is meaningful only from when news ingestion runs.
Without historical news backfill, it returns 0.0 for historical dates — this
is honest, not an error. Every ExperimentTracker run using this feature MUST
record this provenance stamp so no backtest silently treats sentiment as
historically validated.

PIT contract
------------
aggregate_sentiment() filters on first_knowable_at < as_of (strict <).
A story's sentiment is invisible before its first_knowable_at — guaranteed
at two levels: (1) the DB query in news_sentiment uses WHERE first_knowable_at < ?;
(2) aggregate_sentiment re-checks the same condition as a belt-and-suspenders guard.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime, timedelta, timezone
from typing import Callable

import duckdb
import pandas as pd

from src.features.base import DataGate, FeatureRegistry

_LOG = logging.getLogger(__name__)
UTC = timezone.utc

# Provenance stamp for every backtest run using this feature.
# Value is intentionally human-readable and grep-able.
SENTIMENT_GATE_PROVENANCE = "forward_only"

# Default source trust weights (relative; higher = more trusted).
# Research basis: regulatory filings are authoritative; social feeds noisy.
_SOURCE_WEIGHTS: dict[str, float] = {
    "sebi": 2.5,
    "nse_filings": 2.0,
    "bse_filings": 2.0,
    "reuters": 1.5,
    "moneycontrol": 1.0,
    "economictimes": 1.0,
    "livemint": 1.0,
    "businessstandard": 1.0,
}

# FinBERT label → bounded score ∈ {-1.0, 0.0, +1.0}
_LABEL_TO_SCORE: dict[str, float] = {
    "positive": 1.0,
    "negative": -1.0,
    "neutral": 0.0,
}


# ── FinBERTScorer ─────────────────────────────────────────────────────────────

class FinBERTScorer:
    """Batch financial news sentiment scorer using ProsusAI/finbert.

    Design
    ------
    - Lazy model loading: transformers not imported at module level
    - Device detection at load time: CUDA → CPU fallback, never hard-coded
    - Score cache keyed on news_id (dedup hash): same article never re-scored
    - Mock injection via _mock_pipeline: tests never hit the network

    Usage
    -----
        scorer = FinBERTScorer()
        result = scorer.score_batch(["Infosys Q4 revenue beats estimates"], ["n1"])
        # → {"n1": 1.0}

    Test injection
    --------------
        scorer = FinBERTScorer(_mock_pipeline=my_mock_callable)
    """

    MODEL_NAME = "ProsusAI/finbert"

    def __init__(
        self,
        model_name: str | None = None,
        *,
        cache_dir: str | None = None,
        _mock_pipeline: Callable | None = None,
    ) -> None:
        self._model_name = model_name or self.MODEL_NAME
        self._cache_dir = cache_dir
        self._mock_pipeline = _mock_pipeline
        self._pipeline: Callable | None = None
        self._loaded: bool = False
        self._score_cache: dict[str, float] = {}  # news_id → score ∈ [-1, 1]

    @property
    def device(self) -> str:
        """'cuda' if a CUDA-capable GPU is available, else 'cpu'.

        Never returns a hard-coded device string. Falls back to 'cpu'
        gracefully if torch is not installed or CUDA init fails.
        """
        try:
            import torch  # noqa: PLC0415
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    def _ensure_loaded(self) -> None:
        """Lazy-load the FinBERT pipeline. Called on first scoring request."""
        if self._loaded:
            return
        self._loaded = True

        if self._mock_pipeline is not None:
            self._pipeline = self._mock_pipeline
            return

        try:
            import torch  # noqa: PLC0415
            from transformers import pipeline  # noqa: PLC0415

            # Use integer device index: 0=CUDA, -1=CPU
            device_idx = 0 if torch.cuda.is_available() else -1
            self._pipeline = pipeline(
                "text-classification",
                model=self._model_name,
                device=device_idx,
                cache_dir=self._cache_dir,
            )
        except ImportError:
            _LOG.warning(
                "transformers or torch not installed — news_sentiment returns 0.0 "
                "(neutral). Install with: pip install transformers torch"
            )
            self._pipeline = None
        except Exception as exc:
            _LOG.warning(
                "FinBERT model could not be loaded (%s) — news_sentiment returns "
                "0.0 (neutral). Ensure internet access for first-time weight download.",
                exc,
            )
            self._pipeline = None

    def score_batch(self, texts: list[str], news_ids: list[str]) -> dict[str, float]:
        """Score texts and return {news_id: score ∈ [-1.0, 0.0, +1.0]}.

        Already-cached articles are not re-scored. If the pipeline is
        unavailable, all scores are 0.0 (neutral — not a crash).
        """
        if not texts:
            return {}

        self._ensure_loaded()

        uncached_ids: list[str] = []
        uncached_texts: list[str] = []
        for nid, text in zip(news_ids, texts):
            if nid not in self._score_cache:
                uncached_ids.append(nid)
                uncached_texts.append(text)

        if uncached_ids:
            if self._pipeline is not None:
                try:
                    raw = self._pipeline(uncached_texts, truncation=True, max_length=512)
                    for nid, r in zip(uncached_ids, raw):
                        label = r.get("label", "neutral").lower()
                        self._score_cache[nid] = _LABEL_TO_SCORE.get(label, 0.0)
                except Exception as exc:
                    _LOG.warning("FinBERT scoring failed (%s); returning 0.0 for batch", exc)
                    for nid in uncached_ids:
                        self._score_cache[nid] = 0.0
            else:
                for nid in uncached_ids:
                    self._score_cache[nid] = 0.0

        return {nid: self._score_cache.get(nid, 0.0) for nid in news_ids}

    def score_news_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score a news DataFrame; adds a 'sentiment' column ∈ [-1.0, 1.0].

        Required input columns: news_id, headline.
        Optional: body (concatenated with headline for richer context).
        Returns the same DataFrame with 'sentiment' appended.
        """
        if df.empty:
            return df.assign(sentiment=pd.Series(dtype=float))

        headlines = df["headline"].fillna("").tolist()
        bodies = (
            df["body"].fillna("").tolist()
            if "body" in df.columns
            else [""] * len(df)
        )
        texts = [f"{h} {b}".strip() for h, b in zip(headlines, bodies)]
        news_ids = df["news_id"].tolist()

        scores = self.score_batch(texts, news_ids)
        return df.assign(sentiment=[scores.get(nid, 0.0) for nid in news_ids])


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_sentiment(
    scored_df: pd.DataFrame,
    as_of: datetime,
    *,
    decay_halflife_hours: float = 4.0,
    source_weights: dict[str, float] | None = None,
) -> float:
    """Aggregate per-article sentiment into a single per-stock score.

    PIT guarantee (belt-and-suspenders)
    ------------------------------------
    Only articles with first_knowable_at < as_of are included. This re-checks
    the same condition already enforced by the DB query in news_sentiment, so
    bad data cannot leak through via downstream DataFrame manipulation.

    Weighting
    ---------
    Weight per article = source_weight × decay_factor
    where decay_factor = 0.5 ^ (age_hours / decay_halflife_hours).

    Parameters
    ----------
    scored_df : DataFrame with columns: news_id, source, first_knowable_at, sentiment
    as_of : reference datetime for PIT filtering and age calculation
    decay_halflife_hours : exponential decay half-life (default 4h)
    source_weights : override/extend default source weights

    Returns
    -------
    float ∈ [-1.0, 1.0], or 0.0 (neutral) if no qualifying articles.
    """
    if scored_df.empty:
        return 0.0

    as_of_utc = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)

    # Normalize first_knowable_at to UTC-aware
    fka = pd.to_datetime(scored_df["first_knowable_at"])
    if fka.dt.tz is None:
        fka = fka.dt.tz_localize(UTC)
    else:
        fka = fka.dt.tz_convert(UTC)

    # PIT filter: strict <
    pit_mask = fka < as_of_utc
    df = scored_df[pit_mask].copy()

    if df.empty:
        return 0.0

    fka_filtered = fka[pit_mask]
    ages_h = (as_of_utc - fka_filtered).dt.total_seconds() / 3600.0
    decay = (0.5 ** (ages_h / decay_halflife_hours)).values

    sw = _SOURCE_WEIGHTS.copy()
    if source_weights:
        sw.update(source_weights)
    src_w = df["source"].map(lambda s: sw.get(s, 1.0)).values

    combined = decay * src_w
    total_w = combined.sum()
    if total_w <= 0:
        return 0.0

    return float((df["sentiment"].values * combined).sum() / total_w)


# ── Module-level singleton for the registered feature ─────────────────────────

_DEFAULT_SCORER: FinBERTScorer | None = None


def _get_scorer() -> FinBERTScorer:
    global _DEFAULT_SCORER
    if _DEFAULT_SCORER is None:
        _DEFAULT_SCORER = FinBERTScorer()
    return _DEFAULT_SCORER


def _set_scorer(scorer: FinBERTScorer | None) -> None:
    """Replace the module-level scorer. Used in tests."""
    global _DEFAULT_SCORER
    _DEFAULT_SCORER = scorer


# ── Registered feature ────────────────────────────────────────────────────────

@FeatureRegistry.register(
    "news_sentiment",
    DataGate.NEWS_NLP,
    "FinBERT time-decayed source-weighted sentiment ∈ [-1, 1]; "
    f"provenance={SENTIMENT_GATE_PROVENANCE}",
)
def news_sentiment(symbol: str, as_of: datetime, db_path: str) -> float | None:
    """Aggregated FinBERT sentiment for symbol at as_of.

    FORWARD-VALID ONLY (SENTIMENT_GATE_PROVENANCE = 'forward_only').
    Without historical news backfill, returns 0.0 for historical dates — this
    is an honest neutral signal, not an error. Record provenance in every
    ExperimentTracker run that uses this feature.

    Returns 0.0 (neutral) when no news is available. Never returns None.
    """
    as_of_utc = as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
    lookback = as_of_utc - timedelta(hours=24)

    with duckdb.connect(db_path, read_only=True, config={'threads': '1'}) as con:
        df = con.execute(
            "SELECT news_id, headline, body, source, first_knowable_at "
            "FROM news "
            "WHERE first_knowable_at < ? "
            "  AND first_knowable_at >= ? "
            "  AND list_contains(symbols, ?)",
            [as_of_utc, lookback, symbol],
        ).df()

    if df.empty:
        return 0.0

    scored = _get_scorer().score_news_df(df)
    return aggregate_sentiment(scored, as_of)
