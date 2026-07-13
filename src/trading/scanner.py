"""Scanner — progressive funnel from universe to ranked shortlist.

RANKING PRINCIPLE: candidates are ranked by EXPECTED AFTER-COST RISK-ADJUSTED
VALUE, not by raw model confidence. A high-confidence tiny-move candidate that
costs more than it makes must rank BELOW a moderate-confidence larger-move
candidate that nets positive after costs. Ranking by raw confidence is a
structural mistake — this scanner avoids it by design.

After-cost expected value (EV):
    ev_pct = confidence × expected_move_pct
           - (1 − confidence) × stop_loss_pct
           - estimated_round_trip_cost_pct

Only candidates with ev_pct > 0 appear on the shortlist.
Empty shortlist is a first-class valid outcome — not-trading is a signal, not an error.

Stage 1 (cheap, O(n)): filter on direction, confidence, liquidity, volume activity,
    and minimum expected move — eliminates obvious non-candidates without scoring.
Stage 2 (scored, O(survivors)): compute after-cost EV using CostSimulator;
    rank descending; truncate at max_shortlist_size.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal

from src.config import ScannerConfig
from src.trading._types import PredictionDirection
from src.trading.costs import CostSimulator


@dataclass(frozen=True)
class PredictionInput:
    """The contract the real model must satisfy when it replaces mock predictions.

    All scanner tests use mock instances of this class. The trained model will
    produce real PredictionInput objects that plug in without any scanner changes.

    Fields:
      symbol              — NSE ticker
      direction           — LONG, SHORT, or NO_TRADE (model's top prediction)
      confidence          — P(direction correct), strictly in (0, 1)
      expected_move_pct   — predicted % price move if direction hits (always > 0)
      stop_loss_pct       — expected % loss if direction misses (always > 0)
      entry_price         — current quoted price (Decimal)
      avg_daily_volume_cr — average daily traded value in crores (liquidity gate)
      volume_ratio        — today's volume / 20-day average (activity gate)
    """

    symbol: str
    direction: PredictionDirection
    confidence: float
    expected_move_pct: float
    stop_loss_pct: float
    entry_price: Decimal
    avg_daily_volume_cr: float
    volume_ratio: float


@dataclass(frozen=True)
class ScanResult:
    """One ranked candidate from the scanner shortlist.

    after_cost_ev_pct is the primary ranking key — always positive for shortlisted
    candidates. gross_ev_pct and estimated_cost_pct show the breakdown.
    """

    rank: int
    prediction: PredictionInput
    gross_ev_pct: float        # confidence × move_pct − (1−confidence) × stop_pct
    estimated_cost_pct: float  # round-trip cost as % of position value
    after_cost_ev_pct: float   # gross_ev_pct − estimated_cost_pct (ranking key)
    reasoning: str             # human-readable decomposition for logging


@dataclass(frozen=True)
class FilteredOut:
    """A candidate excluded from the shortlist, with the reason recorded."""

    symbol: str
    direction: PredictionDirection
    reason: str


@dataclass(frozen=True)
class ScanOutput:
    """Full output of one scanner run.

    shortlist may be empty — that is a valid, first-class outcome meaning
    no candidate cleared the after-cost EV bar today.
    """

    shortlist: list[ScanResult]
    filtered_out: list[FilteredOut]
    scanned_at: datetime

    @property
    def has_candidates(self) -> bool:
        """False when no candidate cleared the after-cost EV bar."""
        return bool(self.shortlist)

    @property
    def best(self) -> ScanResult | None:
        return self.shortlist[0] if self.shortlist else None


class Scanner:
    """Two-stage funnel: cheap pre-filters then after-cost EV scoring and ranking.

    Instantiate once per session with a CostSimulator and ScannerConfig.
    Call scan() each time a new set of model predictions arrives.
    """

    def __init__(self, cost_simulator: CostSimulator, scanner_config: ScannerConfig) -> None:
        self._cost_sim = cost_simulator
        self._cfg = scanner_config

    def scan(
        self,
        predictions: list[PredictionInput],
        scanned_at: datetime | None = None,
    ) -> ScanOutput:
        """Run the full funnel. Returns a ScanOutput; empty shortlist is valid."""
        ts = scanned_at or datetime.now(timezone.utc)
        all_filtered: list[FilteredOut] = []

        survivors, pre_filtered = self._apply_prefilters(predictions)
        all_filtered.extend(pre_filtered)

        ranked, ev_filtered = self._score_and_rank(survivors)
        all_filtered.extend(ev_filtered)

        return ScanOutput(
            shortlist=ranked[: self._cfg.max_shortlist_size],
            filtered_out=all_filtered,
            scanned_at=ts,
        )

    # ── Stage 1: cheap pre-filters (no scoring, no CostSimulator calls) ──────

    def _apply_prefilters(
        self,
        predictions: list[PredictionInput],
    ) -> tuple[list[PredictionInput], list[FilteredOut]]:
        survivors: list[PredictionInput] = []
        filtered: list[FilteredOut] = []
        for pred in predictions:
            reason = self._first_failing_filter(pred)
            if reason is None:
                survivors.append(pred)
            else:
                filtered.append(FilteredOut(
                    symbol=pred.symbol,
                    direction=pred.direction,
                    reason=reason,
                ))
        return survivors, filtered

    def _first_failing_filter(self, pred: PredictionInput) -> str | None:
        """Return the first failing filter reason, or None if all pass."""
        if pred.direction == "NO_TRADE":
            return "direction=NO_TRADE: model says stay out"

        if pred.confidence < self._cfg.min_confidence:
            return (
                f"confidence={pred.confidence:.3f} below "
                f"min={self._cfg.min_confidence:.3f}"
            )

        if pred.avg_daily_volume_cr < self._cfg.min_avg_daily_volume_cr:
            return (
                f"liquidity={pred.avg_daily_volume_cr:.2f}cr below "
                f"min={self._cfg.min_avg_daily_volume_cr:.2f}cr"
            )

        if pred.volume_ratio < self._cfg.min_volume_ratio:
            return (
                f"volume_ratio={pred.volume_ratio:.2f} below "
                f"min={self._cfg.min_volume_ratio:.2f} (low activity today)"
            )

        if pred.expected_move_pct < self._cfg.min_expected_move_pct:
            return (
                f"expected_move={pred.expected_move_pct:.3f}% below "
                f"min={self._cfg.min_expected_move_pct:.3f}%"
            )

        return None

    # ── Stage 2: after-cost EV scoring and ranking ────────────────────────────

    def _score_and_rank(
        self,
        predictions: list[PredictionInput],
    ) -> tuple[list[ScanResult], list[FilteredOut]]:
        """Score each survivor; split into positive-EV ranked list and EV-excluded list."""
        _Scored = tuple[float, PredictionInput, float, float]  # (ev, pred, gross, cost)
        scored: list[_Scored] = []
        ev_filtered: list[FilteredOut] = []

        for pred in predictions:
            gross_ev = (
                pred.confidence * pred.expected_move_pct
                - (1.0 - pred.confidence) * pred.stop_loss_pct
            )
            cost_pct = self._estimate_cost_pct(pred.entry_price)
            after_cost_ev = gross_ev - cost_pct

            if after_cost_ev <= 0.0:
                ev_filtered.append(FilteredOut(
                    symbol=pred.symbol,
                    direction=pred.direction,
                    reason=(
                        f"after_cost_ev={after_cost_ev:.4f}% ≤ 0 "
                        f"(gross={gross_ev:.4f}% − cost={cost_pct:.4f}%)"
                    ),
                ))
            else:
                scored.append((after_cost_ev, pred, gross_ev, cost_pct))

        # Rank descending by after-cost EV — NOT by raw confidence
        scored.sort(key=lambda t: t[0], reverse=True)

        ranked = [
            ScanResult(
                rank=i + 1,
                prediction=pred,
                gross_ev_pct=gross_ev,
                estimated_cost_pct=cost_pct,
                after_cost_ev_pct=after_cost_ev,
                reasoning=(
                    f"rank={i + 1}: ev={after_cost_ev:.4f}% "
                    f"(gross={gross_ev:.4f}% − cost={cost_pct:.4f}%, "
                    f"conf={pred.confidence:.2f}, dir={pred.direction})"
                ),
            )
            for i, (after_cost_ev, pred, gross_ev, cost_pct) in enumerate(scored)
        ]

        return ranked, ev_filtered

    def _estimate_cost_pct(self, entry_price: Decimal) -> float:
        """Estimate round-trip cost as % of position value via CostSimulator.

        Uses the actual entry_price and the representative_qty from config.
        Sets exit_price = entry_price (zero gross move) to isolate cost only.
        Direction is LONG for cost estimation — with entry=exit at the same price,
        STT and stamp duty are symmetric across LONG and SHORT, so it does not matter.
        """
        qty = self._cfg.representative_qty
        result = self._cost_sim.compute(
            symbol="_COST_EST_",
            qty=qty,
            direction="LONG",
            entry_price=entry_price,
            exit_price=entry_price,
        )
        position_value = Decimal(qty) * entry_price
        return float(result.costs.total / position_value * 100)
