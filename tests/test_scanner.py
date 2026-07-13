"""Tests for the scanner / candidate funnel (src/trading/scanner.py).

Uses mock PredictionInput instances — no trained model or real market data needed.

Test groups:
  TestPreFilters          — each cheap filter correctly eliminates candidates
  TestRankingByAfterCostEV — THE KEY STRUCTURAL GUARANTEE: rank by EV, not confidence
  TestEmptyShortlist      — not-trading is a first-class valid outcome
  TestShortlistSizeLimit  — capped at max_shortlist_size; best candidates retained
  TestScanOutput          — output structure: filtered_out, ranks, reasoning
  TestScannerConfig       — ScannerConfig loads from config.toml; validation errors

Cost arithmetic for entry_price=₹1000, representative_qty=50 (used in key tests):
  position_value = 50,000
  brokerage_buy  = min(0.03% × 50000, ₹20) = ₹15 (uncapped)
  brokerage_sell = ₹15
  STT            = 0.025% × 50000 = ₹12.50   (sell leg)
  exchange       = 0.00297% × 100000 = ₹2.97  (total turnover)
  SEBI           = 0.0001% × 100000 = ₹0.10
  GST            = 18% × (15+15+2.97+0.10) = ₹5.9526
  stamp          = 0.003% × 50000 = ₹1.50    (buy leg)
  total          ≈ ₹53.02  →  cost_pct ≈ 0.1060%
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.config import CostConfig, ScannerConfig
from src.trading.costs import CostSimulator
from src.trading.scanner import (
    FilteredOut,
    PredictionInput,
    ScanOutput,
    ScanResult,
    Scanner,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scanner(
    max_shortlist_size: int = 10,
    min_confidence: float = 0.55,
    min_expected_move_pct: float = 0.10,
    min_avg_daily_volume_cr: float = 1.0,
    min_volume_ratio: float = 0.30,
    representative_qty: int = 50,
) -> Scanner:
    cfg = ScannerConfig(
        max_shortlist_size=max_shortlist_size,
        min_confidence=min_confidence,
        min_expected_move_pct=min_expected_move_pct,
        min_avg_daily_volume_cr=min_avg_daily_volume_cr,
        min_volume_ratio=min_volume_ratio,
        representative_qty=representative_qty,
    )
    return Scanner(CostSimulator(CostConfig()), cfg)


def _pred(
    symbol: str = "TEST",
    direction: str = "LONG",
    confidence: float = 0.70,
    expected_move_pct: float = 0.50,
    stop_loss_pct: float = 0.20,
    entry_price: str = "1000",
    avg_daily_volume_cr: float = 5.0,
    volume_ratio: float = 1.2,
) -> PredictionInput:
    return PredictionInput(
        symbol=symbol,
        direction=direction,  # type: ignore[arg-type]
        confidence=confidence,
        expected_move_pct=expected_move_pct,
        stop_loss_pct=stop_loss_pct,
        entry_price=Decimal(entry_price),
        avg_daily_volume_cr=avg_daily_volume_cr,
        volume_ratio=volume_ratio,
    )


# ── Pre-filter tests ──────────────────────────────────────────────────────────

class TestPreFilters:
    """Cheap filters eliminate candidates before any EV scoring occurs."""

    def test_no_trade_direction_filtered(self):
        out = _scanner().scan([_pred(direction="NO_TRADE")])
        assert not out.has_candidates
        assert len(out.filtered_out) == 1
        assert "NO_TRADE" in out.filtered_out[0].reason

    def test_low_confidence_filtered(self):
        out = _scanner(min_confidence=0.65).scan([_pred(confidence=0.62)])
        assert not out.has_candidates
        assert any("confidence" in f.reason.lower() for f in out.filtered_out)

    def test_confidence_at_threshold_passes_prefilter(self):
        # confidence == min_confidence should not be pre-filtered for confidence
        out = _scanner(min_confidence=0.60).scan([
            _pred(confidence=0.60, expected_move_pct=0.80, stop_loss_pct=0.20)
        ])
        assert not any(
            "confidence" in f.reason.lower() for f in out.filtered_out
        ), "Candidate at the confidence threshold should not be pre-filtered for confidence"

    def test_low_liquidity_filtered(self):
        out = _scanner(min_avg_daily_volume_cr=2.0).scan([
            _pred(avg_daily_volume_cr=1.5)
        ])
        assert not out.has_candidates
        assert any("liquidity" in f.reason.lower() for f in out.filtered_out)

    def test_low_volume_ratio_filtered(self):
        out = _scanner(min_volume_ratio=0.50).scan([_pred(volume_ratio=0.40)])
        assert not out.has_candidates
        assert any("volume_ratio" in f.reason.lower() for f in out.filtered_out)

    def test_low_expected_move_filtered(self):
        out = _scanner(min_expected_move_pct=0.30).scan([_pred(expected_move_pct=0.25)])
        assert not out.has_candidates
        assert any("expected_move" in f.reason.lower() for f in out.filtered_out)

    def test_passing_all_prefilters_reaches_ev_stage(self):
        """A candidate that clears all pre-filters is either shortlisted or EV-filtered."""
        out = _scanner().scan([_pred(
            confidence=0.70, expected_move_pct=0.80, stop_loss_pct=0.20,
            avg_daily_volume_cr=5.0, volume_ratio=1.5,
        )])
        if not out.has_candidates:
            # Must have been EV-filtered (not pre-filtered)
            assert len(out.filtered_out) == 1
            assert "ev" in out.filtered_out[0].reason.lower()
        else:
            assert len(out.shortlist) == 1

    def test_short_direction_passes_prefilter(self):
        """SHORT is a valid direction — should not be pre-filtered for direction."""
        out = _scanner().scan([
            _pred(direction="SHORT", confidence=0.70, expected_move_pct=0.80, stop_loss_pct=0.20)
        ])
        assert not any(
            "NO_TRADE" in f.reason for f in out.filtered_out
        ), "SHORT direction should not be pre-filtered as NO_TRADE"

    def test_multiple_candidates_individually_filtered(self):
        """Each candidate is evaluated independently."""
        out = _scanner().scan([
            _pred(symbol="PASS", confidence=0.70, expected_move_pct=0.80, stop_loss_pct=0.20),
            _pred(symbol="FAIL", direction="NO_TRADE"),
        ])
        filtered_syms = {f.symbol for f in out.filtered_out}
        assert "FAIL" in filtered_syms


# ── THE KEY TEST: ranking by after-cost EV, not by raw confidence ─────────────

class TestRankingByAfterCostEV:
    """THE structural guarantee: ranking is by after-cost EV, never by raw confidence.

    The test below is designed so that the candidate with the HIGHEST confidence
    has NEGATIVE after-cost EV (must be excluded), while candidates with LOWER
    confidence have positive EV. Naive confidence-based ranking would give the
    wrong order. This test fails if the scanner ever ranks by raw confidence.
    """

    def test_high_confidence_tiny_move_loses_to_lower_confidence_big_move(self):
        """
        Candidate A — highest confidence, tiny predicted move, NEGATIVE after-cost EV:
          confidence=0.90, move=0.05%, stop=0.03%
          gross_ev  = 0.90×0.05 − 0.10×0.03 = 0.042%
          cost_pct  ≈ 0.106%   (qty=50, price=₹1000; see module docstring)
          after_cost = 0.042 − 0.106 = −0.064%  →  MUST NOT appear on shortlist.

        Candidate B — lower confidence, large move, HIGHEST after-cost EV:
          confidence=0.65, move=0.80%, stop=0.30%
          gross_ev  = 0.65×0.80 − 0.35×0.30 = 0.415%
          after_cost ≈ 0.415 − 0.106 = 0.309%  →  rank #1.

        Candidate C — medium confidence, medium move, second-highest after-cost EV:
          confidence=0.75, move=0.40%, stop=0.15%
          gross_ev  = 0.75×0.40 − 0.25×0.15 = 0.2625%
          after_cost ≈ 0.2625 − 0.106 = 0.156%  →  rank #2.

        By RAW confidence:  A(0.90) > C(0.75) > B(0.65)  ← wrong order, A should be excluded
        By AFTER-COST EV:   B(0.309%) > C(0.156%);  A excluded   ← correct
        """
        scanner = _scanner(representative_qty=50)
        price = "1000"

        pred_A = _pred(
            symbol="HIGH_CONF_TINY_MOVE",
            confidence=0.90, expected_move_pct=0.05, stop_loss_pct=0.03,
            entry_price=price,
        )
        pred_B = _pred(
            symbol="LOW_CONF_BIG_MOVE",
            confidence=0.65, expected_move_pct=0.80, stop_loss_pct=0.30,
            entry_price=price,
        )
        pred_C = _pred(
            symbol="MED_CONF_MED_MOVE",
            confidence=0.75, expected_move_pct=0.40, stop_loss_pct=0.15,
            entry_price=price,
        )

        out = scanner.scan([pred_A, pred_B, pred_C])

        shortlist_syms = [r.prediction.symbol for r in out.shortlist]
        filtered_syms  = [f.symbol for f in out.filtered_out]

        # A must NOT be shortlisted despite having the highest raw confidence
        assert "HIGH_CONF_TINY_MOVE" not in shortlist_syms, (
            "Candidate A (conf=0.90) has negative after-cost EV and must be excluded, "
            "even though it has the highest confidence"
        )
        assert "HIGH_CONF_TINY_MOVE" in filtered_syms

        # B and C must be shortlisted
        assert "LOW_CONF_BIG_MOVE" in shortlist_syms
        assert "MED_CONF_MED_MOVE" in shortlist_syms

        # B (conf=0.65) must rank ABOVE C (conf=0.75) — after-cost EV drives rank
        rank_B = next(r.rank for r in out.shortlist if r.prediction.symbol == "LOW_CONF_BIG_MOVE")
        rank_C = next(r.rank for r in out.shortlist if r.prediction.symbol == "MED_CONF_MED_MOVE")
        assert rank_B < rank_C, (
            f"B (conf=0.65, higher EV) should rank above C (conf=0.75, lower EV), "
            f"but got rank_B={rank_B}, rank_C={rank_C}. "
            "Scanner appears to be ranking by raw confidence, not after-cost EV."
        )

    def test_after_cost_ev_values_are_positive_for_all_shortlisted(self):
        """Every shortlisted candidate must have a strictly positive after-cost EV."""
        out = _scanner().scan([
            _pred(symbol=f"S{i}", confidence=0.70,
                  expected_move_pct=0.60 + i * 0.10, stop_loss_pct=0.20)
            for i in range(4)
        ])
        for result in out.shortlist:
            assert result.after_cost_ev_pct > 0.0, (
                f"{result.prediction.symbol} shortlisted with "
                f"after_cost_ev={result.after_cost_ev_pct:.6f}% ≤ 0"
            )

    def test_shortlist_is_sorted_descending_by_after_cost_ev(self):
        """Shortlist must be in descending after-cost EV order."""
        # Three candidates with clearly distinct EVs (same price → same cost_pct)
        # S1: gross=0.60×1.00−0.40×0.30=0.48%, ranks highest
        # S2: gross=0.65×0.60−0.35×0.25=0.3025%, middle
        # S3: gross=0.70×0.40−0.30×0.20=0.22%, lowest
        out = _scanner().scan([
            _pred(symbol="S1", confidence=0.60, expected_move_pct=1.00, stop_loss_pct=0.30),
            _pred(symbol="S2", confidence=0.65, expected_move_pct=0.60, stop_loss_pct=0.25),
            _pred(symbol="S3", confidence=0.70, expected_move_pct=0.40, stop_loss_pct=0.20),
        ])
        evs = [r.after_cost_ev_pct for r in out.shortlist]
        assert evs == sorted(evs, reverse=True), (
            f"Shortlist not sorted descending by after-cost EV: {evs}"
        )

    def test_short_direction_ranked_by_ev(self):
        """SHORT predictions follow the same EV ranking logic as LONG."""
        out = _scanner().scan([
            _pred(direction="SHORT", confidence=0.65,
                  expected_move_pct=0.80, stop_loss_pct=0.30)
        ])
        if out.has_candidates:
            assert out.shortlist[0].after_cost_ev_pct > 0.0

    def test_ev_breakdown_fields_consistent(self):
        """gross_ev − estimated_cost == after_cost_ev for every shortlisted result."""
        out = _scanner().scan([
            _pred(confidence=0.70, expected_move_pct=0.80, stop_loss_pct=0.20)
        ])
        for r in out.shortlist:
            expected = r.gross_ev_pct - r.estimated_cost_pct
            assert abs(r.after_cost_ev_pct - expected) < 1e-9, (
                f"EV breakdown inconsistent: {r.gross_ev_pct:.6f} − "
                f"{r.estimated_cost_pct:.6f} ≠ {r.after_cost_ev_pct:.6f}"
            )


# ── Empty shortlist — not-trading is a valid outcome ─────────────────────────

class TestEmptyShortlist:
    def test_empty_when_all_candidates_have_negative_ev(self):
        """All candidates pass pre-filters but have negative after-cost EV → empty shortlist."""
        # gross_ev for both:
        #   FLAT_A: 0.60×0.05 − 0.40×0.03 = 0.018%  →  after_cost ≈ −0.088% < 0
        #   FLAT_B: 0.55×0.08 − 0.45×0.04 = 0.026%  →  after_cost ≈ −0.080% < 0
        out = _scanner().scan([
            _pred(symbol="FLAT_A", confidence=0.60, expected_move_pct=0.05, stop_loss_pct=0.03),
            _pred(symbol="FLAT_B", confidence=0.55, expected_move_pct=0.08, stop_loss_pct=0.04),
        ])
        assert not out.has_candidates
        assert out.best is None
        assert len(out.shortlist) == 0
        # Both must appear in filtered_out (EV-filtered, not pre-filtered)
        filtered_syms = {f.symbol for f in out.filtered_out}
        assert "FLAT_A" in filtered_syms
        assert "FLAT_B" in filtered_syms

    def test_empty_when_all_no_trade(self):
        out = _scanner().scan([
            _pred(symbol="NT_A", direction="NO_TRADE"),
            _pred(symbol="NT_B", direction="NO_TRADE"),
        ])
        assert not out.has_candidates
        assert len(out.filtered_out) == 2

    def test_empty_on_empty_input(self):
        out = _scanner().scan([])
        assert not out.has_candidates
        assert len(out.filtered_out) == 0
        assert len(out.shortlist) == 0

    def test_has_candidates_true_when_positive_ev_exists(self):
        # gross_ev = 0.70×0.80 − 0.30×0.20 = 0.50%  →  after_cost ≈ 0.394% > 0
        out = _scanner().scan([
            _pred(confidence=0.70, expected_move_pct=0.80, stop_loss_pct=0.20)
        ])
        assert out.has_candidates
        assert out.best is not None
        assert out.best.rank == 1


# ── Shortlist size limit ──────────────────────────────────────────────────────

class TestShortlistSizeLimit:
    def test_shortlist_capped_at_max_size(self):
        # 8 candidates all with high EV → only top 3 retained
        scanner = _scanner(max_shortlist_size=3)
        preds = [
            _pred(symbol=f"STOCK_{i}", confidence=0.70,
                  expected_move_pct=1.00 - i * 0.05, stop_loss_pct=0.25)
            for i in range(8)
        ]
        out = scanner.scan(preds)
        assert len(out.shortlist) <= 3

    def test_fewer_than_max_when_fewer_pass(self):
        out = _scanner(max_shortlist_size=10).scan([
            _pred(confidence=0.70, expected_move_pct=0.80, stop_loss_pct=0.20)
        ])
        assert len(out.shortlist) <= 1

    def test_best_candidates_retained_when_capped(self):
        """After capping, the retained candidates are the highest-EV ones."""
        # HIGH_EV: gross = 0.65×1.20 − 0.35×0.40 = 0.64%
        # MED_EV:  gross = 0.65×0.80 − 0.35×0.30 = 0.415%
        # LOW_EV:  gross = 0.65×0.50 − 0.35×0.20 = 0.255%
        out = _scanner(max_shortlist_size=2).scan([
            _pred(symbol="HIGH_EV", confidence=0.65, expected_move_pct=1.20, stop_loss_pct=0.40),
            _pred(symbol="MED_EV",  confidence=0.65, expected_move_pct=0.80, stop_loss_pct=0.30),
            _pred(symbol="LOW_EV",  confidence=0.65, expected_move_pct=0.50, stop_loss_pct=0.20),
        ])
        if len(out.shortlist) == 2:
            symbols = {r.prediction.symbol for r in out.shortlist}
            assert "HIGH_EV" in symbols
            assert "MED_EV" in symbols
            assert "LOW_EV" not in symbols


# ── ScanOutput structure ──────────────────────────────────────────────────────

class TestScanOutput:
    def test_filtered_out_contains_prefilter_and_ev_exclusions(self):
        """Both pre-filter exclusions and negative-EV exclusions appear in filtered_out."""
        out = _scanner().scan([
            _pred(symbol="NO_TRADE_ONE", direction="NO_TRADE"),         # pre-filter
            _pred(symbol="TINY_MOVE",   confidence=0.70,               # EV-filtered
                  expected_move_pct=0.05, stop_loss_pct=0.03),
            _pred(symbol="GOOD_ONE",    confidence=0.70,               # shortlisted
                  expected_move_pct=0.80, stop_loss_pct=0.20),
        ])
        filtered_syms = {f.symbol for f in out.filtered_out}
        assert "NO_TRADE_ONE" in filtered_syms
        assert "TINY_MOVE" in filtered_syms
        assert "GOOD_ONE" not in filtered_syms

    def test_filtered_out_reasons_are_non_empty(self):
        out = _scanner().scan([_pred(direction="NO_TRADE")])
        for f in out.filtered_out:
            assert f.reason, "FilteredOut.reason must not be empty"

    def test_scan_result_ranks_are_sequential_from_one(self):
        out = _scanner().scan([
            _pred(symbol=f"S{i}", confidence=0.70,
                  expected_move_pct=0.80 - i * 0.10, stop_loss_pct=0.20)
            for i in range(3)
        ])
        ranks = [r.rank for r in out.shortlist]
        assert ranks == list(range(1, len(ranks) + 1))

    def test_scan_result_reasoning_is_non_empty(self):
        out = _scanner().scan([
            _pred(confidence=0.70, expected_move_pct=0.80, stop_loss_pct=0.20)
        ])
        for r in out.shortlist:
            assert r.reasoning

    def test_scanned_at_propagated_when_provided(self):
        ts = datetime(2026, 7, 14, 9, 30, tzinfo=timezone.utc)
        out = _scanner().scan([], scanned_at=ts)
        assert out.scanned_at == ts

    def test_scanned_at_defaults_to_now(self):
        before = datetime.now(timezone.utc)
        out = _scanner().scan([])
        after = datetime.now(timezone.utc)
        assert before <= out.scanned_at <= after

    def test_shortlisted_symbol_absent_from_filtered_out(self):
        """A shortlisted symbol must not simultaneously appear in filtered_out."""
        out = _scanner().scan([
            _pred(symbol="GOOD", confidence=0.70, expected_move_pct=0.80, stop_loss_pct=0.20)
        ])
        filtered_syms = {f.symbol for f in out.filtered_out}
        for r in out.shortlist:
            assert r.prediction.symbol not in filtered_syms


# ── ScannerConfig validation ──────────────────────────────────────────────────

class TestScannerConfig:
    def test_scanner_config_loads_from_toml(self):
        from src.config import load_config
        cfg = load_config()
        sc = cfg.trading.scanner
        assert sc.max_shortlist_size >= 1
        assert 0.0 <= sc.min_confidence <= 1.0
        assert sc.min_expected_move_pct >= 0.0
        assert sc.min_avg_daily_volume_cr >= 0.0
        assert sc.min_volume_ratio >= 0.0
        assert sc.representative_qty >= 1

    def test_scanner_config_defaults_are_sane(self):
        sc = ScannerConfig()
        assert sc.max_shortlist_size == 10
        assert sc.min_confidence == 0.55
        assert sc.representative_qty == 50

    def test_zero_max_shortlist_size_rejected(self):
        with pytest.raises(Exception):
            ScannerConfig(max_shortlist_size=0)

    def test_negative_representative_qty_rejected(self):
        with pytest.raises(Exception):
            ScannerConfig(representative_qty=-1)

    def test_negative_min_confidence_rejected(self):
        with pytest.raises(Exception):
            ScannerConfig(min_confidence=-0.1)
