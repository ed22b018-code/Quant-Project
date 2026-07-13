"""Tests for the capital allocator (src/trading/allocator.py).

Sizing formula under test (kelly_fraction=0.25, vol_target=1.0%, max_vol_scalar=2.0):
    raw_pct = 0.25 × (after_cost_ev_pct / stop_loss_pct)
                    × min(1.0 / estimated_volatility_pct, 2.0)

Pre-verified arithmetic for key test values (total_capital = ₹5,00,000):
  Fractional Kelly exact (ev=0.05%, stop=0.50%, vol=1.0%):
    kelly_f = 0.05/0.50 = 0.10  |  frac = 0.025  |  vol_scalar=1.0  →  raw=2.5%
    position_value = 12,500  →  qty = 12  (floor)  →  actual_value = 12,000

  Higher-EV lower-vol candidate B (ev=0.10%, stop=1.00%, vol=1.0%):
    raw = 0.25×0.10×1.0 = 2.5%  →  qty = 12

  Lower-EV higher-vol candidate A (ev=0.02%, stop=1.00%, vol=2.0%):
    raw = 0.25×0.02×0.5 = 0.25%  →  qty = 1

  Not-confidence-weight: A (conf=0.80) ev=0.12%, stop=0.20%, vol=2.0%
    raw = 0.25×0.60×0.50 = 7.5%  →  qty = 37
    B (conf=0.65) ev=0.25%, stop=0.20%, vol=1.0%
    raw = 0.25×1.25×1.0 = 31.25% → individual_cap → 10%  →  qty = 50
    B gets MORE capital despite lower confidence.

  Individual cap (ev=0.50%, stop=0.50%, vol=0.10%):
    raw = 0.25×1.0×2.0 = 50%  →  individual_cap:10%  →  qty = 50

  Sector cap: 3 BANK candidates, each raw=10% (individual_capped), max_sector=25%
    C1: sector_used=0%   → 10% passes  (no cap)
    C2: sector_used=10%  → 10% passes  (no cap)
    C3: sector_used=20%  → available=5%, sector_cap fires  → qty = 25

  Total exposure cap: 5 candidates each raw=10%, max_intraday=40%
    C1-C4: 10% each → total=40%  (no cap fires)
    C5: intraday_remaining=0 → rejected

  Cash reserve cap (min_cash=70%, max_intraday=90%, max_pos=40%):
    Candidate: ev=0.50%, stop=0.50%, vol=0.50%
    raw = 0.25×1.0×2.0 = 50%  → individual_cap:40%  → cash_reserve:70% → 30%
    position_value = 1,50,000  →  qty = 150
"""
from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from src.config import AllocatorConfig
from src.trading.allocator import (
    AllocationInput,
    AllocationResult,
    CapitalAllocator,
    RejectedCandidate,
    SizedOrder,
)
from src.trading.costs import CostConfig, CostSimulator
from src.trading.execution import ExitReason, PaperExecutionEngine
from src.trading.scanner import PredictionInput, ScanResult, Scanner, ScannerConfig
from src.trading.slippage import FixedBpsSlippage


# ── Helpers ───────────────────────────────────────────────────────────────────

_TOTAL = Decimal("500000")


def _alloc(
    total_capital: Decimal = _TOTAL,
    kelly_fraction: float = 0.25,
    vol_target_pct: float = 1.0,
    max_vol_scalar: float = 2.0,
    max_position_pct: float = 0.10,
    max_intraday_exposure_pct: float = 0.40,
    min_cash_reserve_pct: float = 0.10,
    max_sector_exposure_pct: float = 0.25,
) -> CapitalAllocator:
    return CapitalAllocator(
        allocator_config=AllocatorConfig(
            kelly_fraction=kelly_fraction,
            vol_target_pct=vol_target_pct,
            max_vol_scalar=max_vol_scalar,
        ),
        max_position_pct=max_position_pct,
        max_intraday_exposure_pct=max_intraday_exposure_pct,
        min_cash_reserve_pct=min_cash_reserve_pct,
        max_sector_exposure_pct=max_sector_exposure_pct,
        total_capital=total_capital,
    )


def _input(
    symbol: str = "TEST",
    direction: str = "LONG",
    confidence: float = 0.70,
    after_cost_ev_pct: float = 0.30,
    stop_loss_pct: float = 0.20,
    entry_price: str = "1000",
    vol: float = 1.0,
    sector: str = "OTHER",
    rank: int = 1,
) -> AllocationInput:
    pred = PredictionInput(
        symbol=symbol,
        direction=direction,  # type: ignore[arg-type]
        confidence=confidence,
        expected_move_pct=0.50,
        stop_loss_pct=stop_loss_pct,
        entry_price=Decimal(entry_price),
        avg_daily_volume_cr=5.0,
        volume_ratio=1.2,
    )
    result = ScanResult(
        rank=rank,
        prediction=pred,
        gross_ev_pct=after_cost_ev_pct + 0.10,
        estimated_cost_pct=0.10,
        after_cost_ev_pct=after_cost_ev_pct,
        reasoning=f"test rank={rank}",
    )
    return AllocationInput(scan_result=result, estimated_volatility_pct=vol, sector=sector)


def _ts() -> datetime:
    return datetime(2026, 7, 14, 9, 30, tzinfo=timezone.utc)


# ── Sizing tests ──────────────────────────────────────────────────────────────

class TestSizing:
    """Fractional Kelly + vol scaling — not equal-weight, not confidence-weight."""

    def test_fractional_kelly_math_exact(self):
        """Verify the exact formula to the share level.

        ev=0.05%, stop=0.50%, vol=1.0%, kelly=0.25, vol_target=1.0%:
          kelly_f = 0.10  |  frac = 0.025  |  vol_scalar = 1.0  →  raw = 2.5%
          position_value = 0.025 × 5,00,000 = 12,500  →  qty = 12  (floor)
          actual_value = 12,000  →  position_pct = 0.024
        """
        allocator = _alloc()
        inp = _input(after_cost_ev_pct=0.05, stop_loss_pct=0.50, vol=1.0)
        result = allocator.allocate([inp])

        assert len(result.orders) == 1
        order = result.orders[0]
        assert abs(order.raw_position_pct - 0.025) < 1e-9
        assert order.qty == 12
        assert order.position_value == Decimal("12000")
        assert abs(order.position_pct - 0.024) < 1e-9

    def test_higher_ev_lower_vol_gets_larger_size(self):
        """Higher after-cost EV and lower volatility both push size UP.

        Candidate A: ev=0.02%, stop=1.00%, vol=2.0%  →  raw=0.25×0.02×0.5=0.25%  →  qty=1
        Candidate B: ev=0.10%, stop=1.00%, vol=1.0%  →  raw=0.25×0.10×1.0=2.5%   →  qty=12
        """
        allocator = _alloc()
        inp_A = _input(symbol="LOW_EV_HIGH_VOL", after_cost_ev_pct=0.02,
                       stop_loss_pct=1.00, vol=2.0, rank=2)
        inp_B = _input(symbol="HIGH_EV_LOW_VOL", after_cost_ev_pct=0.10,
                       stop_loss_pct=1.00, vol=1.0, rank=1)
        result = allocator.allocate([inp_A, inp_B])

        qty = {o.symbol: o.qty for o in result.orders}
        assert qty["HIGH_EV_LOW_VOL"] > qty["LOW_EV_HIGH_VOL"], (
            f"B (higher EV, lower vol) should get more capital than A, "
            f"but B={qty['HIGH_EV_LOW_VOL']} ≤ A={qty['LOW_EV_HIGH_VOL']}"
        )
        assert qty["HIGH_EV_LOW_VOL"] == 12   # pre-verified
        assert qty["LOW_EV_HIGH_VOL"] == 1    # pre-verified

    def test_not_equal_weight(self):
        """Two candidates with different EVs/vols must NOT receive equal capital."""
        allocator = _alloc()
        inp_A = _input(symbol="SMALL", after_cost_ev_pct=0.02, stop_loss_pct=1.00, vol=2.0, rank=2)
        inp_B = _input(symbol="LARGE", after_cost_ev_pct=0.10, stop_loss_pct=1.00, vol=1.0, rank=1)
        result = allocator.allocate([inp_A, inp_B])

        qty = {o.symbol: o.qty for o in result.orders}
        assert qty["SMALL"] != qty["LARGE"], (
            "Allocator must NOT produce equal-weight allocations — "
            f"both SMALL and LARGE got qty={qty['SMALL']}"
        )

    def test_not_confidence_weight(self):
        """Higher model confidence must NOT guarantee a larger allocation.

        A (conf=0.80): ev=0.12%, stop=0.20%, vol=2.0%  →  raw=7.5%   →  qty=37
        B (conf=0.65): ev=0.25%, stop=0.20%, vol=1.0%  →  raw=31.25% → capped 10% →  qty=50

        B receives MORE capital despite lower confidence — because its after-cost EV
        and vol profile are better. This proves the allocator is NOT confidence-weighted.
        """
        allocator = _alloc()
        inp_A = _input(symbol="HIGH_CONF",  confidence=0.80,
                       after_cost_ev_pct=0.12, stop_loss_pct=0.20, vol=2.0, rank=2)
        inp_B = _input(symbol="LOWER_CONF", confidence=0.65,
                       after_cost_ev_pct=0.25, stop_loss_pct=0.20, vol=1.0, rank=1)
        result = allocator.allocate([inp_A, inp_B])

        qty = {o.symbol: o.qty for o in result.orders}
        assert qty["LOWER_CONF"] > qty["HIGH_CONF"], (
            f"B (conf=0.65) should get more capital than A (conf=0.80) due to better EV, "
            f"but B={qty['LOWER_CONF']} ≤ A={qty['HIGH_CONF']}. Allocator is confidence-weighted."
        )
        assert qty["HIGH_CONF"] == 37    # pre-verified: 7.5% × 500,000 / 1,000 = 37
        assert qty["LOWER_CONF"] == 50   # pre-verified: 10% cap → 10% × 500,000 / 1,000 = 50

    def test_vol_scaling_reduces_size_for_high_vol(self):
        """Same EV — lower volatility gets more capital than high volatility.

        Both: ev=0.05%, stop=0.50%
        A vol=2.0%: raw=0.25×0.10×0.5=1.25%   →  qty=6
        B vol=0.5%: raw=0.25×0.10×2.0=5.0%    →  qty=25
        """
        allocator = _alloc()
        inp_A = _input(symbol="HIGH_VOL", after_cost_ev_pct=0.05, stop_loss_pct=0.50, vol=2.0, rank=2)
        inp_B = _input(symbol="LOW_VOL",  after_cost_ev_pct=0.05, stop_loss_pct=0.50, vol=0.5, rank=1)
        result = allocator.allocate([inp_A, inp_B])

        qty = {o.symbol: o.qty for o in result.orders}
        assert qty["LOW_VOL"] > qty["HIGH_VOL"]
        assert qty["HIGH_VOL"] == 6    # pre-verified
        assert qty["LOW_VOL"] == 25   # pre-verified


# ── Guardrail tests — one test per guardrail ──────────────────────────────────

class TestGuardrails:
    """Each guardrail tested explicitly. Breaches are capped and named, not silently allowed."""

    def test_individual_cap_fires_at_max_position_pct(self):
        """A trade that wants 50% gets hard-capped to 10% (max_position_pct).

        ev=0.50%, stop=0.50%, vol=0.10% → raw=0.25×1.0×2.0=50% → individual_cap:10%
        qty = 10% × 5,00,000 / 1,000 = 50
        """
        allocator = _alloc()
        inp = _input(after_cost_ev_pct=0.50, stop_loss_pct=0.50, vol=0.10)
        result = allocator.allocate([inp])

        assert len(result.orders) == 1
        order = result.orders[0]
        assert "individual_cap:10%" in order.guardrails_applied, (
            f"Expected individual_cap guardrail in {order.guardrails_applied}"
        )
        assert abs(order.raw_position_pct - 0.50) < 1e-6
        assert order.qty == 50
        assert order.position_pct <= 0.10 + 1e-9  # never exceeds max_position_pct

    def test_total_exposure_cap_fires_at_max_intraday_pct(self):
        """Five 10%-sized candidates fill 40% (the cap) — the 5th is rejected.

        Each candidate: ev=0.20%, stop=0.50%, vol=1.0%
          kelly_f=0.40, frac=0.10, vol_scalar=1.0 → raw=10% (exactly at individual cap).
        After C1-C4 fill 40%, C5 has zero remaining → rejected with exposure_cap reason.
        """
        allocator = _alloc(max_intraday_exposure_pct=0.40, max_sector_exposure_pct=1.0)
        inputs = [
            _input(symbol=f"STOCK_{i}", after_cost_ev_pct=0.20,
                   stop_loss_pct=0.50, vol=1.0, rank=i)
            for i in range(1, 6)
        ]
        result = allocator.allocate(inputs)

        assert len(result.orders) == 4, (
            f"Expected 4 orders (4×10%=40% = cap), got {len(result.orders)}"
        )
        assert len(result.rejected) == 1
        assert abs(result.total_exposure_pct - 0.40) < 1e-9
        rejected_reason = result.rejected[0].reason
        assert "zero allocation" in rejected_reason or "exposure_cap" in rejected_reason, (
            f"5th candidate rejection reason should mention cap, got: {rejected_reason!r}"
        )

    def test_cash_reserve_limits_deployable_capital(self):
        """When min_cash_reserve=70%, only 30% of capital can be deployed.

        Config: max_pos=40%, min_cash=70%, max_intraday=90%
        Candidate: ev=0.50%, stop=0.50%, vol=0.50%
          raw = 0.25×1.0×2.0 = 50% → individual_cap:40% → 40%
          cash_remaining = (1-0.70) - 0 = 30% < intraday_remaining=90%
          → cash_reserve:70% fires → final=30%
        qty = 30% × 5,00,000 / 1,000 = 150
        """
        allocator = _alloc(
            max_position_pct=0.40,
            min_cash_reserve_pct=0.70,
            max_intraday_exposure_pct=0.90,
            max_sector_exposure_pct=1.0,
        )
        inp = _input(after_cost_ev_pct=0.50, stop_loss_pct=0.50, vol=0.50)
        result = allocator.allocate([inp])

        assert len(result.orders) == 1
        order = result.orders[0]
        assert "cash_reserve:70%" in order.guardrails_applied, (
            f"Expected cash_reserve guardrail, got {order.guardrails_applied}"
        )
        assert order.qty == 150  # pre-verified: 30% × 500,000 / 1,000
        assert order.position_pct <= 0.30 + 1e-9  # never exceeds deployable

    def test_sector_cap_enforced(self):
        """Three same-sector candidates: first two pass, third is sector-capped.

        Each candidate: ev=0.30%, stop=0.50%, vol=1.0%
          raw = 0.25×0.60×1.0 = 15% → individual_cap:10% → 10%
        After C1 (10%) and C2 (10%), sector BANK = 20%.
        C3 sector_available = 25% - 20% = 5%: sector_cap fires → 5% → qty=25.
        """
        allocator = _alloc(max_sector_exposure_pct=0.25)
        inputs = [
            _input(symbol=f"BANK_{i}", after_cost_ev_pct=0.30,
                   stop_loss_pct=0.50, vol=1.0, sector="BANK", rank=i)
            for i in range(1, 4)
        ]
        result = allocator.allocate(inputs)

        assert len(result.orders) == 3
        by_sym = {o.symbol: o for o in result.orders}

        # First two pass with no sector cap
        assert "sector_cap" not in " ".join(by_sym["BANK_1"].guardrails_applied)
        assert "sector_cap" not in " ".join(by_sym["BANK_2"].guardrails_applied)

        # Third is sector-capped
        cap3 = by_sym["BANK_3"]
        assert any("sector_cap" in g for g in cap3.guardrails_applied), (
            f"Expected sector_cap on BANK_3, got {cap3.guardrails_applied}"
        )
        assert cap3.qty == 25  # pre-verified: 5% × 500,000 / 1,000 = 25
        assert abs(result.sector_exposure.get("BANK", 0) - 0.25) < 1e-9

    def test_all_guardrails_reported_in_guardrails_applied(self):
        """Multiple guardrails that fire all appear in guardrails_applied."""
        # individual_cap + cash_reserve both fire (see test_cash_reserve):
        allocator = _alloc(
            max_position_pct=0.40,
            min_cash_reserve_pct=0.70,
            max_intraday_exposure_pct=0.90,
            max_sector_exposure_pct=1.0,
        )
        inp = _input(after_cost_ev_pct=0.50, stop_loss_pct=0.50, vol=0.50)
        order = allocator.allocate([inp]).orders[0]
        assert "individual_cap:40%" in order.guardrails_applied
        assert "cash_reserve:70%" in order.guardrails_applied

    def test_no_guardrails_applied_is_empty_tuple(self):
        """If no cap fires, guardrails_applied is an empty tuple."""
        allocator = _alloc()
        # Small enough to not hit any cap: raw_pct = 2.5% (well under 10%)
        inp = _input(after_cost_ev_pct=0.05, stop_loss_pct=0.50, vol=1.0)
        order = allocator.allocate([inp]).orders[0]
        assert order.guardrails_applied == ()

    def test_guardrail_breach_is_never_silently_allowed(self):
        """position_pct never exceeds max_position_pct regardless of raw_pct."""
        allocator = _alloc(max_position_pct=0.10)
        for ev in (0.30, 0.50, 1.00, 5.00):  # progressively wild raw_pcts
            inp = _input(after_cost_ev_pct=ev, stop_loss_pct=0.50, vol=0.10)
            result = allocator.allocate([inp])
            if result.orders:
                assert result.orders[0].position_pct <= 0.10 + 1e-9, (
                    f"position_pct {result.orders[0].position_pct} exceeded "
                    f"max_position_pct 10% for ev={ev}%"
                )

    def test_drawdown_pause_blocks_all_orders(self):
        """When drawdown_paused=True, no orders are placed and all inputs are rejected."""
        allocator = _alloc()
        inputs = [_input(symbol=f"S{i}", rank=i) for i in range(1, 4)]
        result = allocator.allocate(inputs, drawdown_paused=True)
        assert result.orders == []
        assert len(result.rejected) == 3
        assert all("drawdown_pause" in r.reason for r in result.rejected)


# ── Empty / degenerate cases ──────────────────────────────────────────────────

class TestEmptyAndDegenerate:
    def test_empty_inputs_gives_no_orders(self):
        result = _alloc().allocate([])
        assert result.orders == []
        assert result.rejected == []
        assert result.total_exposure_pct == 0.0

    def test_all_rejected_gives_no_orders(self):
        """If all candidates get zero allocation, orders is empty."""
        # Zero EV → raw_pct = 0 → rejected
        allocator = _alloc()
        inp = _input(after_cost_ev_pct=0.0)
        result = allocator.allocate([inp])
        assert result.orders == []
        assert len(result.rejected) == 1

    def test_entry_price_larger_than_position_gives_qty_zero_and_rejected(self):
        """If position_value < entry_price, qty=0 → rejected (not a crash)."""
        allocator = _alloc()
        # Very small EV + very high price → tiny position_value < entry_price
        inp = _input(after_cost_ev_pct=0.001, stop_loss_pct=1.00, vol=5.0,
                     entry_price="1000000")  # ₹10 lakh per share
        result = allocator.allocate([inp])
        assert result.orders == []
        assert "qty=0" in result.rejected[0].reason

    def test_no_trade_direction_rejected(self):
        """A NO_TRADE direction in AllocationInput is rejected gracefully."""
        allocator = _alloc()
        inp = _input(direction="NO_TRADE")
        result = allocator.allocate([inp])
        assert result.orders == []
        assert "NO_TRADE" in result.rejected[0].reason


# ── Integration: allocator → PaperExecutionEngine ────────────────────────────

class TestIntegration:
    """Sized orders from the allocator feed the paper engine; portfolio reconciles."""

    def test_sized_orders_feed_paper_engine(self):
        """Allocator output → PaperExecutionEngine.place_order; positions opened correctly."""
        allocator = _alloc()
        inputs = [
            _input(symbol="RELIANCE", after_cost_ev_pct=0.10,
                   stop_loss_pct=1.00, vol=1.0, entry_price="2500", rank=1),
            _input(symbol="TCS",      after_cost_ev_pct=0.05,
                   stop_loss_pct=1.00, vol=2.0, entry_price="3500", rank=2),
        ]
        alloc_result = allocator.allocate(inputs)
        assert len(alloc_result.orders) >= 1  # at least one order placed

        engine = PaperExecutionEngine(
            initial_cash=_TOTAL,
            cost_simulator=CostSimulator(CostConfig()),
            slippage_model=FixedBpsSlippage(entry_bps=0.0, exit_bps=0.0),
        )

        ts = _ts()
        for order in alloc_result.orders:
            engine.place_order(
                symbol=order.symbol,
                qty=order.qty,
                direction=order.direction,
                price=order.entry_price,
                timestamp=ts,
            )

        portfolio = engine.portfolio
        assert len(portfolio.open_positions) == len(alloc_result.orders)
        for order in alloc_result.orders:
            assert engine.has_position(order.symbol)

    def test_portfolio_cash_after_allocation(self):
        """Portfolio cash = initial_cash − sum(position values) after all fills.

        Uses zero slippage so fill_price = entry_price exactly.
        """
        allocator = _alloc()
        inp = _input(symbol="HDFC", after_cost_ev_pct=0.05,
                     stop_loss_pct=0.50, vol=1.0, entry_price="1600", rank=1)
        alloc_result = allocator.allocate([inp])
        assert len(alloc_result.orders) == 1
        order = alloc_result.orders[0]

        engine = PaperExecutionEngine(
            initial_cash=_TOTAL,
            cost_simulator=CostSimulator(CostConfig()),
            slippage_model=FixedBpsSlippage(entry_bps=0.0, exit_bps=0.0),
        )
        engine.place_order(
            symbol=order.symbol, qty=order.qty,
            direction=order.direction, price=order.entry_price, timestamp=_ts(),
        )

        expected_cash = _TOTAL - Decimal(order.qty) * order.entry_price
        assert engine.portfolio.cash == expected_cash

    def test_total_exposure_consistent_with_engine(self):
        """AllocationResult.total_exposure_pct is consistent with orders placed."""
        allocator = _alloc()
        inputs = [
            _input(symbol="A", after_cost_ev_pct=0.05, stop_loss_pct=0.50, vol=1.0, rank=1),
            _input(symbol="B", after_cost_ev_pct=0.10, stop_loss_pct=0.50, vol=1.0, rank=2),
        ]
        result = allocator.allocate(inputs)
        computed_exposure = sum(o.position_pct for o in result.orders)
        assert abs(result.total_exposure_pct - computed_exposure) < 1e-9


# ── AllocatorConfig validation ────────────────────────────────────────────────

class TestAllocatorConfig:
    def test_config_loads_from_toml(self):
        from src.config import load_config
        cfg = load_config()
        ac = cfg.trading.allocator
        assert 0.0 < ac.kelly_fraction <= 1.0
        assert ac.vol_target_pct > 0.0
        assert ac.max_vol_scalar > 0.0

    def test_default_config_is_quarter_kelly(self):
        ac = AllocatorConfig()
        assert ac.kelly_fraction == 0.25
        assert ac.vol_target_pct == 1.0
        assert ac.max_vol_scalar == 2.0

    def test_zero_kelly_fraction_rejected(self):
        with pytest.raises(Exception):
            AllocatorConfig(kelly_fraction=0.0)

    def test_negative_kelly_fraction_rejected(self):
        with pytest.raises(Exception):
            AllocatorConfig(kelly_fraction=-0.1)

    def test_kelly_fraction_above_one_rejected(self):
        with pytest.raises(Exception):
            AllocatorConfig(kelly_fraction=1.01)

    def test_negative_vol_target_rejected(self):
        with pytest.raises(Exception):
            AllocatorConfig(vol_target_pct=-0.5)

    def test_zero_max_vol_scalar_rejected(self):
        with pytest.raises(Exception):
            AllocatorConfig(max_vol_scalar=0.0)

    def test_full_kelly_allowed(self):
        ac = AllocatorConfig(kelly_fraction=1.0)
        assert ac.kelly_fraction == 1.0  # edge: exactly 1.0 is valid

    def test_current_exposure_propagated_to_guardrails(self):
        """current_exposure_pct is consumed by guardrail checks correctly."""
        allocator = _alloc(max_intraday_exposure_pct=0.40)
        # Start with 30% already deployed; only 10% left before 40% cap
        inp = _input(after_cost_ev_pct=0.20, stop_loss_pct=0.50, vol=1.0)
        result = allocator.allocate([inp], current_exposure_pct=0.30)
        if result.orders:
            assert result.orders[0].position_pct <= 0.10 + 1e-9
        # With 30% existing + 10% new = 40% exactly at cap
        assert result.total_exposure_pct <= 0.40 + 1e-9
