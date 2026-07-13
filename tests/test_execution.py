"""Tests for the paper execution engine (src/trading/execution.py).

Covers:
  - Full round-trip net_pnl matches CostSimulator on slippage-adjusted prices
  - Each exit reason (TARGET_HIT, STOP_HIT, TIME_BASED, MANUAL)
  - Portfolio cash invariant: cash == initial_cash + sum(net_pnl)
  - Multiple sequential trades accumulate correctly
  - No-trade = portfolio unchanged
  - Interface swap: MockAlternativeEngine satisfies ExecutionEngine ABC
  - Guards: duplicate position raises; close nonexistent raises
  - Slippage integration: fills worse → lower net_pnl
"""
from __future__ import annotations

import pytest
from datetime import datetime
from decimal import Decimal

from src.config import load_config
from src.trading.costs import CostSimulator
from src.trading.slippage import FixedBpsSlippage, SlippageModel
from src.trading.execution import (
    ClosedTrade,
    ExecutionEngine,
    ExitReason,
    FillResult,
    OpenPosition,
    PaperExecutionEngine,
    Portfolio,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

T0 = datetime(2026, 1, 15, 9, 30, 0)
T1 = datetime(2026, 1, 15, 10, 0, 0)
T2 = datetime(2026, 1, 15, 10, 30, 0)


def _make_engine(
    initial_cash: Decimal = Decimal("500000"),
    entry_bps: float = 0.0,
    exit_bps: float = 0.0,
) -> PaperExecutionEngine:
    """Default: zero slippage for deterministic cost-only assertions."""
    cfg = load_config().trading.costs
    sim = CostSimulator(cfg)
    slippage = FixedBpsSlippage(entry_bps=entry_bps, exit_bps=exit_bps)
    return PaperExecutionEngine(Decimal(str(initial_cash)), sim, slippage)


# ── Round-trip net P&L matches CostSimulator ──────────────────────────────────

class TestRoundTripNetPnl:
    """With zero slippage, engine net_pnl == CostSimulator.compute() result."""

    def test_long_net_pnl_matches_cost_simulator(self):
        engine = _make_engine()
        engine.place_order("RELIANCE", 100, "LONG", Decimal("2500"), T0)
        closed = engine.close_position("RELIANCE", Decimal("2520"), ExitReason.TARGET_HIT, T1)

        cfg = load_config().trading.costs
        expected = CostSimulator(cfg).compute(
            "RELIANCE", 100, "LONG", Decimal("2500"), Decimal("2520")
        )
        assert closed.net_pnl == expected.net_pnl

    def test_short_net_pnl_matches_cost_simulator(self):
        engine = _make_engine()
        engine.place_order("INFY", 50, "SHORT", Decimal("1800"), T0)
        closed = engine.close_position("INFY", Decimal("1780"), ExitReason.TARGET_HIT, T1)

        cfg = load_config().trading.costs
        expected = CostSimulator(cfg).compute(
            "INFY", 50, "SHORT", Decimal("1800"), Decimal("1780")
        )
        assert closed.net_pnl == expected.net_pnl

    def test_losing_long_trade(self):
        """Net P&L is negative when exit < entry by more than costs."""
        engine = _make_engine()
        engine.place_order("SBIN", 200, "LONG", Decimal("600"), T0)
        closed = engine.close_position("SBIN", Decimal("590"), ExitReason.STOP_HIT, T1)
        assert closed.net_pnl < Decimal("0")

    def test_net_pnl_invariant_on_closed_trade(self):
        """ClosedTrade.net_pnl always equals trade_result.net_pnl."""
        engine = _make_engine()
        engine.place_order("TCS", 10, "LONG", Decimal("4000"), T0)
        closed = engine.close_position("TCS", Decimal("4050"), ExitReason.TARGET_HIT, T1)
        assert closed.net_pnl == closed.trade_result.net_pnl


# ── Exit reasons ──────────────────────────────────────────────────────────────

class TestExitReasons:
    @pytest.mark.parametrize("exit_reason", [
        ExitReason.TARGET_HIT,
        ExitReason.STOP_HIT,
        ExitReason.TIME_BASED,
        ExitReason.MANUAL,
    ])
    def test_all_exit_reasons_accepted(self, exit_reason: ExitReason):
        engine = _make_engine()
        engine.place_order("HDFCBANK", 25, "LONG", Decimal("1700"), T0)
        closed = engine.close_position("HDFCBANK", Decimal("1710"), exit_reason, T1)
        assert closed.exit_reason is exit_reason
        assert closed.net_pnl == closed.trade_result.net_pnl


# ── Portfolio cash invariant ──────────────────────────────────────────────────

class TestPortfolioCashInvariant:
    """cash == initial_cash + sum(net_pnl) must hold after every closed trade."""

    def test_after_one_long_trade(self):
        initial = Decimal("500000")
        engine = _make_engine(initial_cash=initial)
        engine.place_order("TCS", 10, "LONG", Decimal("4000"), T0)
        closed = engine.close_position("TCS", Decimal("4050"), ExitReason.TARGET_HIT, T1)
        assert engine.portfolio.cash == initial + closed.net_pnl

    def test_after_one_short_trade(self):
        initial = Decimal("500000")
        engine = _make_engine(initial_cash=initial)
        engine.place_order("SBIN", 200, "SHORT", Decimal("800"), T0)
        closed = engine.close_position("SBIN", Decimal("785"), ExitReason.TARGET_HIT, T1)
        assert engine.portfolio.cash == initial + closed.net_pnl

    def test_after_winning_then_losing(self):
        initial = Decimal("500000")
        engine = _make_engine(initial_cash=initial)

        engine.place_order("RELIANCE", 50, "LONG", Decimal("2500"), T0)
        c1 = engine.close_position("RELIANCE", Decimal("2520"), ExitReason.TARGET_HIT, T1)

        engine.place_order("INFY", 100, "LONG", Decimal("1800"), T1)
        c2 = engine.close_position("INFY", Decimal("1780"), ExitReason.STOP_HIT, T2)

        total_net = c1.net_pnl + c2.net_pnl
        assert engine.portfolio.cash == initial + total_net
        assert engine.portfolio.realized_pnl == total_net

    def test_after_three_trades(self):
        initial = Decimal("500000")
        engine = _make_engine(initial_cash=initial)

        engine.place_order("RELIANCE", 50, "LONG", Decimal("2500"), T0)
        c1 = engine.close_position("RELIANCE", Decimal("2510"), ExitReason.TARGET_HIT, T1)

        engine.place_order("INFY", 100, "SHORT", Decimal("1800"), T1)
        c2 = engine.close_position("INFY", Decimal("1785"), ExitReason.TARGET_HIT, T2)

        engine.place_order("TCS", 20, "LONG", Decimal("4000"), T2)
        c3 = engine.close_position("TCS", Decimal("3990"), ExitReason.TIME_BASED, T2)

        total_net = c1.net_pnl + c2.net_pnl + c3.net_pnl
        assert engine.portfolio.cash == initial + total_net
        assert engine.portfolio.realized_pnl == total_net
        assert len(engine.portfolio.closed_trades) == 3
        assert engine.portfolio.total_realized_net_pnl() == total_net

    def test_no_trade_portfolio_unchanged(self):
        initial = Decimal("100000")
        engine = _make_engine(initial_cash=initial)
        p = engine.portfolio
        assert p.cash == initial
        assert p.realized_pnl == Decimal("0")
        assert len(p.closed_trades) == 0
        assert len(p.open_positions) == 0

    def test_open_position_excluded_from_cash_invariant(self):
        """While a position is open, cash reflects the committed/credited amount."""
        initial = Decimal("500000")
        engine = _make_engine(initial_cash=initial)

        # After opening LONG, cash is reduced by qty * entry_filled
        engine.place_order("WIPRO", 100, "LONG", Decimal("500"), T0)
        assert engine.portfolio.cash == initial - Decimal("100") * Decimal("500")
        assert engine.has_position("WIPRO")

        # After closing, invariant resumes
        closed = engine.close_position("WIPRO", Decimal("510"), ExitReason.TARGET_HIT, T1)
        assert engine.portfolio.cash == initial + closed.net_pnl


# ── Guard rails ───────────────────────────────────────────────────────────────

class TestGuards:
    def test_duplicate_position_raises(self):
        engine = _make_engine()
        engine.place_order("WIPRO", 100, "LONG", Decimal("500"), T0)
        with pytest.raises(ValueError, match="already open"):
            engine.place_order("WIPRO", 100, "LONG", Decimal("505"), T1)

    def test_close_nonexistent_raises(self):
        engine = _make_engine()
        with pytest.raises(ValueError, match="No open position"):
            engine.close_position("WIPRO", Decimal("500"), ExitReason.MANUAL, T0)

    def test_has_position_lifecycle(self):
        engine = _make_engine()
        assert not engine.has_position("AXISBANK")
        engine.place_order("AXISBANK", 50, "LONG", Decimal("1100"), T0)
        assert engine.has_position("AXISBANK")
        engine.close_position("AXISBANK", Decimal("1110"), ExitReason.TARGET_HIT, T1)
        assert not engine.has_position("AXISBANK")

    def test_can_reopen_after_close(self):
        """Same symbol can be traded again after closing."""
        engine = _make_engine()
        engine.place_order("SBIN", 100, "LONG", Decimal("600"), T0)
        engine.close_position("SBIN", Decimal("610"), ExitReason.TARGET_HIT, T1)
        # Should not raise
        engine.place_order("SBIN", 50, "SHORT", Decimal("605"), T1)
        assert engine.has_position("SBIN")


# ── Slippage integration ──────────────────────────────────────────────────────

class TestSlippageIntegration:
    def test_fill_prices_differ_from_quoted_with_slippage(self):
        engine = _make_engine(entry_bps=2.0, exit_bps=2.0)
        fill = engine.place_order("MARUTI", 10, "LONG", Decimal("10000"), T0)
        assert fill.filled_price != fill.quoted_price
        assert fill.filled_price > fill.quoted_price  # LONG entry fills higher

    def test_slippage_reduces_net_pnl_vs_zero_slippage(self):
        initial = Decimal("500000")
        engine_zero = _make_engine(initial_cash=initial, entry_bps=0.0, exit_bps=0.0)
        engine_slip = _make_engine(initial_cash=initial, entry_bps=2.0, exit_bps=2.0)

        for engine in [engine_zero, engine_slip]:
            engine.place_order("MARUTI", 10, "LONG", Decimal("10000"), T0)

        c_zero = engine_zero.close_position("MARUTI", Decimal("10100"), ExitReason.TARGET_HIT, T1)
        c_slip = engine_slip.close_position("MARUTI", Decimal("10100"), ExitReason.TARGET_HIT, T1)

        assert c_slip.net_pnl < c_zero.net_pnl

    def test_portfolio_invariant_holds_with_slippage(self):
        initial = Decimal("500000")
        engine = _make_engine(initial_cash=initial, entry_bps=2.0, exit_bps=2.0)
        engine.place_order("RELIANCE", 100, "LONG", Decimal("2500"), T0)
        closed = engine.close_position("RELIANCE", Decimal("2520"), ExitReason.TARGET_HIT, T1)
        assert engine.portfolio.cash == initial + closed.net_pnl

    def test_slippage_estimate_on_closed_trade(self):
        engine = _make_engine(entry_bps=2.0, exit_bps=2.0)
        engine.place_order("INFY", 50, "LONG", Decimal("1800"), T0)
        closed = engine.close_position("INFY", Decimal("1820"), ExitReason.TARGET_HIT, T1)
        assert closed.slippage.is_estimate is True
        assert closed.slippage.total_cost > Decimal("0")
        assert closed.slippage.model_name == "fixed_bps"

    def test_slippage_applied_once_per_leg_not_twice(self):
        """Engine calls apply_exit() once; estimate uses that fill, not a second call.

        A counting SlippageModel returns different values on each apply_exit() call.
        With estimate_from_fills(), the stored SlippageEstimate.exit_filled and the
        actual exit_filled_price used in cost computation must always agree — even if
        repeated calls to apply_exit() would produce different results.
        """
        class _CountingExitSlippage(SlippageModel):
            def __init__(self) -> None:
                self.exit_call_count = 0

            @property
            def model_name(self) -> str:
                return "counting_test"

            def apply_entry(self, direction, price):
                return price  # no entry slippage

            def apply_exit(self, direction, price):
                self.exit_call_count += 1
                # 1st call: 2 bps; 2nd call: 50 bps — obviously inconsistent
                bps = Decimal("2") if self.exit_call_count == 1 else Decimal("50")
                if direction == "LONG":
                    return price * (Decimal("1") - bps / Decimal("10000"))
                return price * (Decimal("1") + bps / Decimal("10000"))

        counting = _CountingExitSlippage()
        cfg = load_config().trading.costs
        engine = PaperExecutionEngine(Decimal("500000"), CostSimulator(cfg), counting)

        engine.place_order("RELIANCE", 100, "LONG", Decimal("2500"), T0)
        closed = engine.close_position("RELIANCE", Decimal("2520"), ExitReason.TARGET_HIT, T1)

        # apply_exit must have been called exactly once
        assert counting.exit_call_count == 1, (
            f"apply_exit called {counting.exit_call_count} times; should be 1"
        )
        # Stored estimate fill must match the actual fill used for cost computation
        assert closed.slippage.exit_filled == closed.exit_filled_price, (
            "SlippageEstimate.exit_filled diverged from the fill used in CostSimulator"
        )
        # Costs were computed on the actual fill (first call = 2 bps)
        assert closed.trade_result.exit_price == closed.exit_filled_price

    def test_round_trip_with_slippage_costs_computed_on_filled_prices(self):
        """CostSimulator receives slippage-adjusted prices, not quoted prices."""
        cfg = load_config().trading.costs
        sim = CostSimulator(cfg)
        slip = FixedBpsSlippage(entry_bps=2.0, exit_bps=2.0)
        engine = PaperExecutionEngine(Decimal("500000"), sim, slip)

        quoted_entry = Decimal("2500")
        quoted_exit = Decimal("2520")

        engine.place_order("RELIANCE", 100, "LONG", quoted_entry, T0)
        closed = engine.close_position("RELIANCE", quoted_exit, ExitReason.TARGET_HIT, T1)

        # Manually derive what CostSimulator should have received
        entry_filled = slip.apply_entry("LONG", quoted_entry)
        exit_filled = slip.apply_exit("LONG", quoted_exit)
        expected = sim.compute("RELIANCE", 100, "LONG", entry_filled, exit_filled)

        assert closed.net_pnl == expected.net_pnl


# ── Interface swap proof ──────────────────────────────────────────────────────

class TestInterfaceSwap:
    """A minimal alternative engine satisfying ExecutionEngine can replace PaperExecutionEngine."""

    def test_mock_engine_satisfies_abc(self):
        class MockAlternativeEngine(ExecutionEngine):
            """Trivial stub: records calls, never modifies cash."""

            def __init__(self) -> None:
                self._port = Portfolio(
                    cash=Decimal("0"),
                    open_positions={},
                    realized_pnl=Decimal("0"),
                    closed_trades=[],
                )

            def place_order(
                self, symbol, qty, direction, price, timestamp
            ) -> FillResult:
                return FillResult(
                    symbol=symbol,
                    qty=qty,
                    direction=direction,
                    quoted_price=price,
                    filled_price=price,
                    timestamp=timestamp,
                )

            def close_position(self, symbol, exit_price, exit_reason, timestamp) -> ClosedTrade:
                raise NotImplementedError("mock only tests place_order")

            @property
            def portfolio(self) -> Portfolio:
                return self._port

            def has_position(self, symbol: str) -> bool:
                return False

        engine: ExecutionEngine = MockAlternativeEngine()

        fill = engine.place_order("TEST", 1, "LONG", Decimal("100"), T0)
        assert fill.symbol == "TEST"
        assert fill.filled_price == Decimal("100")
        assert engine.portfolio.cash == Decimal("0")
        assert not engine.has_position("TEST")

    def test_abstract_engine_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            ExecutionEngine()  # type: ignore[abstract]


# ── FillResult and ClosedTrade dataclasses ────────────────────────────────────

class TestDataclasses:
    def test_fill_result_is_paper_true_by_default(self):
        engine = _make_engine()
        fill = engine.place_order("RELIANCE", 10, "LONG", Decimal("2500"), T0)
        assert fill.is_paper is True

    def test_closed_trade_net_pnl_invariant_prevents_bad_construction(self):
        """Cannot construct ClosedTrade where net_pnl != trade_result.net_pnl."""
        cfg = load_config().trading.costs
        result = CostSimulator(cfg).compute(
            "TEST", 100, "LONG", Decimal("2500"), Decimal("2520")
        )
        from src.trading.slippage import FixedBpsSlippage, SlippageEstimate
        slip_model = FixedBpsSlippage(0.0, 0.0)
        slip_est = slip_model.estimate("LONG", 100, Decimal("2500"), Decimal("2520"))
        fill = FillResult(
            symbol="TEST", qty=100, direction="LONG",
            quoted_price=Decimal("2500"), filled_price=Decimal("2500"),
            timestamp=T0,
        )
        with pytest.raises(ValueError, match="Invariant violation"):
            ClosedTrade(
                entry_fill=fill,
                exit_quoted_price=Decimal("2520"),
                exit_filled_price=Decimal("2520"),
                exit_reason=ExitReason.TARGET_HIT,
                trade_result=result,
                slippage=slip_est,
                net_pnl=Decimal("9999"),  # wrong — does not match trade_result.net_pnl
            )

    def test_open_position_unrealized_pnl_long(self):
        engine = _make_engine()
        engine.place_order("RELIANCE", 100, "LONG", Decimal("2500"), T0)
        pos = engine.portfolio.open_positions["RELIANCE"]
        assert pos.unrealized_pnl(Decimal("2520")) == Decimal("2000")
        assert pos.unrealized_pnl(Decimal("2480")) == Decimal("-2000")

    def test_open_position_unrealized_pnl_short(self):
        engine = _make_engine()
        engine.place_order("INFY", 50, "SHORT", Decimal("1800"), T0)
        pos = engine.portfolio.open_positions["INFY"]
        assert pos.unrealized_pnl(Decimal("1780")) == Decimal("1000")   # profit: price fell
        assert pos.unrealized_pnl(Decimal("1820")) == Decimal("-1000")  # loss: price rose
