"""tests/test_costs.py — hand-verified cost simulation tests.

Default rates used in all scenarios unless stated otherwise:
  brokerage  : percentage model, min(0.03% × turnover, ₹20) per leg
  STT        : 0.025% on sell leg
  exchange   : 0.00297% per leg (NSE equity cash FY2024-25; both legs charged)
  SEBI fees  : 0.0001% on total turnover (₹10 per crore)
  GST        : 18% on (brokerage + exchange + SEBI); NOT on STT or stamp
  stamp duty : 0.003% on buy leg only

Each scenario shows full working so you can verify every rupee independently.
"""
from __future__ import annotations

from decimal import Decimal

import pytest

from src.config import CostConfig
from src.trading.costs import CostSimulator, TradeCostBreakdown, TradeResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def _default_sim() -> CostSimulator:
    """CostSimulator with the standard percentage-brokerage config."""
    return CostSimulator(CostConfig())   # all pydantic defaults


def _flat_sim() -> CostSimulator:
    """CostSimulator with flat ₹20-per-order brokerage."""
    return CostSimulator(CostConfig(brokerage_model="flat"))


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 1 — LONG, profitable, brokerage hits ₹20 cap
# ══════════════════════════════════════════════════════════════════════════════
#
#   Symbol:  RELIANCE   Qty: 100   Direction: LONG
#   Entry:   ₹2,500     Exit: ₹2,520
#
#   Buy  turnover = 100 × 2500 = ₹2,50,000
#   Sell turnover = 100 × 2520 = ₹2,52,000
#   Total turnover                ₹5,02,000
#
#   Brokerage (buy) : min(0.03% × 2,50,000, 20) = min(75.00, 20)  = ₹20.00
#   Brokerage (sell): min(0.03% × 2,52,000, 20) = min(75.60, 20)  = ₹20.00
#   STT              : 0.025% × 2,52,000                           = ₹63.00
#   Exchange charges : 0.00297% × 5,02,000 = 2.97 × 5.02          = ₹14.9094
#   SEBI fees        : 0.0001%  × 5,02,000                         = ₹0.502
#   GST base         : 20 + 20 + 14.9094 + 0.502                   = ₹55.4114
#   GST (18%)        : 0.18 × 55.4114                              = ₹9.974052
#   Stamp duty       : 0.003% × 2,50,000                           = ₹7.50
#   ─────────────────────────────────────────────────────────────────────────
#   Total costs      : 20 + 20 + 63 + 14.9094 + 0.502 + 9.974052 + 7.5
#                    = ₹135.885452
#   Gross P&L        : 100 × (2520 - 2500)                        = ₹2,000.00
#   Net P&L          : 2000 - 135.885452                           = ₹1,864.114548

class TestScenario1LongProfit:
    def test_brokerage_both_legs_capped(self) -> None:
        result = _default_sim().compute(
            "RELIANCE", qty=100, direction="LONG",
            entry_price=Decimal("2500"), exit_price=Decimal("2520"),
        )
        assert result.costs.brokerage_buy == Decimal("20")
        assert result.costs.brokerage_sell == Decimal("20")
        assert result.costs.total_brokerage == Decimal("40")

    def test_stt_on_sell_leg_only(self) -> None:
        result = _default_sim().compute(
            "RELIANCE", qty=100, direction="LONG",
            entry_price=Decimal("2500"), exit_price=Decimal("2520"),
        )
        # 0.025% × 252,000 = 63 exactly
        assert result.costs.stt == Decimal("63")

    def test_exchange_charges_both_legs(self) -> None:
        result = _default_sim().compute(
            "RELIANCE", qty=100, direction="LONG",
            entry_price=Decimal("2500"), exit_price=Decimal("2520"),
        )
        # 0.0000297 × 502,000 = 2.97 × 5.02 = 14.9094
        assert result.costs.exchange_charges == Decimal("14.9094")

    def test_sebi_fees(self) -> None:
        result = _default_sim().compute(
            "RELIANCE", qty=100, direction="LONG",
            entry_price=Decimal("2500"), exit_price=Decimal("2520"),
        )
        # 0.000001 × 502,000 = 0.502
        assert result.costs.sebi_fees == Decimal("0.502")

    def test_gst_on_correct_base(self) -> None:
        result = _default_sim().compute(
            "RELIANCE", qty=100, direction="LONG",
            entry_price=Decimal("2500"), exit_price=Decimal("2520"),
        )
        # base = 40 + 14.9094 + 0.502 = 55.4114; GST = 0.18 × 55.4114 = 9.974052
        assert result.costs.gst == Decimal("9.974052")

    def test_stamp_duty_on_buy_leg_only(self) -> None:
        result = _default_sim().compute(
            "RELIANCE", qty=100, direction="LONG",
            entry_price=Decimal("2500"), exit_price=Decimal("2520"),
        )
        # 0.003% × 250,000 = 7.5
        assert result.costs.stamp_duty == Decimal("7.5")

    def test_total_costs(self) -> None:
        result = _default_sim().compute(
            "RELIANCE", qty=100, direction="LONG",
            entry_price=Decimal("2500"), exit_price=Decimal("2520"),
        )
        assert result.costs.total == Decimal("135.885452")

    def test_gross_pnl(self) -> None:
        result = _default_sim().compute(
            "RELIANCE", qty=100, direction="LONG",
            entry_price=Decimal("2500"), exit_price=Decimal("2520"),
        )
        assert result.gross_pnl == Decimal("2000")

    def test_net_pnl(self) -> None:
        result = _default_sim().compute(
            "RELIANCE", qty=100, direction="LONG",
            entry_price=Decimal("2500"), exit_price=Decimal("2520"),
        )
        assert result.net_pnl == Decimal("1864.114548")

    def test_net_is_less_than_gross(self) -> None:
        result = _default_sim().compute(
            "RELIANCE", qty=100, direction="LONG",
            entry_price=Decimal("2500"), exit_price=Decimal("2520"),
        )
        assert result.net_pnl < result.gross_pnl

    def test_turnover_fields(self) -> None:
        result = _default_sim().compute(
            "RELIANCE", qty=100, direction="LONG",
            entry_price=Decimal("2500"), exit_price=Decimal("2520"),
        )
        assert result.buy_turnover == Decimal("250000")
        assert result.sell_turnover == Decimal("252000")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 2 — SHORT, profitable, brokerage uncapped
# ══════════════════════════════════════════════════════════════════════════════
#
#   Symbol:  INFY   Qty: 50   Direction: SHORT
#   Entry (opening sell): ₹1,000   Exit (closing buy): ₹990
#
#   Sell turnover = 50 × 1000 = ₹50,000   (opening sell leg)
#   Buy  turnover = 50 ×  990 = ₹49,500   (closing buy leg)
#   Total                        ₹99,500
#
#   Brokerage (sell): min(0.03% × 50,000, 20) = min(15.00, 20)    = ₹15.00
#   Brokerage (buy) : min(0.03% × 49,500, 20) = min(14.85, 20)    = ₹14.85
#   STT              : 0.025% × 50,000  (opening sell)             = ₹12.50
#   Exchange charges : 0.00297% × 99,500 = 2.97 × 9.95 / 10       = ₹2.95515
#   SEBI fees        : 0.0001% × 99,500                            = ₹0.0995
#   GST base         : 15 + 14.85 + 2.95515 + 0.0995              = ₹32.90465
#   GST (18%)        : 0.18 × 32.90465                             = ₹5.922837
#   Stamp duty       : 0.003% × 49,500  (closing buy)              = ₹1.485
#   ─────────────────────────────────────────────────────────────────────────
#   Total costs  : 15 + 14.85 + 12.5 + 2.95515 + 0.0995 + 5.922837 + 1.485
#               = ₹52.812487
#   Gross P&L    : 50 × (1000 - 990)                               = ₹500.00
#   Net P&L      : 500 - 52.812487                                  = ₹447.187513

class TestScenario2ShortProfit:
    def test_stt_on_opening_sell_leg(self) -> None:
        result = _default_sim().compute(
            "INFY", qty=50, direction="SHORT",
            entry_price=Decimal("1000"), exit_price=Decimal("990"),
        )
        # Opening sell = entry for SHORT; 0.025% × 50,000 = 12.5
        assert result.costs.stt == Decimal("12.5")
        assert result.sell_turnover == Decimal("50000")

    def test_stamp_duty_on_closing_buy_leg(self) -> None:
        result = _default_sim().compute(
            "INFY", qty=50, direction="SHORT",
            entry_price=Decimal("1000"), exit_price=Decimal("990"),
        )
        # Closing buy = exit for SHORT; 0.003% × 49,500 = 1.485
        assert result.costs.stamp_duty == Decimal("1.485")
        assert result.buy_turnover == Decimal("49500")

    def test_brokerage_uncapped(self) -> None:
        result = _default_sim().compute(
            "INFY", qty=50, direction="SHORT",
            entry_price=Decimal("1000"), exit_price=Decimal("990"),
        )
        # Sell: 0.03% × 50,000 = 15 < 20 cap → 15
        # Buy:  0.03% × 49,500 = 14.85 < 20 cap → 14.85
        assert result.costs.brokerage_sell == Decimal("15")
        assert result.costs.brokerage_buy == Decimal("14.85")

    def test_exchange_charges(self) -> None:
        result = _default_sim().compute(
            "INFY", qty=50, direction="SHORT",
            entry_price=Decimal("1000"), exit_price=Decimal("990"),
        )
        # 0.0000297 × 99,500 = 2.97 × 9.95 / 10 = 29.5515 / 10 = 2.95515
        assert result.costs.exchange_charges == Decimal("2.95515")

    def test_gst(self) -> None:
        result = _default_sim().compute(
            "INFY", qty=50, direction="SHORT",
            entry_price=Decimal("1000"), exit_price=Decimal("990"),
        )
        # base = 29.85 + 2.95515 + 0.0995 = 32.90465
        # GST  = 0.18 × 32.90465 = 5.922837
        assert result.costs.gst == Decimal("5.922837")

    def test_total_costs(self) -> None:
        result = _default_sim().compute(
            "INFY", qty=50, direction="SHORT",
            entry_price=Decimal("1000"), exit_price=Decimal("990"),
        )
        assert result.costs.total == Decimal("52.812487")

    def test_gross_and_net_pnl(self) -> None:
        result = _default_sim().compute(
            "INFY", qty=50, direction="SHORT",
            entry_price=Decimal("1000"), exit_price=Decimal("990"),
        )
        assert result.gross_pnl == Decimal("500")
        assert result.net_pnl == Decimal("447.187513")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 3 — LONG, losing trade; costs deepen the loss
# ══════════════════════════════════════════════════════════════════════════════
#
#   Symbol:  TCS   Qty: 20   Direction: LONG
#   Entry:   ₹5,000   Exit: ₹4,950   (₹50 adverse move per share)
#
#   Buy  turnover = 20 × 5000 = ₹1,00,000
#   Sell turnover = 20 × 4950 = ₹99,000
#   Total                        ₹1,99,000
#
#   Brokerage (buy) : min(0.03% × 1,00,000, 20) = min(30, 20)     = ₹20.00
#   Brokerage (sell): min(0.03% × 99,000,   20) = min(29.7, 20)   = ₹20.00
#   STT              : 0.025% × 99,000                             = ₹24.75
#   Exchange charges : 0.00297% × 1,99,000 = 2.97 × 1.99          = ₹5.9103
#   SEBI fees        : 0.0001% × 1,99,000                          = ₹0.199
#   GST base         : 20 + 20 + 5.9103 + 0.199                   = ₹46.1093
#   GST (18%)        : 0.18 × 46.1093                              = ₹8.299674
#   Stamp duty       : 0.003% × 1,00,000                           = ₹3.00
#   ─────────────────────────────────────────────────────────────────────────
#   Total costs  : 20 + 20 + 24.75 + 5.9103 + 0.199 + 8.299674 + 3
#               = ₹82.158974
#   Gross P&L    : 20 × (4950 - 5000) = 20 × -50                  = -₹1,000
#   Net P&L      : -1000 - 82.158974                               = -₹1,082.158974
#   (costs turn a -₹50/share loss into -₹54.11/share effective loss)

class TestScenario3LongLoss:
    def test_gross_pnl_is_negative(self) -> None:
        result = _default_sim().compute(
            "TCS", qty=20, direction="LONG",
            entry_price=Decimal("5000"), exit_price=Decimal("4950"),
        )
        assert result.gross_pnl == Decimal("-1000")

    def test_costs_total(self) -> None:
        result = _default_sim().compute(
            "TCS", qty=20, direction="LONG",
            entry_price=Decimal("5000"), exit_price=Decimal("4950"),
        )
        assert result.costs.stt == Decimal("24.75")
        assert result.costs.exchange_charges == Decimal("5.9103")
        assert result.costs.sebi_fees == Decimal("0.199")
        assert result.costs.gst == Decimal("8.299674")
        assert result.costs.stamp_duty == Decimal("3")
        assert result.costs.total == Decimal("82.158974")

    def test_net_pnl_worse_than_gross(self) -> None:
        result = _default_sim().compute(
            "TCS", qty=20, direction="LONG",
            entry_price=Decimal("5000"), exit_price=Decimal("4950"),
        )
        assert result.net_pnl == Decimal("-1082.158974")
        assert result.net_pnl < result.gross_pnl   # costs deepen the loss


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIO 4 — Tiny trade: percentage vs flat brokerage model divergence
# ══════════════════════════════════════════════════════════════════════════════
#
#   Symbol:  COALINDIA   Qty: 5   Direction: LONG
#   Entry:   ₹100   Exit: ₹102   Gross P&L: ₹10
#
#   Turnover: buy=500, sell=510, total=1,010
#
#   ── Percentage model (min 0.03% × turnover, cap ₹20) ────────────────────
#   Brokerage (buy)  : min(0.03% × 500, 20) = min(0.15, 20)        = ₹0.15
#   Brokerage (sell) : min(0.03% × 510, 20) = min(0.153, 20)       = ₹0.153
#   STT              : 0.025% × 510                                 = ₹0.1275
#   Exchange charges : 0.00297% × 1,010 = 2.97 × 1.01 / 100        = ₹0.029997
#   SEBI fees        : 0.0001% × 1,010                              = ₹0.00101
#   GST base         : 0.303 + 0.029997 + 0.00101                  = ₹0.334007
#   GST (18%)        : 0.18 × 0.334007                              = ₹0.06012126
#   Stamp duty       : 0.003% × 500                                 = ₹0.015
#   Total costs (pct): ₹0.53662826
#   Net P&L (pct)    : 10 - 0.53662826                              = ₹9.46337174
#
#   ── Flat model (always ₹20 per leg) ─────────────────────────────────────
#   Brokerage (buy)  : ₹20.00
#   Brokerage (sell) : ₹20.00
#   STT + exchange + sebi + stamp: same as above (₹0.1275 + 0.029997 + 0.00101 + 0.015)
#   GST base (flat)  : 40 + 0.029997 + 0.00101                     = ₹40.031007
#   GST (18%, flat)  : 0.18 × 40.031007                            = ₹7.20558126
#   Total costs (flat): 40 + 0.1275 + 0.029997 + 0.00101 + 7.20558126 + 0.015
#                     = ₹47.37908826
#   Net P&L (flat)   : 10 - 47.37908826                            = -₹37.37908826
#   (flat brokerage turns a ₹10 gross profit into a ₹37 net loss)

class TestScenario4BrokerageModelDivergence:
    def test_percentage_model_brokerage_uncapped(self) -> None:
        result = _default_sim().compute(
            "COALINDIA", qty=5, direction="LONG",
            entry_price=Decimal("100"), exit_price=Decimal("102"),
        )
        assert result.costs.brokerage_buy == Decimal("0.15")
        assert result.costs.brokerage_sell == Decimal("0.153")

    def test_percentage_model_total_costs(self) -> None:
        result = _default_sim().compute(
            "COALINDIA", qty=5, direction="LONG",
            entry_price=Decimal("100"), exit_price=Decimal("102"),
        )
        assert result.costs.exchange_charges == Decimal("0.029997")
        assert result.costs.sebi_fees == Decimal("0.00101")
        assert result.costs.gst == Decimal("0.06012126")
        assert result.costs.stamp_duty == Decimal("0.015")
        assert result.costs.total == Decimal("0.53662826")

    def test_percentage_model_net_pnl(self) -> None:
        result = _default_sim().compute(
            "COALINDIA", qty=5, direction="LONG",
            entry_price=Decimal("100"), exit_price=Decimal("102"),
        )
        assert result.gross_pnl == Decimal("10")
        assert result.net_pnl == Decimal("9.46337174")

    def test_flat_model_brokerage_always_cap(self) -> None:
        result = _flat_sim().compute(
            "COALINDIA", qty=5, direction="LONG",
            entry_price=Decimal("100"), exit_price=Decimal("102"),
        )
        assert result.costs.brokerage_buy == Decimal("20")
        assert result.costs.brokerage_sell == Decimal("20")

    def test_flat_model_gst_on_larger_base(self) -> None:
        result = _flat_sim().compute(
            "COALINDIA", qty=5, direction="LONG",
            entry_price=Decimal("100"), exit_price=Decimal("102"),
        )
        # GST base = 40 + 0.029997 + 0.00101 = 40.031007
        # GST = 0.18 × 40.031007 = 7.20558126
        assert result.costs.gst == Decimal("7.20558126")

    def test_flat_model_total_costs(self) -> None:
        result = _flat_sim().compute(
            "COALINDIA", qty=5, direction="LONG",
            entry_price=Decimal("100"), exit_price=Decimal("102"),
        )
        assert result.costs.total == Decimal("47.37908826")

    def test_flat_model_turns_profit_into_loss(self) -> None:
        result = _flat_sim().compute(
            "COALINDIA", qty=5, direction="LONG",
            entry_price=Decimal("100"), exit_price=Decimal("102"),
        )
        assert result.gross_pnl == Decimal("10")
        assert result.net_pnl == Decimal("-37.37908826")
        assert result.net_pnl < 0   # ₹10 gross → ₹37 net loss due to fixed overhead

    def test_percentage_cheaper_than_flat_for_small_trades(self) -> None:
        trade = dict(symbol="COALINDIA", qty=5, direction="LONG",
                     entry_price=Decimal("100"), exit_price=Decimal("102"))
        pct_result = _default_sim().compute(**trade)
        flat_result = _flat_sim().compute(**trade)
        assert pct_result.costs.total_brokerage < flat_result.costs.total_brokerage
        assert pct_result.net_pnl > flat_result.net_pnl


# ══════════════════════════════════════════════════════════════════════════════
# STRUCTURAL INVARIANTS
# ══════════════════════════════════════════════════════════════════════════════

class TestStructuralInvariants:
    def test_net_pnl_invariant_enforced(self) -> None:
        """TradeResult rejects construction with wrong net_pnl."""
        costs = TradeCostBreakdown(
            brokerage_buy=Decimal("20"), brokerage_sell=Decimal("20"),
            stt=Decimal("63"), exchange_charges=Decimal("14.9094"),
            sebi_fees=Decimal("0.502"), gst=Decimal("9.974052"),
            stamp_duty=Decimal("7.5"),
        )
        with pytest.raises(ValueError, match="Invariant violation"):
            TradeResult(
                symbol="RELIANCE", qty=100, direction="LONG",
                entry_price=Decimal("2500"), exit_price=Decimal("2520"),
                buy_turnover=Decimal("250000"), sell_turnover=Decimal("252000"),
                gross_pnl=Decimal("2000"),
                costs=costs,
                net_pnl=Decimal("2000"),  # wrong: ignores costs entirely
            )

    def test_costs_total_is_sum_of_components(self) -> None:
        result = _default_sim().compute(
            "RELIANCE", qty=100, direction="LONG",
            entry_price=Decimal("2500"), exit_price=Decimal("2520"),
        )
        c = result.costs
        manual_total = (
            c.brokerage_buy + c.brokerage_sell
            + c.stt + c.exchange_charges + c.sebi_fees + c.gst + c.stamp_duty
        )
        assert c.total == manual_total

    def test_net_pnl_equals_gross_minus_total(self) -> None:
        result = _default_sim().compute(
            "INFY", qty=50, direction="SHORT",
            entry_price=Decimal("1000"), exit_price=Decimal("990"),
        )
        assert result.net_pnl == result.gross_pnl - result.costs.total

    def test_gst_not_applied_to_stt_or_stamp(self) -> None:
        """GST base must be brokerage + exchange + SEBI, never including STT/stamp."""
        result = _default_sim().compute(
            "RELIANCE", qty=100, direction="LONG",
            entry_price=Decimal("2500"), exit_price=Decimal("2520"),
        )
        c = result.costs
        expected_gst_base = c.total_brokerage + c.exchange_charges + c.sebi_fees
        expected_gst = expected_gst_base * Decimal("0.18")
        assert c.gst == expected_gst

    def test_stt_always_on_sell_leg_regardless_of_direction(self) -> None:
        """For LONG, sell leg = exit. For SHORT, sell leg = entry. STT follows the sell."""
        long_r = _default_sim().compute(
            "X", qty=10, direction="LONG",
            entry_price=Decimal("100"), exit_price=Decimal("110"),
        )
        short_r = _default_sim().compute(
            "X", qty=10, direction="SHORT",
            entry_price=Decimal("110"), exit_price=Decimal("100"),
        )
        # LONG: STT on exit (110); SHORT: STT on entry (110) — same sell price, same STT
        assert long_r.costs.stt == short_r.costs.stt
        assert long_r.costs.stt == Decimal("10") * Decimal("110") * Decimal("0.00025")

    def test_stamp_always_on_buy_leg_regardless_of_direction(self) -> None:
        """For LONG, buy leg = entry. For SHORT, buy leg = exit. Stamp follows the buy."""
        long_r = _default_sim().compute(
            "X", qty=10, direction="LONG",
            entry_price=Decimal("100"), exit_price=Decimal("110"),
        )
        short_r = _default_sim().compute(
            "X", qty=10, direction="SHORT",
            entry_price=Decimal("110"), exit_price=Decimal("100"),
        )
        # LONG: stamp on entry (100); SHORT: stamp on exit (100) — same buy price, same stamp
        assert long_r.costs.stamp_duty == short_r.costs.stamp_duty
        assert long_r.costs.stamp_duty == Decimal("10") * Decimal("100") * Decimal("0.00003")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG LOADING
# ══════════════════════════════════════════════════════════════════════════════

class TestConfigLoading:
    def test_cost_config_loads_from_toml(self) -> None:
        from src.config import load_config
        cfg = load_config()
        c = cfg.trading.costs
        assert c.brokerage_model == "percentage"
        assert c.brokerage_pct == pytest.approx(0.0003)
        assert c.brokerage_flat_cap == pytest.approx(20.0)
        assert c.stt_intraday_sell_pct == pytest.approx(0.00025)
        assert c.exchange_charges_pct == pytest.approx(0.0000297)
        assert c.sebi_fees_pct == pytest.approx(0.000001)
        assert c.gst_pct == pytest.approx(0.18)
        assert c.stamp_duty_intraday_buy_pct == pytest.approx(0.00003)

    def test_cost_config_from_toml_produces_correct_scenario1(self) -> None:
        """Verify config-loaded simulator gives the same result as the direct-config sim."""
        from src.config import load_config
        cfg_sim = CostSimulator(load_config().trading.costs)
        default_sim = _default_sim()
        kwargs = dict(
            symbol="RELIANCE", qty=100, direction="LONG",
            entry_price=Decimal("2500"), exit_price=Decimal("2520"),
        )
        assert cfg_sim.compute(**kwargs).net_pnl == default_sim.compute(**kwargs).net_pnl

    def test_negative_rate_rejected(self) -> None:
        with pytest.raises(Exception):
            CostConfig(stt_intraday_sell_pct=-0.001)

    def test_as_dict_has_all_keys(self) -> None:
        result = _default_sim().compute(
            "RELIANCE", qty=100, direction="LONG",
            entry_price=Decimal("2500"), exit_price=Decimal("2520"),
        )
        d = result.costs.as_dict()
        for key in ("brokerage_buy", "brokerage_sell", "total_brokerage",
                    "stt", "exchange_charges", "sebi_fees", "gst", "stamp_duty", "total"):
            assert key in d
