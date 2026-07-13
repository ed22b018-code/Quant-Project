"""Tests for the slippage model (src/trading/slippage.py).

Covers:
  - Direction correctness: fills always unfavourable
  - Exact bps arithmetic
  - is_estimate invariant (always True; False raises)
  - Zero bps = no slippage
  - estimate() combines both legs correctly
"""
from __future__ import annotations

import pytest
from decimal import Decimal

from src.trading.slippage import FixedBpsSlippage, SlippageEstimate, SlippageModel


# ── Direction correctness ──────────────────────────────────────────────────────

class TestFixedBpsDirection:
    """Fills must always go against the trader."""

    def test_long_entry_fills_higher_than_quoted(self):
        model = FixedBpsSlippage(entry_bps=2.0, exit_bps=0.0)
        quoted = Decimal("1000")
        filled = model.apply_entry("LONG", quoted)
        assert filled > quoted, "LONG entry should fill higher (market impact)"

    def test_long_exit_fills_lower_than_quoted(self):
        model = FixedBpsSlippage(entry_bps=0.0, exit_bps=2.0)
        quoted = Decimal("1000")
        filled = model.apply_exit("LONG", quoted)
        assert filled < quoted, "LONG exit should fill lower (selling at bid)"

    def test_short_entry_fills_lower_than_quoted(self):
        model = FixedBpsSlippage(entry_bps=2.0, exit_bps=0.0)
        quoted = Decimal("1000")
        filled = model.apply_entry("SHORT", quoted)
        assert filled < quoted, "SHORT entry should fill lower (selling at bid)"

    def test_short_exit_fills_higher_than_quoted(self):
        model = FixedBpsSlippage(entry_bps=0.0, exit_bps=2.0)
        quoted = Decimal("1000")
        filled = model.apply_exit("SHORT", quoted)
        assert filled > quoted, "SHORT exit should fill higher (buying at ask)"


# ── Exact bps arithmetic ──────────────────────────────────────────────────────

class TestFixedBpsMath:
    """Verify exact basis-point calculations."""

    def test_2bps_long_entry(self):
        # 2 bps = 0.02%, factor = 1.0002
        # filled = 2500 * 1.0002 = 2500.5
        model = FixedBpsSlippage(entry_bps=2.0, exit_bps=0.0)
        filled = model.apply_entry("LONG", Decimal("2500"))
        expected = Decimal("2500") * (Decimal("1") + Decimal("2") / Decimal("10000"))
        assert filled == expected

    def test_2bps_long_exit(self):
        # filled = 2500 * 0.9998 = 2499.5
        model = FixedBpsSlippage(entry_bps=0.0, exit_bps=2.0)
        filled = model.apply_exit("LONG", Decimal("2500"))
        expected = Decimal("2500") * (Decimal("1") - Decimal("2") / Decimal("10000"))
        assert filled == expected

    def test_5bps_short_entry(self):
        # 5 bps = 0.05%, factor = 0.9995
        # filled = 1000 * 0.9995 = 999.5
        model = FixedBpsSlippage(entry_bps=5.0, exit_bps=0.0)
        filled = model.apply_entry("SHORT", Decimal("1000"))
        expected = Decimal("1000") * (Decimal("1") - Decimal("5") / Decimal("10000"))
        assert filled == expected

    def test_10bps_short_exit(self):
        # 10 bps = 0.1%, filled = 800 * 1.001 = 800.8
        model = FixedBpsSlippage(entry_bps=0.0, exit_bps=10.0)
        filled = model.apply_exit("SHORT", Decimal("800"))
        expected = Decimal("800") * (Decimal("1") + Decimal("10") / Decimal("10000"))
        assert filled == expected

    def test_zero_bps_no_slippage_long(self):
        model = FixedBpsSlippage(entry_bps=0.0, exit_bps=0.0)
        price = Decimal("1500")
        assert model.apply_entry("LONG", price) == price
        assert model.apply_exit("LONG", price) == price

    def test_zero_bps_no_slippage_short(self):
        model = FixedBpsSlippage(entry_bps=0.0, exit_bps=0.0)
        price = Decimal("1500")
        assert model.apply_entry("SHORT", price) == price
        assert model.apply_exit("SHORT", price) == price


# ── is_estimate invariant ─────────────────────────────────────────────────────

class TestIsEstimateInvariant:
    """SlippageEstimate.is_estimate must always be True."""

    def test_estimate_is_always_true(self):
        model = FixedBpsSlippage(entry_bps=2.0, exit_bps=2.0)
        est = model.estimate("LONG", 100, Decimal("2500"), Decimal("2520"))
        assert est.is_estimate is True

    def test_setting_is_estimate_false_raises(self):
        with pytest.raises(ValueError, match="is_estimate must always be True"):
            SlippageEstimate(
                entry_quoted=Decimal("1000"),
                entry_filled=Decimal("1000.2"),
                exit_quoted=Decimal("1010"),
                exit_filled=Decimal("1009.798"),
                entry_cost=Decimal("20"),
                exit_cost=Decimal("20.2"),
                total_cost=Decimal("40.2"),
                model_name="fixed_bps",
                is_estimate=False,
            )

    def test_estimate_from_model_always_true(self):
        """estimate() result from any direction is always marked is_estimate=True."""
        model = FixedBpsSlippage(entry_bps=3.0, exit_bps=3.0)
        for direction in ("LONG", "SHORT"):
            est = model.estimate(direction, 50, Decimal("500"), Decimal("495"))  # type: ignore[arg-type]
            assert est.is_estimate is True


# ── estimate() combining both legs ───────────────────────────────────────────

class TestSlippageEstimate:
    """estimate() must correctly compose entry and exit fills."""

    def test_model_name_propagated(self):
        model = FixedBpsSlippage(entry_bps=2.0, exit_bps=2.0)
        est = model.estimate("LONG", 10, Decimal("100"), Decimal("100"))
        assert est.model_name == "fixed_bps"

    def test_long_estimate_entry_cost(self):
        # LONG entry: filled = 1000 * 1.0002 = 1000.2
        # entry_cost = (1000.2 - 1000) * 100 = 20
        model = FixedBpsSlippage(entry_bps=2.0, exit_bps=0.0)
        est = model.estimate("LONG", 100, Decimal("1000"), Decimal("1010"))
        expected_entry_cost = Decimal("1000") * Decimal("2") / Decimal("10000") * Decimal("100")
        assert est.entry_cost == expected_entry_cost
        assert est.exit_cost == Decimal("0")

    def test_long_estimate_exit_cost(self):
        # LONG exit: filled = 1010 * 0.9998 = 1009.798
        # exit_cost = (1010 - 1009.798) * 100 = 20.2
        model = FixedBpsSlippage(entry_bps=0.0, exit_bps=2.0)
        est = model.estimate("LONG", 100, Decimal("1000"), Decimal("1010"))
        expected_exit_cost = Decimal("1010") * Decimal("2") / Decimal("10000") * Decimal("100")
        assert est.exit_cost == expected_exit_cost
        assert est.entry_cost == Decimal("0")

    def test_total_cost_is_entry_plus_exit(self):
        model = FixedBpsSlippage(entry_bps=2.0, exit_bps=3.0)
        est = model.estimate("LONG", 50, Decimal("2000"), Decimal("2010"))
        assert est.total_cost == est.entry_cost + est.exit_cost

    def test_short_estimate_direction_costs(self):
        # SHORT entry: filled = 2000 * 0.9998 = 1999.6  (sell lower)
        # SHORT exit:  filled = 1980 * 1.0002 = 1980.396 (buy higher)
        # entry_cost = (2000 - 1999.6) * 10 = 4
        # exit_cost  = (1980.396 - 1980) * 10 = 3.96
        model = FixedBpsSlippage(entry_bps=2.0, exit_bps=2.0)
        est = model.estimate("SHORT", 10, Decimal("2000"), Decimal("1980"))
        assert est.entry_filled < est.entry_quoted  # SHORT entry fills lower
        assert est.exit_filled > est.exit_quoted    # SHORT exit fills higher
        assert est.entry_cost > Decimal("0")
        assert est.exit_cost > Decimal("0")
        assert est.total_cost == est.entry_cost + est.exit_cost

    def test_zero_bps_all_fields_equal_quoted(self):
        model = FixedBpsSlippage(entry_bps=0.0, exit_bps=0.0)
        est = model.estimate("LONG", 100, Decimal("1500"), Decimal("1520"))
        assert est.entry_filled == est.entry_quoted
        assert est.exit_filled == est.exit_quoted
        assert est.entry_cost == Decimal("0")
        assert est.exit_cost == Decimal("0")
        assert est.total_cost == Decimal("0")


# ── ABC compliance ─────────────────────────────────────────────────────────────

class TestSlippageModelABC:
    def test_cannot_instantiate_abstract_base(self):
        with pytest.raises(TypeError):
            SlippageModel()  # type: ignore[abstract]

    def test_fixed_bps_is_concrete(self):
        model = FixedBpsSlippage(entry_bps=1.0, exit_bps=1.0)
        assert isinstance(model, SlippageModel)
