"""Cost simulation for NSE intraday equity (MIS) round-trip trades.

Covers the full realistic cost stack, computed per-leg correctly:

  Component            Leg(s)         Rate (default config)
  ─────────────────────────────────────────────────────────
  Brokerage            buy + sell     min(0.03% × turnover, ₹20) each
  STT                  sell only      0.025% of sell-leg turnover
  Exchange charges     buy + sell     0.00297% of each leg's turnover
  SEBI fees            total turnover 0.0001% of (buy + sell) turnover
  GST                  derived        18% on (brokerage + exchange + SEBI)
  Stamp duty           buy only       0.003% of buy-leg turnover

All rates are read from CostConfig (config.toml [trading.costs]).
Calculations use Python's decimal.Decimal to avoid floating-point drift.

Usage
-----
    from src.config import load_config
    from src.trading.costs import CostSimulator
    from decimal import Decimal

    sim = CostSimulator(load_config().trading.costs)
    result = sim.compute("RELIANCE", qty=100, direction="LONG",
                         entry_price=Decimal("2500"), exit_price=Decimal("2520"))
    print(result.net_pnl)   # always gross_pnl - costs.total
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from src.config import CostConfig
from src.trading._types import Direction

# Tolerance for the __post_init__ net_pnl invariant check (sub-paise).
_NET_PNL_TOLERANCE = Decimal("0.001")


@dataclass(frozen=True)
class TradeCostBreakdown:
    """All cost components for a single intraday round-trip trade.

    Fields are the seven individually-auditable line items.
    Use .total_brokerage and .total (both are properties) to aggregate —
    no other aggregation method exists, preventing accidental mis-combination.
    """
    brokerage_buy: Decimal
    brokerage_sell: Decimal
    stt: Decimal                # 0.025% on sell leg only
    exchange_charges: Decimal   # NSE charge on buy + sell combined
    sebi_fees: Decimal          # on total (buy + sell) turnover
    gst: Decimal                # 18% on brokerage + exchange + sebi
    stamp_duty: Decimal         # 0.003% on buy leg only

    @property
    def total_brokerage(self) -> Decimal:
        return self.brokerage_buy + self.brokerage_sell

    @property
    def total(self) -> Decimal:
        return (
            self.total_brokerage
            + self.stt
            + self.exchange_charges
            + self.sebi_fees
            + self.gst
            + self.stamp_duty
        )

    def as_dict(self) -> dict[str, Decimal]:
        """All components and aggregates as a flat dict (for logging/display)."""
        return {
            "brokerage_buy": self.brokerage_buy,
            "brokerage_sell": self.brokerage_sell,
            "total_brokerage": self.total_brokerage,
            "stt": self.stt,
            "exchange_charges": self.exchange_charges,
            "sebi_fees": self.sebi_fees,
            "gst": self.gst,
            "stamp_duty": self.stamp_duty,
            "total": self.total,
        }


@dataclass(frozen=True)
class TradeResult:
    """Result of a completed intraday round-trip trade, net of all costs.

    net_pnl is ALWAYS gross_pnl - costs.total.  __post_init__ enforces this
    as a hard invariant: it is structurally impossible to create a TradeResult
    where net_pnl is inconsistent with gross_pnl and costs.  Gross P&L is
    retained as a breakdown field only — it must never be used as a headline.
    """
    symbol: str
    qty: int
    direction: Direction
    entry_price: Decimal
    exit_price: Decimal
    buy_turnover: Decimal
    sell_turnover: Decimal
    gross_pnl: Decimal
    costs: TradeCostBreakdown
    net_pnl: Decimal

    def __post_init__(self) -> None:
        expected_net = self.gross_pnl - self.costs.total
        if abs(expected_net - self.net_pnl) > _NET_PNL_TOLERANCE:
            raise ValueError(
                f"Invariant violation: net_pnl={self.net_pnl} does not equal "
                f"gross_pnl({self.gross_pnl}) - costs.total({self.costs.total}) "
                f"= {expected_net}.  Never construct TradeResult manually; "
                f"use CostSimulator.compute() instead."
            )


class CostSimulator:
    """Compute the full NSE intraday equity cost stack for a round-trip trade.

    Instantiate once with a CostConfig and call .compute() per trade.
    No state is mutated between calls; safe to share across threads.
    """

    def __init__(self, cost_config: CostConfig) -> None:
        self._cfg = cost_config

    def compute(
        self,
        symbol: str,
        qty: int,
        direction: Direction,
        entry_price: Decimal,
        exit_price: Decimal,
    ) -> TradeResult:
        """Compute all costs and return a TradeResult with verified net P&L.

        For LONG:  entry = buy leg, exit = sell leg.
        For SHORT: entry = opening sell leg, exit = closing buy leg.

        STT applies to the sell leg; stamp duty to the buy leg.
        Both rules hold regardless of direction.
        """
        cfg = self._cfg
        qty_d = Decimal(qty)

        # Map direction to buy/sell prices
        if direction == "LONG":
            buy_price, sell_price = entry_price, exit_price
        else:  # SHORT: open with a sell, close with a buy
            sell_price, buy_price = entry_price, exit_price

        buy_turnover = qty_d * buy_price
        sell_turnover = qty_d * sell_price
        total_turnover = buy_turnover + sell_turnover

        # ── Brokerage ────────────────────────────────────────────────────────
        brokerage_buy = self._brokerage(buy_turnover)
        brokerage_sell = self._brokerage(sell_turnover)

        # ── STT: sell leg only ───────────────────────────────────────────────
        stt = sell_turnover * Decimal(str(cfg.stt_intraday_sell_pct))

        # ── Exchange charges: both legs combined ─────────────────────────────
        exchange_charges = total_turnover * Decimal(str(cfg.exchange_charges_pct))

        # ── SEBI fees: total turnover ────────────────────────────────────────
        sebi_fees = total_turnover * Decimal(str(cfg.sebi_fees_pct))

        # ── GST: 18% on brokerage + exchange + SEBI (NOT on STT/stamp) ──────
        gst_base = brokerage_buy + brokerage_sell + exchange_charges + sebi_fees
        gst = gst_base * Decimal(str(cfg.gst_pct))

        # ── Stamp duty: buy leg only ─────────────────────────────────────────
        stamp_duty = buy_turnover * Decimal(str(cfg.stamp_duty_intraday_buy_pct))

        costs = TradeCostBreakdown(
            brokerage_buy=brokerage_buy,
            brokerage_sell=brokerage_sell,
            stt=stt,
            exchange_charges=exchange_charges,
            sebi_fees=sebi_fees,
            gst=gst,
            stamp_duty=stamp_duty,
        )

        # ── Gross P&L ────────────────────────────────────────────────────────
        if direction == "LONG":
            gross_pnl = qty_d * (exit_price - entry_price)
        else:
            gross_pnl = qty_d * (entry_price - exit_price)

        net_pnl = gross_pnl - costs.total

        return TradeResult(
            symbol=symbol,
            qty=qty,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            buy_turnover=buy_turnover,
            sell_turnover=sell_turnover,
            gross_pnl=gross_pnl,
            costs=costs,
            net_pnl=net_pnl,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _brokerage(self, turnover: Decimal) -> Decimal:
        """Compute one-leg brokerage: flat cap or percentage-with-cap."""
        cfg = self._cfg
        if cfg.brokerage_model == "flat":
            return Decimal(str(cfg.brokerage_flat_cap))
        pct = Decimal(str(cfg.brokerage_pct))
        cap = Decimal(str(cfg.brokerage_flat_cap))
        return min(turnover * pct, cap)
