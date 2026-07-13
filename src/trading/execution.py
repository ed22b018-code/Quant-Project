"""Paper execution engine for NSE intraday equity simulation.

ExecutionEngine ABC allows a real-broker implementation to be swapped in
without changing any caller. PaperExecutionEngine is the only implementation now.

Cash accounting:
  LONG entry:  cash -= qty * entry_filled_price
  LONG exit:   cash += qty * exit_filled_price - costs.total
  SHORT entry: cash += qty * entry_filled_price
  SHORT exit:  cash -= qty * exit_filled_price + costs.total

Portfolio invariant (maintained continuously):
  cash == initial_cash + sum(net_pnl for all closed trades)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Literal

from src.trading.costs import CostSimulator
from src.trading.slippage import SlippageEstimate, SlippageModel

Direction = Literal["LONG", "SHORT"]


class ExitReason(Enum):
    TARGET_HIT = "TARGET_HIT"
    STOP_HIT = "STOP_HIT"
    TIME_BASED = "TIME_BASED"   # 15:15 MIS auto-close
    MANUAL = "MANUAL"


@dataclass(frozen=True)
class FillResult:
    """Confirmed (simulated) fill for a single order leg."""

    symbol: str
    qty: int
    direction: Direction
    quoted_price: Decimal
    filled_price: Decimal
    timestamp: datetime
    is_paper: bool = True


@dataclass(frozen=True)
class ClosedTrade:
    """A completed round-trip trade including costs, slippage, and net P&L.

    net_pnl == trade_result.net_pnl is enforced in __post_init__.
    """

    entry_fill: FillResult
    exit_quoted_price: Decimal
    exit_filled_price: Decimal
    exit_reason: ExitReason
    trade_result: object          # TradeResult — avoid circular import hint
    slippage: SlippageEstimate
    net_pnl: Decimal

    def __post_init__(self) -> None:
        if self.net_pnl != self.trade_result.net_pnl:  # type: ignore[union-attr]
            raise ValueError(
                f"Invariant violation: ClosedTrade.net_pnl={self.net_pnl} "
                f"!= trade_result.net_pnl={self.trade_result.net_pnl}"  # type: ignore[union-attr]
            )


@dataclass
class OpenPosition:
    """A live intraday position."""

    symbol: str
    qty: int
    direction: Direction
    entry_fill: FillResult

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        qty_d = Decimal(self.qty)
        if self.direction == "LONG":
            return qty_d * (current_price - self.entry_fill.filled_price)
        return qty_d * (self.entry_fill.filled_price - current_price)


@dataclass(frozen=True)
class Portfolio:
    """Snapshot of portfolio state at a point in time."""

    cash: Decimal
    open_positions: dict[str, OpenPosition]
    realized_pnl: Decimal
    closed_trades: list[ClosedTrade]

    def total_realized_net_pnl(self) -> Decimal:
        return sum((t.net_pnl for t in self.closed_trades), Decimal("0"))


class ExecutionEngine(ABC):
    """Abstract base class for execution engines.

    Real-broker engines implement the same interface; callers depend only on this.
    """

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        qty: int,
        direction: Direction,
        price: Decimal,
        timestamp: datetime,
    ) -> FillResult:
        """Open a new position. Raises ValueError if symbol already has an open position."""
        ...

    @abstractmethod
    def close_position(
        self,
        symbol: str,
        exit_price: Decimal,
        exit_reason: ExitReason,
        timestamp: datetime,
    ) -> ClosedTrade:
        """Close the open position for symbol. Raises ValueError if no position exists."""
        ...

    @property
    @abstractmethod
    def portfolio(self) -> Portfolio:
        """Current portfolio snapshot."""
        ...

    @abstractmethod
    def has_position(self, symbol: str) -> bool: ...


class PaperExecutionEngine(ExecutionEngine):
    """Simulated execution engine using CostSimulator and SlippageModel."""

    def __init__(
        self,
        initial_cash: Decimal,
        cost_simulator: CostSimulator,
        slippage_model: SlippageModel,
    ) -> None:
        self._initial_cash = initial_cash
        self._cash = initial_cash
        self._cost_sim = cost_simulator
        self._slippage = slippage_model
        self._open: dict[str, OpenPosition] = {}
        self._closed: list[ClosedTrade] = []
        self._realized_pnl = Decimal("0")

    def place_order(
        self,
        symbol: str,
        qty: int,
        direction: Direction,
        price: Decimal,
        timestamp: datetime,
    ) -> FillResult:
        if symbol in self._open:
            raise ValueError(f"Position already open for {symbol!r}")

        filled_price = self._slippage.apply_entry(direction, price)
        fill = FillResult(
            symbol=symbol,
            qty=qty,
            direction=direction,
            quoted_price=price,
            filled_price=filled_price,
            timestamp=timestamp,
        )

        # Cash accounting on entry
        if direction == "LONG":
            self._cash -= Decimal(qty) * filled_price
        else:  # SHORT: selling opens a credit
            self._cash += Decimal(qty) * filled_price

        self._open[symbol] = OpenPosition(
            symbol=symbol, qty=qty, direction=direction, entry_fill=fill
        )
        return fill

    def close_position(
        self,
        symbol: str,
        exit_price: Decimal,
        exit_reason: ExitReason,
        timestamp: datetime,
    ) -> ClosedTrade:
        if symbol not in self._open:
            raise ValueError(f"No open position for {symbol!r}")

        pos = self._open.pop(symbol)
        entry_filled = pos.entry_fill.filled_price
        exit_filled = self._slippage.apply_exit(pos.direction, exit_price)

        # Cost calculation on slippage-adjusted prices
        trade_result = self._cost_sim.compute(
            symbol=symbol,
            qty=pos.qty,
            direction=pos.direction,
            entry_price=entry_filled,
            exit_price=exit_filled,
        )

        slippage_est = self._slippage.estimate(
            direction=pos.direction,
            qty=pos.qty,
            entry_quoted=pos.entry_fill.quoted_price,
            exit_quoted=exit_price,
        )

        # Cash accounting on exit
        if pos.direction == "LONG":
            self._cash += Decimal(pos.qty) * exit_filled - trade_result.costs.total
        else:  # SHORT: buying back closes the credit; costs come out too
            self._cash -= Decimal(pos.qty) * exit_filled + trade_result.costs.total

        self._realized_pnl += trade_result.net_pnl

        closed = ClosedTrade(
            entry_fill=pos.entry_fill,
            exit_quoted_price=exit_price,
            exit_filled_price=exit_filled,
            exit_reason=exit_reason,
            trade_result=trade_result,
            slippage=slippage_est,
            net_pnl=trade_result.net_pnl,
        )
        self._closed.append(closed)
        return closed

    @property
    def portfolio(self) -> Portfolio:
        return Portfolio(
            cash=self._cash,
            open_positions=dict(self._open),
            realized_pnl=self._realized_pnl,
            closed_trades=list(self._closed),
        )

    def has_position(self, symbol: str) -> bool:
        return symbol in self._open
