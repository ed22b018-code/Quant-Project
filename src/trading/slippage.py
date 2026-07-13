"""Slippage model for NSE intraday equity round-trip trades.

Slippage is modelled, not measured — SlippageEstimate.is_estimate is always True.
SlippageModel is an ABC; implementations can be swapped without changing callers.

Sign convention: fills always go against the trader.
  LONG entry:  fills HIGHER than quoted (buying at ask)
  LONG exit:   fills LOWER than quoted  (selling at bid)
  SHORT entry: fills LOWER than quoted  (selling at bid)
  SHORT exit:  fills HIGHER than quoted (buying at ask)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal

from src.trading._types import Direction


@dataclass(frozen=True)
class SlippageEstimate:
    """Estimated slippage for a round-trip trade.

    is_estimate is always True — __post_init__ enforces this as a hard invariant.
    Structural reminder that slippage is a model estimate, never a measured fill.
    """

    entry_quoted: Decimal
    entry_filled: Decimal
    exit_quoted: Decimal
    exit_filled: Decimal
    entry_cost: Decimal   # abs(entry_filled - entry_quoted) * qty
    exit_cost: Decimal    # abs(exit_filled - exit_quoted) * qty
    total_cost: Decimal   # entry_cost + exit_cost
    model_name: str
    is_estimate: bool = True

    def __post_init__(self) -> None:
        if not self.is_estimate:
            raise ValueError(
                "SlippageEstimate.is_estimate must always be True — "
                "slippage is modelled, not measured."
            )


class SlippageModel(ABC):
    """Abstract base for slippage models.

    Subclasses implement apply_entry and apply_exit.
    estimate() is provided by the base class and composes both.
    """

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    def apply_entry(self, direction: Direction, price: Decimal) -> Decimal:
        """Return expected fill price for the entry leg."""
        ...

    @abstractmethod
    def apply_exit(self, direction: Direction, price: Decimal) -> Decimal:
        """Return expected fill price for the exit leg."""
        ...

    def estimate(
        self,
        direction: Direction,
        qty: int,
        entry_quoted: Decimal,
        exit_quoted: Decimal,
    ) -> SlippageEstimate:
        """Estimate round-trip slippage from quoted prices (calls the model for fills)."""
        entry_filled = self.apply_entry(direction, entry_quoted)
        exit_filled = self.apply_exit(direction, exit_quoted)
        return self.estimate_from_fills(
            qty=qty,
            entry_quoted=entry_quoted,
            entry_filled=entry_filled,
            exit_quoted=exit_quoted,
            exit_filled=exit_filled,
        )

    def estimate_from_fills(
        self,
        qty: int,
        entry_quoted: Decimal,
        entry_filled: Decimal,
        exit_quoted: Decimal,
        exit_filled: Decimal,
    ) -> SlippageEstimate:
        """Build a SlippageEstimate from already-computed fill prices.

        Use this when fill prices have already been determined (e.g., via
        apply_entry / apply_exit) to avoid calling the model a second time.
        Guarantees the stored estimate is consistent with the fills actually
        used in cost computation, even for stateful or non-deterministic models.
        """
        qty_d = Decimal(qty)
        entry_cost = abs(entry_filled - entry_quoted) * qty_d
        exit_cost = abs(exit_filled - exit_quoted) * qty_d
        return SlippageEstimate(
            entry_quoted=entry_quoted,
            entry_filled=entry_filled,
            exit_quoted=exit_quoted,
            exit_filled=exit_filled,
            entry_cost=entry_cost,
            exit_cost=exit_cost,
            total_cost=entry_cost + exit_cost,
            model_name=self.model_name,
            is_estimate=True,
        )


class FixedBpsSlippage(SlippageModel):
    """Fixed basis-point slippage applied symmetrically at entry and exit.

    Fills always hurt the trader — never a windfall.
    """

    def __init__(self, entry_bps: float, exit_bps: float) -> None:
        self._entry_factor = Decimal(str(entry_bps)) / Decimal("10000")
        self._exit_factor = Decimal(str(exit_bps)) / Decimal("10000")

    @property
    def model_name(self) -> str:
        return "fixed_bps"

    def apply_entry(self, direction: Direction, price: Decimal) -> Decimal:
        if direction == "LONG":
            return price * (Decimal("1") + self._entry_factor)
        return price * (Decimal("1") - self._entry_factor)

    def apply_exit(self, direction: Direction, price: Decimal) -> Decimal:
        if direction == "LONG":
            return price * (Decimal("1") - self._exit_factor)
        return price * (Decimal("1") + self._exit_factor)
