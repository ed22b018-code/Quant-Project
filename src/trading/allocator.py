"""Capital allocator — sizes shortlisted candidates and enforces hard guardrails.

SEPARATION OF CONCERNS: the scanner (5.1) decides WHICH candidates to trade.
The allocator decides HOW MUCH capital each candidate gets. Sizing logic must NOT
leak back into scanner ranking — they are independent concerns.

SIZING FORMULA (fractional Kelly + volatility scaling):

    kelly_f  = after_cost_ev_pct / stop_loss_pct
               Kelly fraction: edge per unit risked if wrong. All values after cost.

    raw_pct  = kelly_fraction × kelly_f
               × min(vol_target_pct / estimated_volatility_pct, max_vol_scalar)

    Kelly fraction is FRACTIONAL (default ¼) — see AllocatorConfig docstring for why.
    Volatility scaling: position grows when realised vol is below target (more room
    per unit of risk budget), shrinks when vol is high.

HARD GUARDRAILS applied in order after sizing (non-overridable):

    1. Individual cap   — position_value ≤ total_capital × max_position_pct
    2. Sector cap       — sector total ≤ total_capital × max_sector_exposure_pct
    3. Exposure / cash  — min( max_intraday_exposure_pct,
                               1 − min_cash_reserve_pct ) is the binding ceiling;
                          whichever guardrail is tighter is applied and named.

All three guardrails that FIRE are listed in SizedOrder.guardrails_applied.
Nothing is silently swallowed: every breach is logged at WARNING level.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal

from src.config import AllocatorConfig
from src.trading._types import Direction
from src.trading.scanner import ScanResult

_LOG = logging.getLogger(__name__)


# ── Input / output dataclasses ─────────────────────────────────────────────────

@dataclass(frozen=True)
class AllocationInput:
    """Wraps a ScanResult with the volatility and sector data needed for sizing.

    The real pipeline derives estimated_volatility_pct from recent ATR or realised
    vol. For testing, mock values are used — the interface is defined here so the
    real model plugs in without changing the allocator.
    """

    scan_result: ScanResult
    estimated_volatility_pct: float  # recent daily vol as % of price (e.g. ATR%)
    sector: str                       # NSE sector string — for concentration guardrail


@dataclass(frozen=True)
class SizedOrder:
    """A guardrail-checked, sized order ready to hand to PaperExecutionEngine.

    guardrails_applied: tuple of human-readable strings naming each guardrail that
    fired and capped this order. Empty tuple means raw size was within all limits.
    raw_position_pct: what Kelly+vol scaling wanted BEFORE any guardrail cap.
    position_pct:     what was actually allocated AFTER caps.
    """

    symbol: str
    direction: Direction
    qty: int
    entry_price: Decimal
    position_value: Decimal       # actual capital deployed (qty × entry_price)
    position_pct: float           # final allocation as fraction of total capital
    raw_position_pct: float       # pre-guardrail allocation fraction
    guardrails_applied: tuple[str, ...]  # empty if no cap fired


@dataclass(frozen=True)
class RejectedCandidate:
    """A shortlisted candidate that received zero allocation after guardrails."""

    symbol: str
    reason: str


@dataclass
class AllocationResult:
    """Full output of one allocator run."""

    orders: list[SizedOrder]
    rejected: list[RejectedCandidate]
    total_exposure_pct: float         # total capital deployed as fraction
    sector_exposure: dict[str, float] # sector → fraction of total capital


# ── Allocator ─────────────────────────────────────────────────────────────────

class CapitalAllocator:
    """Sizes shortlisted candidates using fractional Kelly + vol scaling,
    then enforces hard guardrails in a fixed priority order.

    Constructor accepts individual limit parameters rather than the full TradingConfig
    so it can be constructed directly in tests. The production trading loop extracts
    the values from TradingConfig and passes them here.
    """

    def __init__(
        self,
        allocator_config: AllocatorConfig,
        max_position_pct: float,
        max_intraday_exposure_pct: float,
        min_cash_reserve_pct: float,
        max_sector_exposure_pct: float,
        total_capital: Decimal,
    ) -> None:
        self._alloc = allocator_config
        self._max_pos = max_position_pct
        self._max_intraday = max_intraday_exposure_pct
        self._min_cash = min_cash_reserve_pct
        self._max_sector = max_sector_exposure_pct
        self._total = total_capital

    def allocate(
        self,
        inputs: list[AllocationInput],
        current_exposure_pct: float = 0.0,
        current_sector_exposure: dict[str, float] | None = None,
        drawdown_paused: bool = False,
    ) -> AllocationResult:
        """Size and guardrail-check each AllocationInput; return AllocationResult.

        inputs:                  AllocationInputs in any order; processed by rank.
        current_exposure_pct:    fraction of capital already in open positions.
        current_sector_exposure: pre-existing sector allocations as fractions.
        drawdown_paused:         if True, return zero orders (portfolio down >threshold).
        """
        if drawdown_paused:
            return AllocationResult(
                orders=[],
                rejected=[
                    RejectedCandidate(
                        symbol=inp.scan_result.prediction.symbol,
                        reason="drawdown_pause: new entries halted (portfolio down from peak)",
                    )
                    for inp in inputs
                ],
                total_exposure_pct=current_exposure_pct,
                sector_exposure=dict(current_sector_exposure or {}),
            )

        if not inputs:
            return AllocationResult(
                orders=[],
                rejected=[],
                total_exposure_pct=current_exposure_pct,
                sector_exposure=dict(current_sector_exposure or {}),
            )

        # Process best-EV candidates first (ascending rank = best first)
        sorted_inputs = sorted(inputs, key=lambda x: x.scan_result.rank)

        orders: list[SizedOrder] = []
        rejected: list[RejectedCandidate] = []

        running_exposure = Decimal(str(current_exposure_pct)) * self._total
        sector_values: dict[str, Decimal] = {
            k: Decimal(str(v)) * self._total
            for k, v in (current_sector_exposure or {}).items()
        }

        for inp in sorted_inputs:
            result = self._process_candidate(inp, running_exposure, sector_values)
            if isinstance(result, RejectedCandidate):
                rejected.append(result)
                _LOG.warning("ALLOCATOR REJECTED %s: %s", result.symbol, result.reason)
            else:
                orders.append(result)
                running_exposure += result.position_value
                sector_values[inp.sector] = (
                    sector_values.get(inp.sector, Decimal("0")) + result.position_value
                )
                if result.guardrails_applied:
                    _LOG.warning(
                        "GUARDRAIL fired for %s — %s  raw=%.2f%% → final=%.2f%%",
                        result.symbol,
                        ", ".join(result.guardrails_applied),
                        result.raw_position_pct * 100,
                        result.position_pct * 100,
                    )

        return AllocationResult(
            orders=orders,
            rejected=rejected,
            total_exposure_pct=float(running_exposure / self._total),
            sector_exposure={k: float(v / self._total) for k, v in sector_values.items()},
        )

    # ── Internal methods ──────────────────────────────────────────────────────

    def _process_candidate(
        self,
        inp: AllocationInput,
        running_exposure: Decimal,
        sector_values: dict[str, Decimal],
    ) -> SizedOrder | RejectedCandidate:
        pred = inp.scan_result.prediction
        symbol = pred.symbol
        entry_price = pred.entry_price

        direction = pred.direction
        if direction == "NO_TRADE":
            return RejectedCandidate(
                symbol=symbol,
                reason="direction=NO_TRADE (scanner should have filtered this)",
            )

        raw_pct = self._compute_raw_pct(inp)
        final_pct, guardrails = self._apply_guardrails(
            raw_pct=raw_pct,
            sector=inp.sector,
            running_exposure=running_exposure,
            sector_values=sector_values,
        )

        if final_pct <= 0.0:
            return RejectedCandidate(
                symbol=symbol,
                reason=f"zero allocation after guardrails (raw={raw_pct:.4%})",
            )

        position_value = Decimal(str(final_pct)) * self._total
        qty = int(position_value / entry_price)  # floor — never buy fractional shares

        if qty <= 0:
            return RejectedCandidate(
                symbol=symbol,
                reason=(
                    f"qty=0 after integer rounding: "
                    f"position_value={float(position_value):.2f} < entry_price={float(entry_price):.2f}"
                ),
            )

        actual_value = Decimal(qty) * entry_price
        actual_pct = float(actual_value / self._total)

        return SizedOrder(
            symbol=symbol,
            direction=direction,  # type: ignore[arg-type]  — NO_TRADE filtered above
            qty=qty,
            entry_price=entry_price,
            position_value=actual_value,
            position_pct=actual_pct,
            raw_position_pct=raw_pct,
            guardrails_applied=tuple(guardrails),
        )

    def _compute_raw_pct(self, inp: AllocationInput) -> float:
        """Pure fractional Kelly + vol scaling. No guardrail logic here."""
        ev = inp.scan_result.after_cost_ev_pct
        stop = inp.scan_result.prediction.stop_loss_pct
        vol = inp.estimated_volatility_pct

        if stop <= 0.0 or vol <= 0.0 or ev <= 0.0:
            return 0.0

        kelly_f = ev / stop
        fractional_f = self._alloc.kelly_fraction * kelly_f
        vol_scalar = min(
            self._alloc.vol_target_pct / vol,
            self._alloc.max_vol_scalar,
        )
        return max(0.0, fractional_f * vol_scalar)

    def _apply_guardrails(
        self,
        raw_pct: float,
        sector: str,
        running_exposure: Decimal,
        sector_values: dict[str, Decimal],
    ) -> tuple[float, list[str]]:
        """Cap raw_pct through each guardrail in priority order.

        Returns (final_pct, list_of_guardrail_names_that_fired).
        The order matters: individual → sector → exposure/cash (binding wins).

        All comparisons are done in Decimal to avoid float accumulation errors
        (e.g. 0.25 − float(Decimal("0.2")) = 0.04999... not 0.05000...).
        """
        applied: list[str] = []
        total = self._total

        # Convert once to Decimal — all guardrail thresholds kept exact
        max_pos = Decimal(str(self._max_pos))
        max_sector = Decimal(str(self._max_sector))
        max_intraday = Decimal(str(self._max_intraday))
        min_cash = Decimal(str(self._min_cash))
        pct = Decimal(str(raw_pct))

        # Guardrail 1 — individual position cap
        if pct > max_pos:
            applied.append(f"individual_cap:{self._max_pos:.0%}")
            pct = max_pos

        # Guardrail 2 — sector concentration cap
        sector_used = sector_values.get(sector, Decimal("0")) / total
        sector_available = max(Decimal("0"), max_sector - sector_used)
        if pct > sector_available:
            applied.append(f"sector_cap:{self._max_sector:.0%}:{sector}")
            pct = sector_available

        # Guardrails 3 + 4 — total intraday exposure AND cash reserve
        # Both limits constrain how much total capital can be deployed; the
        # binding (smaller) one is applied and named.
        used = running_exposure / total
        intraday_remaining = max(Decimal("0"), max_intraday - used)
        cash_remaining = max(Decimal("0"), Decimal("1") - min_cash - used)
        deployable_remaining = min(intraday_remaining, cash_remaining)

        if pct > deployable_remaining:
            # Name the binding guardrail (the one with less room)
            if intraday_remaining <= cash_remaining:
                applied.append(f"exposure_cap:{self._max_intraday:.0%}")
            else:
                applied.append(f"cash_reserve:{self._min_cash:.0%}")
            pct = deployable_remaining

        return float(pct), applied
