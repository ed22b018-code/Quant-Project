"""DataHealthMonitor — tracks feed staleness and gates model confidence.

Design
──────
A feed is "healthy" if either:
  * its most-recently-seen timestamp is within stale_threshold_seconds of as_of, OR
  * the market is currently closed (stale data during off-hours is expected/irrelevant).

A feed that has never been seen is treated as healthy when the market is closed and
unhealthy when the market is open — absence of ticks during trading hours is a real
problem; absence during nights and weekends is normal.

confidence_multiplier() returns:
  1.0  — feed healthy (fresh, or market closed)
  0.0  — feed stale during market hours

Extension point: Phase 2 may replace the binary 0/1 with a graduated ramp
(e.g., linear decay from 1.0 at threshold to 0.0 at 2× threshold), controlled
by a multiplier_mode parameter.  The interface is intentionally 0/1 for Phase 1
so it is a hard gate rather than a soft nudge.

check_value():
  Validates a scalar value against its previous reading.
  First observation (prev_value=None or prev_value=0) → always ok (True).
  Large jump (|Δ|/|prev| > jump_threshold) → log WARNING and return False.
  Used for sanity-checking tick prices, OI, etc. for data quality.

CPU-only — no GPU library imports.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

_log = logging.getLogger(__name__)


class DataHealthMonitor:
    """Track feed freshness; gate model confidence when feeds are stale.

    Parameters
    ----------
    stale_threshold_seconds:
        Age (in seconds) beyond which a feed is considered stale during market hours.
    jump_threshold:
        Relative change (|Δ|/|prev|) above which check_value() flags a suspicious jump.
        Default 0.5 = 50 % change.
    """

    def __init__(
        self,
        stale_threshold_seconds: float = 60.0,
        jump_threshold: float = 0.5,
    ) -> None:
        self._threshold = timedelta(seconds=stale_threshold_seconds)
        self._jump_threshold = jump_threshold
        self._last_seen: dict[str, datetime] = {}

    # ── Feed tracking ─────────────────────────────────────────────────────────

    def update_feed(self, name: str, ts: datetime) -> None:
        """Record that feed `name` was last seen at `ts`."""
        self._last_seen[name] = ts

    # ── Health checks ─────────────────────────────────────────────────────────

    def is_feed_healthy(self, feed: str, as_of: datetime, calendar) -> bool:
        """Return True if `feed` is fresh enough (or market is closed).

        Parameters
        ----------
        feed:
            Feed name as passed to update_feed().
        as_of:
            Current reference time.  Must be tz-aware (same requirement as
            NSETradingCalendar.is_market_open).
        calendar:
            NSETradingCalendar instance (or any object with .is_market_open(dt)).

        Raises
        ------
        ValueError
            If as_of is tz-naive (consistent with is_market_open's requirement).
        """
        if as_of.tzinfo is None:
            raise ValueError(
                "DataHealthMonitor.is_feed_healthy: as_of must be tz-aware "
                "(consistent with NSETradingCalendar.is_market_open).  "
                f"Got naive datetime: {as_of!r}"
            )

        market_open = calendar.is_market_open(as_of)

        if feed not in self._last_seen:
            # Never seen — healthy only if market is closed
            return not market_open

        last = self._last_seen[feed]

        # Compute age, handling tz-consistency
        if as_of.tzinfo is not None and last.tzinfo is not None:
            age = as_of - last
        elif as_of.tzinfo is None and last.tzinfo is None:
            age = as_of - last
        else:
            raise ValueError(
                f"DataHealthMonitor: tz-mismatch between as_of ({as_of!r}) and "
                f"last-seen timestamp for feed {feed!r} ({last!r}).  "
                "Normalize timestamps to a common timezone."
            )

        if age <= self._threshold:
            return True

        # Stale — healthy only if market is closed
        return not market_open

    def confidence_multiplier(self, feed: str, as_of: datetime, calendar) -> float:
        """Return 0.0 if feed is stale during market hours, 1.0 otherwise.

        Extension point: Phase 2 may introduce a graduated multiplier
        (e.g., linear decay from 1.0 at threshold to 0.0 at 2× threshold)
        based on staleness age and market regime.  The Phase 1 binary gate is
        intentionally conservative — a questionable feed contributes nothing
        rather than a fractional signal.
        """
        return 1.0 if self.is_feed_healthy(feed, as_of, calendar) else 0.0

    # ── Value sanity check ────────────────────────────────────────────────────

    def check_value(
        self,
        feed: str,
        value: float,
        prev_value: float | None = None,
    ) -> bool:
        """Check if `value` is within the expected range relative to `prev_value`.

        Returns True (ok / unknown) when:
          * prev_value is None — first observation, nothing to compare against.
          * prev_value is 0 — divide-by-zero case; comparison undefined.
          * Change is within jump_threshold.

        Returns False and emits a WARNING when:
          * |value - prev_value| / |prev_value| > jump_threshold.

        The warning logs both values and the % change so the caller can triage
        without needing to re-run.
        """
        if prev_value is None or prev_value == 0:
            return True

        change = abs(value - prev_value) / abs(prev_value)
        if change > self._jump_threshold:
            _log.warning(
                "DataHealthMonitor: suspicious value jump in feed %r: "
                "prev=%.6g → new=%.6g (%.0f%% change, threshold=%.0f%%).",
                feed, prev_value, value, change * 100, self._jump_threshold * 100,
            )
            return False
        return True
