"""Triple-barrier labeling and temporal validation utilities.

Triple-barrier method (López de Prado, AFML ch. 3)
----------------------------------------------------
For each observation (t0, entry_price) three barriers are set:
    upper:    entry_price * (1 + profit_pct)   — profit target
    lower:    entry_price * (1 - stop_pct)     — stop-loss
    vertical: t0 + horizon_bars forward bars   — time limit

Label = which barrier the price path touches first:
    BULLISH  ( 1) — upper barrier hit first
    BEARISH  (-1) — lower barrier hit first
    NO_TRADE ( 0) — vertical barrier hit first (flat / noisy window)

NO_TRADE is a first-class label, not a derived catch-all. Most short
windows are flat noise; the model is explicitly rewarded for staying out.

Purging + embargo (López de Prado, AFML ch. 7)
-----------------------------------------------
Labels span a forward window [t0, t1]. A train/val split leaks if any
training sample's t1 >= val_start. purge_and_embargo() removes:
    Purge:   training samples where t1 >= val_start
             (label window overlaps the validation period).
    Embargo: training samples where t0 >= val_start - embargo_bars * bar_timedelta
             (extra gap for residual feature autocorrelation).

Call purge_and_embargo on each (train_fold, val_fold) pair BEFORE fitting.
"""
from __future__ import annotations

from datetime import timedelta
from enum import IntEnum
from typing import Optional

import pandas as pd


class Label(IntEnum):
    """Triple-barrier label — first-class for all three outcomes."""

    NO_TRADE = 0
    BULLISH = 1
    BEARISH = -1

    @property
    def label_str(self) -> str:
        return {0: "no_trade", 1: "bullish", -1: "bearish"}[int(self)]


class TripleBarrierLabeler:
    """Assign triple-barrier labels to a price time-series.

    Parameters
    ----------
    profit_pct : float
        Upper barrier as a fraction above entry price (e.g. 0.01 = +1 %).
    stop_pct : float
        Lower barrier as a fraction below entry price (e.g. 0.005 = -0.5 %).
    horizon_bars : int
        Maximum forward bars to look. Time barrier fires if neither price
        barrier is reached, and the label is NO_TRADE.
    """

    def __init__(
        self,
        profit_pct: float,
        stop_pct: float,
        horizon_bars: int,
    ) -> None:
        if profit_pct <= 0:
            raise ValueError(f"profit_pct must be positive, got {profit_pct}")
        if stop_pct <= 0:
            raise ValueError(f"stop_pct must be positive, got {stop_pct}")
        if horizon_bars <= 0:
            raise ValueError(f"horizon_bars must be positive, got {horizon_bars}")
        self.profit_pct = profit_pct
        self.stop_pct = stop_pct
        self.horizon_bars = horizon_bars

    def label_series(
        self,
        prices: pd.Series,
        timestamps: pd.Series,
    ) -> pd.DataFrame:
        """Label each (timestamp, price) observation.

        Parameters
        ----------
        prices : Series of entry prices (typically close prices), length N.
        timestamps : Series of prediction timestamps, same length, sorted ascending.

        Returns
        -------
        DataFrame with N rows and columns:
            t0          : prediction timestamp
            t1          : barrier-hit timestamp (last forward bar for NO_TRADE)
            label       : int Label value (0 = NO_TRADE, 1 = BULLISH, -1 = BEARISH)
            barrier     : 'profit' | 'stop' | 'time'
            entry_price : close price at t0
        """
        prices = prices.reset_index(drop=True)
        timestamps = timestamps.reset_index(drop=True)
        prices_arr = prices.to_numpy(dtype=float)
        timestamps_arr = timestamps.to_numpy()
        n = len(prices_arr)

        records = []
        for i in range(n):
            p0 = prices_arr[i]
            t0 = timestamps_arr[i]
            upper = p0 * (1.0 + self.profit_pct)
            lower = p0 * (1.0 - self.stop_pct)

            end_idx = min(i + self.horizon_bars, n - 1)
            label: Label = Label.NO_TRADE
            t1 = timestamps_arr[end_idx]
            barrier = "time"

            for j in range(i + 1, end_idx + 1):
                p_j = prices_arr[j]
                if p_j >= upper:
                    label = Label.BULLISH
                    t1 = timestamps_arr[j]
                    barrier = "profit"
                    break
                if p_j <= lower:
                    label = Label.BEARISH
                    t1 = timestamps_arr[j]
                    barrier = "stop"
                    break

            records.append(
                {
                    "t0": t0,
                    "t1": t1,
                    "label": int(label),
                    "barrier": barrier,
                    "entry_price": p0,
                }
            )

        return pd.DataFrame(records)


def purge_and_embargo(
    train_labels: pd.DataFrame,
    val_labels: pd.DataFrame,
    *,
    t0_col: str = "t0",
    t1_col: str = "t1",
    embargo_bars: int = 0,
    bar_timedelta: Optional[timedelta] = None,
) -> pd.DataFrame:
    """Remove training samples whose label window leaks into the validation period.

    Purge rule (López de Prado AFML ch. 7)
    ---------------------------------------
    Drop any training sample where t1 >= val_start. That sample's label was
    determined using prices observed in the validation period, which leaks
    future information into the training signal.

    Embargo rule
    ------------
    Additionally drop training samples where
        t0 >= val_start - embargo_bars * bar_timedelta.
    The extra gap absorbs residual autocorrelation in features.

    Parameters
    ----------
    train_labels : DataFrame with at least t0_col and t1_col columns.
    val_labels   : DataFrame with at least t0_col column (val_start = min(t0)).
    embargo_bars : Extra bars of gap beyond the purge boundary (0 = purge only).
    bar_timedelta: Duration of one bar. Required when embargo_bars > 0.

    Returns
    -------
    Filtered copy of train_labels. Input is never modified.

    Raises
    ------
    ValueError if embargo_bars > 0 but bar_timedelta is None.
    """
    if embargo_bars > 0 and bar_timedelta is None:
        raise ValueError("bar_timedelta is required when embargo_bars > 0")

    if val_labels.empty or train_labels.empty:
        return train_labels.copy()

    val_start = pd.Timestamp(val_labels[t0_col].min())

    # Purge: label window must end strictly before the validation start
    t1_series = pd.to_datetime(train_labels[t1_col])
    keep_mask = t1_series < val_start

    # Embargo: t0 must also precede the embargo cutoff
    if embargo_bars > 0:
        embargo_cutoff = val_start - embargo_bars * bar_timedelta
        t0_series = pd.to_datetime(train_labels[t0_col])
        keep_mask = keep_mask & (t0_series < embargo_cutoff)

    return train_labels[keep_mask].copy()
