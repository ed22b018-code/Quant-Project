"""DatasetBuilder and WalkForwardSplitter for the training pipeline.

DatasetBuilder
--------------
Assembles PIT (point-in-time) training rows by calling FeatureRegistry for
every (symbol, t0) pair. Labels come from TripleBarrierLabeler and are joined
on t0. Provenance columns record which gates were open at build time.

WalkForwardSplitter
-------------------
Time-ordered walk-forward folds with a sealed final test set.

The final test set (last final_test_fraction of data) is split off first and
NEVER included in any (train, val) fold. It is accessed exactly once via
get_final_test_set(), which raises RuntimeError on a second call to prevent
accidental re-use in parameter search or calibration decisions.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from src.features import DataGate, FeatureRegistry


class WalkForwardSplitter:
    """Time-ordered walk-forward splits with a sealed final test set.

    Parameters
    ----------
    n_splits : int
        Number of (train, val) folds to generate from the development set.
    final_test_fraction : float
        Fraction of the dataset (tail, by time) reserved as the final test set.
        Excluded from all folds. Accessed exactly once via get_final_test_set().
    t0_col : str
        Column name for the observation timestamp.
    """

    def __init__(
        self,
        n_splits: int = 5,
        final_test_fraction: float = 0.10,
        t0_col: str = "t0",
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be >= 1")
        if not 0 < final_test_fraction < 1:
            raise ValueError("final_test_fraction must be in (0, 1)")
        self.n_splits = n_splits
        self.final_test_fraction = final_test_fraction
        self.t0_col = t0_col
        self._test_accessed = False

    # ── internal ──────────────────────────────────────────────────────────────

    def _dev_test_split(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split into development set and sealed final test set."""
        sorted_df = df.sort_values(self.t0_col).reset_index(drop=True)
        test_size = max(1, int(len(sorted_df) * self.final_test_fraction))
        dev_df = sorted_df.iloc[:-test_size]
        test_df = sorted_df.iloc[-test_size:]
        return dev_df, test_df

    # ── public ────────────────────────────────────────────────────────────────

    def generate_splits(
        self, df: pd.DataFrame
    ) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
        """Return a list of (train_df, val_df) pairs in chronological order.

        The sealed final test set is EXCLUDED from every split.
        Within each fold, all train rows precede all val rows in time.
        """
        dev_df, _ = self._dev_test_split(df)
        n = len(dev_df)
        if n == 0:
            return []

        fold_size = max(1, n // (self.n_splits + 1))
        splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
        for k in range(self.n_splits):
            train_end = (k + 1) * fold_size
            val_end = (k + 2) * fold_size
            if train_end >= n:
                break
            train_fold = dev_df.iloc[:train_end]
            val_fold = dev_df.iloc[train_end:val_end]
            if len(train_fold) > 0 and len(val_fold) > 0:
                splits.append((train_fold.copy(), val_fold.copy()))
        return splits

    def get_final_test_set(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the sealed final test set. Raises RuntimeError on second call.

        The final test set must be evaluated exactly once, after all model
        selection and calibration decisions are frozen. Evaluating it again
        would constitute implicit hyperparameter tuning on the test set.
        """
        if self._test_accessed:
            raise RuntimeError(
                "Final test set has already been accessed. "
                "It must be touched exactly once, after all model and "
                "calibration decisions are frozen. A second access invalidates "
                "the evaluation and is not permitted."
            )
        self._test_accessed = True
        _, test_df = self._dev_test_split(df)
        return test_df.copy()


class DatasetBuilder:
    """Assemble PIT training rows: features at t0 joined with triple-barrier labels.

    All features are computed strictly before t0 via FeatureRegistry.compute_all().
    Labels from TripleBarrierLabeler look forward; purge_and_embargo() must be
    applied to each (train, val) split before fitting (see labels.py).

    Provenance columns in the output DataFrame record:
        gate_intraday_open, gate_fo_open, gate_news_open, gate_news_nlp_open
        forward_only_features — comma-separated names of forward-only features

    Parameters
    ----------
    feature_names : list[str] | None
        Subset of registered feature names to include. None = all registered.
    gates_open : set[DataGate] | None
        Which gates are open. None = all gates open.
    """

    def __init__(
        self,
        feature_names: Optional[list[str]] = None,
        gates_open: Optional[set] = None,
    ) -> None:
        self.feature_names = feature_names
        self.gates_open = gates_open

    def build(
        self,
        symbol: str,
        timestamps: list[datetime],
        db_path: str,
        labels_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build a PIT dataset for one symbol.

        Parameters
        ----------
        symbol     : NSE ticker.
        timestamps : Prediction timestamps. Features computed strictly before each.
        db_path    : Path to DuckDB database.
        labels_df  : Optional DataFrame from TripleBarrierLabeler with columns
                     (t0, t1, label, barrier). Joined on t0. Rows without a
                     matching label get NaN in the label columns.

        Returns
        -------
        DataFrame with columns: symbol, t0, <features>, [label, t1, barrier],
        gate_intraday_open, gate_fo_open, gate_news_open, gate_news_nlp_open,
        forward_only_features.

        Each feature is computed at its t0 — no future data included.
        """
        import logging
        _BUILD_LOG = logging.getLogger(__name__)

        rows = []
        n_ts = len(timestamps)
        for i, t0 in enumerate(timestamps):
            if i > 0 and i % max(1, n_ts // 10) == 0:
                pct = (i / n_ts) * 100
                _BUILD_LOG.info(f"[{symbol}] Building PIT features... {i}/{n_ts} timestamps ({pct:.1f}%)")
            
            feat = FeatureRegistry.compute_all(
                symbol, t0, db_path, gates_open=self.gates_open
            )
            if self.feature_names is not None:
                feat = {k: v for k, v in feat.items() if k in self.feature_names}
            rows.append({"symbol": symbol, "t0": t0, **feat})
            
        _BUILD_LOG.info(f"[{symbol}] Completed building {n_ts} timestamps.")

        df = pd.DataFrame(rows)

        if labels_df is not None and not labels_df.empty:
            label_cols = [
                c for c in ("t0", "t1", "label", "barrier") if c in labels_df.columns
            ]
            df = df.merge(labels_df[label_cols], on="t0", how="left")

        gates = self.gates_open if self.gates_open is not None else set(DataGate)
        df["gate_intraday_open"] = DataGate.INTRADAY in gates
        df["gate_fo_open"] = DataGate.FO in gates
        df["gate_news_open"] = DataGate.NEWS in gates
        df["gate_news_nlp_open"] = DataGate.NEWS_NLP in gates
        df["forward_only_features"] = (
            "news_sentiment" if DataGate.NEWS_NLP in gates else ""
        )

        return df
