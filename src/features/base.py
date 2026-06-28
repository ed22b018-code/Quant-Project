"""Feature registry — central catalog of all features.

Design
------
- Each feature is a plain function: (symbol, as_of, db_path) -> float | None.
- Features are registered with @FeatureRegistry.register(name, gate, description).
- The DataGate enum marks which data source a feature depends on, letting the
  dataset builder skip gated features cleanly and record which gates were open.
- FeatureMeta is immutable; the 'dead' flag is set by SHAP pruning (Checkpoint 2.6).

PIT contract
------------
Every feature function MUST query its underlying table with strict timestamp
comparisons (< as_of, not <=). This module does not enforce it mechanically —
the SQL in each feature function is the enforcement. Tests in test_features.py
verify it per feature.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable


class DataGate(str, Enum):
    """Which data source a feature depends on.

    Use in ExperimentTracker.log_run(gates_open=..., gates_closed=...)
    to stamp what was and wasn't available for a given training run.
    """
    NONE = "none"          # Always available: daily OHLCV (NSE bhavcopy)
    INTRADAY = "intraday"  # ohlcv_intraday  — Dhan 5-yr 1/5/15/25/60-min bars
    FO = "fo"              # fo_daily_metrics — Dhan F&O / OI history
    NEWS = "news"          # news table       — RSS ingest
    NEWS_NLP = "news_nlp"  # FinBERT scoring  — Checkpoint 2.2


@dataclass(frozen=True)
class FeatureMeta:
    """Immutable descriptor attached to a registered feature."""
    name: str
    gate: DataGate
    description: str = ""
    dead: bool = False  # set True by SHAP pruning in Checkpoint 2.6


class FeatureRegistry:
    """Central catalog of all registered features.

    Register:
        @FeatureRegistry.register("return_1d", DataGate.NONE, "1-day return")
        def return_1d(symbol, as_of, db_path): ...

    Compute:
        results = FeatureRegistry.compute_all(symbol, as_of, db_path)
        results = FeatureRegistry.compute_all(symbol, as_of, db_path,
                                               gates_open={DataGate.NONE, DataGate.FO})

    Thread safety: register() is called at import time (single-threaded);
    compute_all() reads the registry without modifying it.
    """

    _registry: dict[str, tuple[FeatureMeta, Callable]] = {}

    @classmethod
    def register(cls, name: str, gate: DataGate, description: str = "") -> Callable:
        """Decorator factory. The decorated function is returned unchanged."""
        def decorator(fn: Callable) -> Callable:
            cls._registry[name] = (FeatureMeta(name=name, gate=gate, description=description), fn)
            return fn
        return decorator

    @classmethod
    def compute_all(
        cls,
        symbol: str,
        as_of: datetime,
        db_path: str,
        *,
        gates_open: set[DataGate] | None = None,
    ) -> dict[str, float | None]:
        """Compute every registered feature for (symbol, as_of).

        Features whose gate is not in gates_open, or that are marked dead,
        silently return None — never raise. Exceptions inside a feature
        function are caught and logged as None (never propagate).

        Parameters
        ----------
        gates_open : set[DataGate] or None
            None means all gates open (compute everything).
        """
        if gates_open is None:
            gates_open = set(DataGate)
        results: dict[str, float | None] = {}
        for name, (meta, fn) in cls._registry.items():
            if meta.dead:
                results[name] = None
                continue
            if meta.gate != DataGate.NONE and meta.gate not in gates_open:
                results[name] = None
                continue
            try:
                results[name] = fn(symbol, as_of, db_path)
            except Exception:
                results[name] = None
        return results

    @classmethod
    def list_features(cls) -> list[FeatureMeta]:
        """All registered FeatureMeta objects."""
        return [meta for meta, _ in cls._registry.values()]

    @classmethod
    def get_meta(cls, name: str) -> FeatureMeta | None:
        entry = cls._registry.get(name)
        return entry[0] if entry else None
