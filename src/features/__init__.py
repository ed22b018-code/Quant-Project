"""Feature engineering for the intraday prediction system.

Public API:
    DataGate        — enum marking which data source a feature requires
    FeatureMeta     — immutable descriptor for a registered feature
    FeatureRegistry — central feature catalog; register + compute_all
    compute_features — convenience wrapper

Available features by gate
--------------------------
Gate NONE   (always available — daily OHLCV from NSE bhavcopy):
    return_1d, return_5d, vol_20d, volume_ratio_20d, atr_20d

Gate INTRADAY   (requires ohlcv_intraday — Dhan bulk download):
    vwap_deviation, intraday_momentum, opening_range_position

Gate FO   (requires fo_daily_metrics — Dhan bulk download):
    pcr_oi, pcr_volume, net_oi_change, max_pain_distance, spot_futures_basis

Gate NEWS   (requires news table — RSS ingest):
    news_count_1d, news_count_5d

Gate NEWS_NLP   (requires FinBERT scorer — Checkpoint 2.2):
    news_sentiment  [registered in src/features/sentiment.py]

Data-availability flags
-----------------------
INTRADAY and FO features are built and unit-tested against synthetic data now.
Full validation against real minute-bar / F&O data runs after the Dhan bulk
download completes.  Every ExperimentTracker run must record which gates were
open at training time (see DESIGN.md §2 rule 5 and §13 anti-leakage table).
"""
from __future__ import annotations

from src.features.base import DataGate, FeatureMeta, FeatureRegistry

# Side-effect imports register all features into the registry.
import src.features.price_volume  # noqa: F401
import src.features.fo            # noqa: F401
import src.features.news_count    # noqa: F401
import src.features.sentiment     # noqa: F401

__all__ = ["DataGate", "FeatureMeta", "FeatureRegistry", "compute_features"]


def compute_features(
    symbol: str,
    as_of,
    db_path: str,
    *,
    gates_open=None,
) -> dict[str, float | None]:
    """Compute all registered features for a single (symbol, as_of) point.

    Parameters
    ----------
    symbol : str
        NSE ticker symbol.
    as_of : datetime
        Point-in-time reference. Only data timestamped strictly BEFORE as_of
        is used — guaranteed at the SQL level in each feature function.
    db_path : str
        Path to the DuckDB database file.
    gates_open : set of DataGate or None
        Which data gates are active. None means all gates open.
        Features whose gate is not in gates_open return None.

    Returns
    -------
    dict mapping feature name → float or None.
    None means: gate closed, data unavailable, or computation failed.
    """
    return FeatureRegistry.compute_all(symbol, as_of, db_path, gates_open=gates_open)
