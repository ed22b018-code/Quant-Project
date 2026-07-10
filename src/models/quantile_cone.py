"""LightGBM quantile-regression cone for 15 / 30 / 60-minute price moves.

Five quantile regressors (q = 0.10 / 0.25 / 0.50 / 0.75 / 0.90) trained on
the same feature set. After prediction, row-wise monotonicity is enforced so
that q_0.10 ≤ q_0.25 ≤ … ≤ q_0.90 for every row.

LightGBM is imported lazily inside fit() / load() so this module can be
imported even when lightgbm is not installed. Instantiating the class, calling
_enforce_monotonic(), and inspecting metadata all work without lightgbm.
fit() raises ImportError with a clear pip install hint if lightgbm is missing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)

QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)


class QuantileConeTrainer:
    """LightGBM quantile regression cone.

    Parameters
    ----------
    horizon_minutes : int
        Look-ahead horizon (15, 30, or 60). Stored in metadata only — the
        caller supplies the target series; this class does not resample.
    feature_names : list[str] | None
        Expected feature column names. None = infer from X at fit time.
    lgbm_params : dict | None
        Extra parameters merged into the LightGBM training dict.
        'num_boost_round' (default 100) may be set here.
    """

    def __init__(
        self,
        horizon_minutes: int = 30,
        feature_names: list[str] | None = None,
        lgbm_params: dict | None = None,
    ) -> None:
        self.horizon_minutes = horizon_minutes
        self.feature_names = feature_names
        self.lgbm_params = lgbm_params or {}
        self._models: dict[float, Any] = {}
        self._is_fitted = False
        self._meta: dict = {}

    # ── public ────────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "QuantileConeTrainer":
        """Train one LightGBM quantile regressor per quantile level.

        Raises ImportError if lightgbm is not installed.
        """
        try:
            import lightgbm as lgb  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "lightgbm is required for training. "
                "Uncomment it in requirements.txt and run: pip install lightgbm"
            )

        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = y.values if isinstance(y, pd.Series) else np.asarray(y)

        base_params: dict[str, Any] = {
            "objective": "quantile",
            "verbose": -1,
        }
        user_params = dict(self.lgbm_params)
        n_rounds = int(user_params.pop("num_boost_round", 100))

        for q in QUANTILES:
            params = {**base_params, "alpha": q, **user_params}
            dtrain = lgb.Dataset(X_arr, label=y_arr)
            self._models[q] = lgb.train(params, dtrain, num_boost_round=n_rounds)
            _LOG.debug("Trained q=%.2f quantile model (%d rounds)", q, n_rounds)

        self._is_fitted = True
        self._meta = {
            "horizon_minutes": self.horizon_minutes,
            "feature_names": list(X.columns) if isinstance(X, pd.DataFrame) else [],
            "n_samples": int(len(y_arr)),
            "n_boost_rounds": n_rounds,
        }
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Predict quantile values with monotonicity enforced.

        Returns DataFrame with columns q_0.10, q_0.25, q_0.50, q_0.75, q_0.90.
        Monotonicity guarantee: q_0.10 ≤ q_0.25 ≤ … ≤ q_0.90 for every row.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "QuantileConeTrainer is not fitted. Call fit() before predict()."
            )
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        raw = np.column_stack([self._models[q].predict(X_arr) for q in QUANTILES])
        mono = self._enforce_monotonic(raw)
        return pd.DataFrame(mono, columns=[f"q_{q:.2f}" for q in QUANTILES])

    def _enforce_monotonic(self, preds: np.ndarray) -> np.ndarray:
        """Ensure each row's quantile predictions are non-decreasing.

        np.sort(axis=1) is an exact isotonic projection for 5 values:
        the sorted vector minimises squared distance to the original while
        satisfying the monotonicity constraint.
        """
        return np.sort(preds, axis=1)

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save all quantile models + metadata to a directory."""
        if not self._is_fitted:
            raise RuntimeError("Nothing to save — model not fitted.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        for q, model in self._models.items():
            model.save_model(str(out / f"q_{q:.2f}.txt"))
        with open(out / "meta.json", "w") as f:
            json.dump(self._meta, f, indent=2)
        _LOG.info("QuantileConeTrainer saved to %s", out)

    @classmethod
    def load(cls, path: str | Path) -> "QuantileConeTrainer":
        """Load a saved QuantileConeTrainer from directory."""
        try:
            import lightgbm as lgb  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "lightgbm is required to load saved models. pip install lightgbm"
            )
        p = Path(path)
        with open(p / "meta.json") as f:
            meta = json.load(f)
        instance = cls(
            horizon_minutes=meta.get("horizon_minutes", 30),
            feature_names=meta.get("feature_names"),
        )
        for q in QUANTILES:
            model_file = p / f"q_{q:.2f}.txt"
            if model_file.exists():
                instance._models[q] = lgb.Booster(model_file=str(model_file))
        instance._is_fitted = len(instance._models) == len(QUANTILES)
        instance._meta = meta
        return instance
