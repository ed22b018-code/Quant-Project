"""LightGBM multi-class scenario classifier: BULLISH / NO_TRADE / BEARISH.

NO_TRADE is a first-class label. The model is explicitly trained to identify
flat/noisy windows and rewarded for abstaining, rather than being forced to
pick a direction every bar.

Internal label encoding for LightGBM (must be 0 … N-1):
    NO_TRADE ( 0) → class index 0
    BULLISH  ( 1) → class index 1
    BEARISH  (-1) → class index 2

Output columns: p_no_trade, p_bullish, p_bearish (always sum to 1.0).

LightGBM is imported lazily. Instantiating the class and inspecting metadata
work without lightgbm. fit() raises ImportError if lightgbm is missing.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.labels import Label

_LOG = logging.getLogger(__name__)

_LABEL_TO_IDX: dict[Label, int] = {
    Label.NO_TRADE: 0,
    Label.BULLISH: 1,
    Label.BEARISH: 2,
}
_PROB_COLS = ("p_no_trade", "p_bullish", "p_bearish")


class ScenarioClassifier:
    """LightGBM multi-class scenario classifier.

    Parameters
    ----------
    feature_names : list[str] | None
        Expected feature column names. None = infer from X at fit time.
    lgbm_params : dict | None
        Extra parameters merged into the LightGBM training dict.
        'num_boost_round' (default 100) may be set here.
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        lgbm_params: dict | None = None,
    ) -> None:
        self.feature_names = feature_names
        self.lgbm_params = lgbm_params or {}
        self._model: Any = None
        self._is_fitted = False
        self._meta: dict = {}

    # ── public ────────────────────────────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> "ScenarioClassifier":
        """Train the multi-class classifier.

        y must contain Label enum values or equivalent ints (0, 1, -1).
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
        y_raw = y.values if isinstance(y, pd.Series) else np.asarray(y)
        y_idx = np.array([_LABEL_TO_IDX[Label(int(v))] for v in y_raw])

        user_params = dict(self.lgbm_params)
        n_rounds = int(user_params.pop("num_boost_round", 100))
        params: dict[str, Any] = {
            "objective": "multiclass",
            "num_class": 3,
            "verbose": -1,
            "n_jobs": -1,
            **user_params,
        }
        dtrain = lgb.Dataset(X_arr, label=y_idx)
        self._model = lgb.train(params, dtrain, num_boost_round=n_rounds)

        self._is_fitted = True
        self._meta = {
            "feature_names": list(X.columns) if isinstance(X, pd.DataFrame) else [],
            "n_samples": int(len(y_raw)),
            "n_boost_rounds": n_rounds,
            "label_encoding": {str(k.name): v for k, v in _LABEL_TO_IDX.items()},
        }
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Return class probabilities.

        Returns DataFrame with columns p_no_trade, p_bullish, p_bearish.
        Each row sums to 1.0.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ScenarioClassifier is not fitted. Call fit() before predict_proba()."
            )
        X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        raw = self._model.predict(X_arr)  # shape (n, 3)
        return pd.DataFrame(raw, columns=list(_PROB_COLS))

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Save model + metadata to a directory."""
        if not self._is_fitted:
            raise RuntimeError("Nothing to save — model not fitted.")
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(out / "scenario_classifier.txt"))
        with open(out / "meta.json", "w") as f:
            json.dump(self._meta, f, indent=2)
        _LOG.info("ScenarioClassifier saved to %s", out)

    @classmethod
    def load(cls, path: str | Path) -> "ScenarioClassifier":
        """Load a saved ScenarioClassifier from directory."""
        try:
            import lightgbm as lgb  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "lightgbm is required to load saved models. pip install lightgbm"
            )
        p = Path(path)
        with open(p / "meta.json") as f:
            meta = json.load(f)
        instance = cls(feature_names=meta.get("feature_names"))
        instance._model = lgb.Booster(model_file=str(p / "scenario_classifier.txt"))
        instance._is_fitted = True
        instance._meta = meta
        return instance
