"""Isotonic / Platt calibration for model confidence values.

Design rule (structural, not just convention)
----------------------------------------------
No confidence value may be propagated downstream without calibration.
Enforced by: calibrate(require_fitted=True)  ← the default.
If the calibrator has not been fitted on a validation set, calibrate() raises
UncalibratedConfidenceError. Pass require_fitted=False only for diagnostics.

Fit on a VALIDATION slice only:
    - Never on training data (that leaks; calibration would be over-fitted).
    - Never on the final test set (that contaminates the held-out evaluation).

Reliability diagram
-------------------
reliability_diagram(y_true, y_prob, n_bins=10) → dict

When the model outputs P = x, a perfectly calibrated model is correct ~x * 100%
of the time. The reliability diagram measures the gap between predicted
probability and empirical frequency across n_bins equal-width probability bins.

Keys:
    bin_centers         — midpoint of each bin (list of n_bins floats)
    empirical_freqs     — actual positive rate per bin (list of n_bins floats)
    counts              — samples per bin (list of n_bins ints)
    brier_score         — mean squared error; lower is better
    max_calibration_error — max |predicted - empirical| across non-empty bins
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

_LOG = logging.getLogger(__name__)


class UncalibratedConfidenceError(RuntimeError):
    """Raised when an uncalibrated confidence value is used downstream.

    Fit ModelCalibrator on a validation set before propagating any confidence
    to position sizing, signal generation, or experiment tracking.
    """


class ModelCalibrator:
    """Isotonic regression or Platt scaling for probability calibration.

    One calibrator is fit per probability column (e.g. one per class for
    multi-class outputs). For multi-class, each column is calibrated
    independently; the caller is responsible for re-normalising if needed.

    Parameters
    ----------
    method : 'isotonic' | 'platt'
        'isotonic' — IsotonicRegression (sklearn). Non-parametric; tends to
                     over-fit with very few validation samples.
        'platt'    — LogisticRegression on the raw probability (Platt scaling).
                     Parametric; better generalises when validation set is small.
    """

    def __init__(self, method: str = "isotonic") -> None:
        if method not in ("isotonic", "platt"):
            raise ValueError(f"method must be 'isotonic' or 'platt', got {method!r}")
        self.method = method
        self._calibrators: dict[str, object] = {}
        self._is_fitted = False
        self._fit_metadata: dict = {}

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        y_true: np.ndarray,
        y_prob_df: pd.DataFrame,
        *,
        val_period: Optional[tuple] = None,
    ) -> "ModelCalibrator":
        """Fit one calibrator per probability column.

        IMPORTANT: pass only validation data. Never training data (leakage),
        never the final test set (contaminates evaluation).

        Parameters
        ----------
        y_true    : 1-D array of ground-truth labels.
        y_prob_df : DataFrame; each column holds predicted probabilities for one
                    class or output (values in [0, 1]).
        val_period: Optional (start, end) of the validation window. Stored in
                    provenance; does not affect fitting.

        Raises ImportError if scikit-learn is not installed.
        """
        try:
            from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415
            from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "scikit-learn is required for calibration. "
                "Uncomment it in requirements.txt and run: pip install scikit-learn"
            )

        y_arr = np.asarray(y_true, dtype=float)
        for col in y_prob_df.columns:
            probs = y_prob_df[col].values.astype(float)
            if self.method == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip")
                cal.fit(probs, y_arr)
            else:  # platt
                cal = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
                cal.fit(probs.reshape(-1, 1), y_arr)
            self._calibrators[col] = cal

        self._is_fitted = True
        self._fit_metadata = {
            "method": self.method,
            "val_period": (
                [str(p) for p in val_period] if val_period is not None else None
            ),
            "n_samples": int(len(y_arr)),
            "columns": list(y_prob_df.columns),
        }
        return self

    # ── calibrate ─────────────────────────────────────────────────────────────

    def calibrate(
        self,
        y_prob_df: pd.DataFrame,
        *,
        require_fitted: bool = True,
    ) -> pd.DataFrame:
        """Apply calibration to a probability DataFrame.

        Parameters
        ----------
        y_prob_df     : DataFrame of raw probabilities (same columns as fit).
        require_fitted: Default True. Raises UncalibratedConfidenceError when the
                        calibrator has not been fitted. Set False only for
                        diagnostics where you deliberately want raw probabilities.

        Returns
        -------
        DataFrame of calibrated probabilities, same shape and column names.

        Raises
        ------
        UncalibratedConfidenceError if require_fitted=True and not fitted.
        """
        if not self._is_fitted:
            if require_fitted:
                raise UncalibratedConfidenceError(
                    "ModelCalibrator has not been fitted. Fit on a held-out validation "
                    "set before calibrating. No confidence value may be propagated "
                    "downstream — to position sizing, signal generation, or experiment "
                    "tracking — without calibration."
                )
            return y_prob_df.copy()

        result: dict[str, np.ndarray] = {}
        for col in y_prob_df.columns:
            if col not in self._calibrators:
                _LOG.warning("Column %r not in fitted calibrators — returning raw.", col)
                result[col] = y_prob_df[col].values.copy()
                continue
            cal = self._calibrators[col]
            probs = y_prob_df[col].values.astype(float)
            if self.method == "isotonic":
                result[col] = cal.predict(probs)
            else:  # platt
                result[col] = cal.predict_proba(probs.reshape(-1, 1))[:, 1]

        return pd.DataFrame(result, index=y_prob_df.index)

    # ── diagnostics ───────────────────────────────────────────────────────────

    def reliability_diagram(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> dict:
        """Compute reliability diagram data for a single probability column.

        When the model says P = x, a well-calibrated model should be correct
        approximately x * 100 % of the time. This function quantifies that gap.

        Parameters
        ----------
        y_true : Binary labels (0 or 1) for the class of interest.
        y_prob : Predicted probabilities in [0, 1] for the class of interest.
        n_bins : Number of equal-width bins spanning [0, 1].

        Returns
        -------
        dict with keys:
            bin_centers          — midpoint of each probability bin (n_bins floats)
            empirical_freqs      — actual positive rate per bin (n_bins floats)
            counts               — sample count per bin (n_bins ints)
            brier_score          — mean squared error between prob and outcome
            max_calibration_error— max |predicted - empirical| over non-empty bins
        """
        y_t = np.asarray(y_true, dtype=float)
        y_p = np.asarray(y_prob, dtype=float)

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        empirical_freqs = np.zeros(n_bins)
        counts = np.zeros(n_bins, dtype=int)

        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            # Include right edge in last bin to catch prob == 1.0
            mask = (y_p >= lo) & (y_p < hi) if i < n_bins - 1 else (y_p >= lo) & (y_p <= hi)
            counts[i] = int(mask.sum())
            if counts[i] > 0:
                empirical_freqs[i] = float(y_t[mask].mean())

        brier_score = float(np.mean((y_p - y_t) ** 2))
        nonempty = counts > 0
        max_cal_error = (
            float(np.max(np.abs(bin_centers[nonempty] - empirical_freqs[nonempty])))
            if nonempty.any()
            else 0.0
        )

        return {
            "bin_centers": bin_centers.tolist(),
            "empirical_freqs": empirical_freqs.tolist(),
            "counts": counts.tolist(),
            "brier_score": brier_score,
            "max_calibration_error": max_cal_error,
        }

    # ── provenance ────────────────────────────────────────────────────────────

    def calibration_provenance(self) -> dict:
        """Return a provenance dict to attach to any confidence value this calibrator produces."""
        return {
            "is_calibrated": self._is_fitted,
            "calibration_method": self._fit_metadata.get("method"),
            "validation_period": self._fit_metadata.get("val_period"),
            "n_calibration_samples": self._fit_metadata.get("n_samples"),
        }

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
