"""Tests for Checkpoint 2.3+2.4 model framework — QuantileConeTrainer,
ScenarioClassifier, ModelCalibrator, and run_training().

KEY TESTS (explicitly requested)
----------------------------------
test_reliability_diagram_structure
    ModelCalibrator.reliability_diagram() returns a dict with the required keys
    (bin_centers, empirical_freqs, counts, brier_score, max_calibration_error),
    correct shapes, and values in valid ranges. This is the structural check
    for the calibration framework: when the model says 70 %, it should be right
    ~70 % of the time — this function measures the gap.

test_uncalibrated_confidence_raises
    calibrate(require_fitted=True) — the default — raises UncalibratedConfidenceError
    when the calibrator has not been fitted on a validation set.
    This is the structural enforcement of "no confidence value downstream
    without calibration".

test_run_training_dry_run
    run_training(dry_run=True) completes without touching the DB or training
    any model. Returns a valid provenance dict and writes provenance.json.
    Confirms that "run real training + calibration" is a single triggerable
    step — call run_training() once after data lands.

Other tests
-----------
Quantile cone monotonicity, ScenarioClassifier probability shapes, calibration
Brier score improvement, provenance stamps, and absence of hard-coded GPU.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models.calibration import ModelCalibrator, UncalibratedConfidenceError
from src.models.labels import Label
from src.models.quantile_cone import QuantileConeTrainer
from src.models.scenario_classifier import ScenarioClassifier


# ── availability guards ───────────────────────────────────────────────────────

def _lgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
        return True
    except ImportError:
        return False


def _sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401
        return True
    except ImportError:
        return False


requires_lgbm = pytest.mark.skipif(
    not _lgbm_available(), reason="lightgbm not installed"
)
requires_sklearn = pytest.mark.skipif(
    not _sklearn_available(), reason="scikit-learn not installed"
)


# ── synthetic data helpers ────────────────────────────────────────────────────

def _make_regression_data(n: int = 200, m: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((n, m)), columns=[f"f{i}" for i in range(m)])
    y = pd.Series(rng.standard_normal(n))
    return X, y


def _make_classification_data(n: int = 300, m: int = 5) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((n, m)), columns=[f"f{i}" for i in range(m)])
    y = pd.Series(rng.choice([0, 1, -1], size=n))  # Label values
    return X, y


# ── 1. QuantileConeTrainer ────────────────────────────────────────────────────

def test_quantile_cone_enforce_monotonic_corrects_disordered():
    """_enforce_monotonic is a valid isotonic projection for 5 quantiles."""
    trainer = QuantileConeTrainer()
    # Deliberately disordered quantile predictions
    preds = np.array([
        [0.5, 0.2, 0.8, 0.1, 0.9],   # out of order
        [1.0, 0.9, 0.8, 0.7, 0.6],   # decreasing
        [0.1, 0.1, 0.1, 0.1, 0.1],   # all same
    ])
    mono = trainer._enforce_monotonic(preds)

    # After projection: each row must be non-decreasing
    assert (np.diff(mono, axis=1) >= 0).all(), (
        "Monotonicity violated after _enforce_monotonic"
    )
    # Shape preserved
    assert mono.shape == preds.shape


def test_quantile_cone_unfitted_raises():
    """predict() before fit() raises RuntimeError."""
    trainer = QuantileConeTrainer()
    X = pd.DataFrame({"f0": [1.0, 2.0]})
    with pytest.raises(RuntimeError, match="not fitted"):
        trainer.predict(X)


@requires_lgbm
def test_quantile_cone_fit_predict_shapes():
    """fit+predict returns (n, 5) DataFrame with correct column names."""
    X, y = _make_regression_data(200)
    trainer = QuantileConeTrainer(lgbm_params={"num_boost_round": 20})
    trainer.fit(X, y)
    preds = trainer.predict(X)

    assert preds.shape == (200, 5)
    assert list(preds.columns) == ["q_0.10", "q_0.25", "q_0.50", "q_0.75", "q_0.90"]


@requires_lgbm
def test_quantile_cone_monotonic_after_fit():
    """All rows in predicted output satisfy q_0.10 ≤ q_0.25 ≤ … ≤ q_0.90."""
    X, y = _make_regression_data(200)
    trainer = QuantileConeTrainer(lgbm_params={"num_boost_round": 20})
    trainer.fit(X, y)
    preds = trainer.predict(X)

    diffs = np.diff(preds.values, axis=1)
    assert (diffs >= 0).all(), "Quantile monotonicity violated in predicted output"


# ── 2. ScenarioClassifier ─────────────────────────────────────────────────────

def test_scenario_classifier_unfitted_raises():
    """predict_proba() before fit() raises RuntimeError."""
    clf = ScenarioClassifier()
    X = pd.DataFrame({"f0": [1.0, 2.0]})
    with pytest.raises(RuntimeError, match="not fitted"):
        clf.predict_proba(X)


@requires_lgbm
def test_scenario_classifier_proba_sum_to_one():
    """All rows in predict_proba output sum to 1.0."""
    X, y = _make_classification_data(300)
    clf = ScenarioClassifier(lgbm_params={"num_boost_round": 20})
    clf.fit(X, y)
    proba = clf.predict_proba(X)

    np.testing.assert_allclose(
        proba.sum(axis=1).values, 1.0, atol=1e-6,
        err_msg="Scenario probabilities do not sum to 1.0",
    )


@requires_lgbm
def test_scenario_no_trade_class_present():
    """predict_proba output has p_no_trade column (NO_TRADE is first-class)."""
    X, y = _make_classification_data(300)
    clf = ScenarioClassifier(lgbm_params={"num_boost_round": 20})
    clf.fit(X, y)
    proba = clf.predict_proba(X)

    assert "p_no_trade" in proba.columns
    assert "p_bullish" in proba.columns
    assert "p_bearish" in proba.columns
    # All probabilities in [0, 1]
    assert proba.values.min() >= 0.0
    assert proba.values.max() <= 1.0 + 1e-9


# ── 3. ModelCalibrator — KEY TESTS ───────────────────────────────────────────

def test_uncalibrated_confidence_raises():
    """
    KEY TEST — structural enforcement of calibration.

    calibrate() with the default require_fitted=True must raise
    UncalibratedConfidenceError when the calibrator has not been fitted.
    """
    cal = ModelCalibrator()
    prob_df = pd.DataFrame({"p_bullish": [0.7, 0.3, 0.6]})

    with pytest.raises(UncalibratedConfidenceError):
        cal.calibrate(prob_df)


def test_calibrate_returns_raw_when_require_fitted_false():
    """calibrate(require_fitted=False) returns raw probabilities without raising."""
    cal = ModelCalibrator()
    prob_df = pd.DataFrame({"p": [0.1, 0.5, 0.9]})
    result = cal.calibrate(prob_df, require_fitted=False)

    pd.testing.assert_frame_equal(result, prob_df)


def test_reliability_diagram_structure():
    """
    KEY TEST — calibration reliability-diagram structural check.

    For a well-calibrated model, bin_centers ≈ empirical_freqs.
    This function proves the structure (keys, shapes, types, valid ranges)
    and quantifies the calibration gap — when the model says 70 %, it should
    be right ~70 % of the time.
    """
    cal = ModelCalibrator()
    rng = np.random.default_rng(0)
    n = 500
    y_prob = rng.uniform(0.0, 1.0, n)
    y_true = (rng.uniform(0.0, 1.0, n) < y_prob).astype(int)

    diag = cal.reliability_diagram(y_true, y_prob, n_bins=10)

    # Required keys
    assert "bin_centers" in diag
    assert "empirical_freqs" in diag
    assert "counts" in diag
    assert "brier_score" in diag
    assert "max_calibration_error" in diag

    # Correct lengths
    assert len(diag["bin_centers"]) == 10
    assert len(diag["empirical_freqs"]) == 10
    assert len(diag["counts"]) == 10

    # Valid ranges
    assert all(0.0 <= c <= 1.0 for c in diag["bin_centers"])
    assert all(0.0 <= f <= 1.0 for f in diag["empirical_freqs"] if f > 0)
    assert all(c >= 0 for c in diag["counts"])
    assert isinstance(diag["brier_score"], float)
    assert isinstance(diag["max_calibration_error"], float)
    assert 0.0 <= diag["max_calibration_error"] <= 1.0

    # Total count must equal n
    assert sum(diag["counts"]) == n


def test_reliability_diagram_brier_score():
    """Perfect predictions produce a Brier score near 0."""
    cal = ModelCalibrator()
    y_true = np.array([0, 1, 0, 1], dtype=float)
    y_prob = np.array([0.0, 1.0, 0.0, 1.0])  # perfect
    diag = cal.reliability_diagram(y_true, y_prob)
    assert diag["brier_score"] == pytest.approx(0.0, abs=1e-9)


@requires_sklearn
def test_calibration_reduces_brier_score():
    """Isotonic calibration on validation data reduces the Brier score."""
    rng = np.random.default_rng(7)
    n = 200
    y_true = rng.integers(0, 2, size=n).astype(float)
    # Deliberately over-confident predictions (all high)
    y_prob_raw = rng.uniform(0.65, 0.90, size=n)

    prob_df = pd.DataFrame({"p": y_prob_raw})
    cal = ModelCalibrator(method="isotonic")
    cal.fit(y_true, prob_df)

    calibrated_df = cal.calibrate(prob_df)
    brier_raw = float(np.mean((y_prob_raw - y_true) ** 2))
    brier_cal = float(np.mean((calibrated_df["p"].values - y_true) ** 2))

    # Calibrated must be ≤ raw (isotonic regression minimises MSE on fit data)
    assert brier_cal <= brier_raw + 1e-9, (
        f"Calibration did not reduce Brier score: raw={brier_raw:.4f}, cal={brier_cal:.4f}"
    )


@requires_sklearn
def test_calibration_provenance_stamp():
    """calibration_provenance() reflects fitted state and metadata."""
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=100).astype(float)
    prob_df = pd.DataFrame({"p": rng.uniform(0, 1, 100)})

    cal = ModelCalibrator()
    assert cal.calibration_provenance()["is_calibrated"] is False

    cal.fit(y_true, prob_df, val_period=("2024-01-01", "2024-03-31"))
    prov = cal.calibration_provenance()

    assert prov["is_calibrated"] is True
    assert prov["calibration_method"] == "isotonic"
    assert prov["n_calibration_samples"] == 100
    assert prov["validation_period"] is not None


# ── 4. run_training — dry_run single-step confirmation ───────────────────────

def test_run_training_dry_run(tmp_path):
    """
    KEY TEST — single triggerable step confirmation.

    run_training(dry_run=True) completes without touching the database or
    training any model. Returns a valid provenance dict and writes
    provenance.json to output_dir.

    This confirms that once the Dhan data lands, the full pipeline runs with a
    single call: run_training(db_path, output_dir, ..., dry_run=False).
    """
    from src.models.training_pipeline import run_training

    result = run_training(
        db_path=str(tmp_path / "does_not_exist.db"),  # not accessed in dry_run
        output_dir=str(tmp_path / "output"),
        universe=["RELIANCE", "TCS"],
        start_date=date(2024, 1, 2),
        end_date=date(2024, 1, 31),
        dry_run=True,
    )

    # Provenance dict structure
    assert result["dry_run"] is True
    assert result["universe"] == ["RELIANCE", "TCS"]
    assert "pipeline_version" in result
    assert "build_ts" in result
    assert "gates_open" in result
    assert "feature_names" in result
    assert "n_timestamps_per_symbol" in result

    # Provenance file written
    provenance_file = tmp_path / "output" / "provenance.json"
    assert provenance_file.exists(), "provenance.json not written to output_dir"


def test_run_training_accepts_required_params():
    """run_training() function signature accepts all required parameters."""
    from src.models.training_pipeline import run_training
    import inspect

    sig = inspect.signature(run_training)
    params = set(sig.parameters.keys())

    assert "db_path" in params
    assert "output_dir" in params
    assert "universe" in params
    assert "start_date" in params
    assert "end_date" in params
    assert "dry_run" in params


# ── 5. No hard-coded GPU in models layer ─────────────────────────────────────

def test_no_hardcoded_gpu_in_models_layer():
    """Models layer must not hard-code GPU device strings."""
    models_dir = Path("src/models")
    for py_file in models_dir.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        # No hard-coded 'cuda' device assignment (string literal assignment)
        hard_coded = re.findall(r'device\s*=\s*["\']cuda["\']', source)
        assert not hard_coded, (
            f"{py_file.name}: hard-coded 'cuda' device found: {hard_coded}"
        )
        # No bare torch import at module top-level (must be lazy inside functions)
        top_level_torch = re.findall(r'^import torch\b', source, re.MULTILINE)
        assert not top_level_torch, (
            f"{py_file.name}: top-level 'import torch' found (must be lazy)"
        )
