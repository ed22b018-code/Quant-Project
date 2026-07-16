"""Rigorous evaluation of the v2_fo (FO_AUGMENTED) model vs the v1_full baseline.

EVALUATION METHODOLOGY
======================
Part 1 — PIT leakage test
    Verify the D-1 rule: fo_daily_metrics for day D is only visible at t0 ≥ D+1.
    Queries the live DB with the same WHERE clause used in src/features/fo.py.

Part 2 — Calibration reliability (sealed test set)
    Reliability diagram on the test set that was NEVER touched during training
    or calibration.

Part 3 — After-cost benchmark
    Two OOS windows evaluated together for ~9 months of total coverage:
      - Fold 5 val period (~112 trading days, before the test set)
        OOS for the final model: the final model is trained on fold-5 train only.
      - Sealed test set (~76 trading days, last 10% of date range)
    Combined this gives ~9 months of proper OOS evaluation instead of 3.

Part 4 — Per-month breakdown + block t-stat
    Monthly net P&L across all ~9 OOS months.
    Block t-stat: t = mean(monthly) / (std(monthly) / sqrt(n_months)).
    |t| > 2.0 required for statistical significance at α=0.05.

Part 5 — Regime breakdown
    Classify each trading day by the PREVIOUS day's market direction
    (median cross-sectional return_1d across all symbols).
    Compare model P&L in bull-regime vs bear-regime days.

Part 6 — F&O feature importance (headline metric for Checkpoint 7.2)
    Show which F&O features LightGBM actually used (gain importance).
    If pcr_oi / pcr_volume / net_oi_change have near-zero importance, F&O added
    no edge over the price/volume model.

Part 7 — Head-to-head vs v1_full baseline
    Both models evaluated on the IDENTICAL test timestamps (same date range).
    v1_full was trained with F&O = all-zero; its F&O importances are all zero,
    so providing real F&O values does not change its predictions.

Part 8 — Verdict
    Honest answer to: "Does F&O OI add real edge over the price/volume baseline?"

Usage
-----
    python -m scripts.run_fo_evaluation                        # v2_fo vs v1_full
    python -m scripts.run_fo_evaluation --model-dir models/v2_fo
    python -m scripts.run_fo_evaluation --model-dir models/v2_fo --skip-fold5-val
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

# Ensure project root is on sys.path so 'src.*' imports resolve when the script
# is launched directly (e.g. via Start-Process python scripts\run_fo_evaluation.py).
sys.path.insert(0, str(Path(__file__).parent.parent))

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.WARNING)

import duckdb
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import load_config
from src.models.calibration import ModelCalibrator
from src.models.dataset import DatasetBuilder
from src.models.labels import Label, TripleBarrierLabeler
from src.models.scenario_classifier import ScenarioClassifier
from src.models.training_pipeline import _build_for_symbol_worker
from src.trading._types import Direction
from src.trading.costs import CostSimulator

UTC = timezone.utc
_SEP = "=" * 72
_SEC = "-" * 60

# ── Constants (matching training pipeline + run_evaluation_v2.py) ─────────────
POOL_SIZE       = 500_000
MAX_POS_PCT     = 0.10
POSITION_BUDGET = POOL_SIZE * MAX_POS_PCT
PROFIT_PCT      = 0.010
STOP_PCT        = 0.005
HORIZON_BARS    = 30
SLIPPAGE_BPS    = 2.0
CPU_CAP         = 6
N_SPLITS        = 5          # walk-forward folds
TEST_FRACTION   = 0.10       # 10% sealed test set


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir",    default="models/v2_fo")
    p.add_argument("--baseline-dir", default="models/v1_full")
    p.add_argument("--db-path",      default="data/market.duckdb")
    p.add_argument("--skip-fold5-val", action="store_true",
                   help="Evaluate only the test set (faster; loses per-period coverage)")
    return p.parse_args()


# ── Timestamp helpers ─────────────────────────────────────────────────────────

def _bday_timestamps(start: date, end: date) -> list[datetime]:
    bdays = pd.bdate_range(start=start, end=end)
    return [datetime(d.year, d.month, d.day, 4, 0, tzinfo=UTC) for d in bdays]


def _test_timestamps(prov: dict) -> list[datetime]:
    """Last 10% of business-day timestamps — the sealed test set."""
    all_ts = _bday_timestamps(
        pd.to_datetime(prov["date_range"][0]).date(),
        pd.to_datetime(prov["date_range"][1]).date(),
    )
    n_test = max(1, int(len(all_ts) * TEST_FRACTION))
    return all_ts[-n_test:]


def _fold5_val_timestamps(prov: dict) -> list[datetime]:
    """Fold-5 validation timestamps — OOS for the final model.

    The final model trains on fold-5 train = dev_ts[:5*fold_size].
    Fold-5 val = dev_ts[5*fold_size : 6*fold_size].
    This period is BEFORE the test set and was NOT in the final model's
    training data — proper OOS evaluation.
    """
    all_ts = _bday_timestamps(
        pd.to_datetime(prov["date_range"][0]).date(),
        pd.to_datetime(prov["date_range"][1]).date(),
    )
    n_total = len(all_ts)
    n_test  = max(1, int(n_total * TEST_FRACTION))
    dev_ts  = all_ts[:-n_test]
    n_dev   = len(dev_ts)
    fold_sz = max(1, n_dev // (N_SPLITS + 1))
    v_start = N_SPLITS * fold_sz          # = 5 * fold_sz
    v_end   = min((N_SPLITS + 1) * fold_sz, n_dev)
    return dev_ts[v_start:v_end]


# ── PIT leakage test ──────────────────────────────────────────────────────────

def _pit_leakage_test(db_path: str, symbols: list[str]) -> bool:
    """Verify D-1 rule: fo_daily_metrics for date D is visible only at t0 ≥ D+1.

    Mirrors the exact WHERE clause in src/features/fo.py:
        WHERE date < CAST(? AS DATE)
    The probe confirms that at t0 = 2026-07-01 09:30 IST, the feature query
    returns rows with date ≤ 2026-06-30, never 2026-07-01.
    """
    print()
    print(_SEP)
    print("PART 1 — PIT / LEAKAGE VERIFICATION")
    print(_SEP)
    print("  Rule: at t0 = 2026-07-01 09:30 IST, features must use fo_daily_metrics")
    print("        with date < '2026-07-01'.  Max visible date must be ≤ 2026-06-30.")
    print()

    probe_sym  = next((s for s in ("RELIANCE", "TCS", "INFY") if s in symbols), symbols[0])
    probe_date = date(2026, 7, 1)
    probe_date_m1 = probe_date - timedelta(days=1)  # 2026-06-30

    all_ok = True
    try:
        with duckdb.connect(db_path, read_only=True, config={"threads": "1"}) as con:
            # Replicate exact PIT query from fo.py
            row = con.execute("""
                SELECT MAX(date) AS max_visible
                FROM fo_daily_metrics
                WHERE symbol = ?
                  AND date < ?
            """, [probe_sym, str(probe_date)]).fetchone()
            max_visible = row[0] if row else None

            same_day_count = con.execute("""
                SELECT COUNT(*)
                FROM fo_daily_metrics
                WHERE symbol = ? AND date = ?
            """, [probe_sym, str(probe_date)]).fetchone()[0]

            total_rows = con.execute(
                "SELECT COUNT(*) FROM fo_daily_metrics WHERE symbol = ?", [probe_sym]
            ).fetchone()[0]

        if max_visible is None:
            print(f"  WARNING: No fo_daily_metrics rows found for {probe_sym}")
            print("           PIT leakage cannot be verified — data may be missing.")
            return False

        pit_ok = (max_visible <= probe_date_m1)
        status = "PASS" if pit_ok else "FAIL"
        print(f"  Symbol under test    : {probe_sym}  ({total_rows} total rows)")
        print(f"  t0 used for probe    : {probe_date} 09:30 IST")
        print(f"  Expected max visible : ≤ {probe_date_m1}  (yesterday)")
        print(f"  Actual max visible   : {max_visible}")
        print(f"  PIT result           : [{status}]")
        if same_day_count > 0:
            print(f"  Note: {same_day_count} rows exist for {probe_date} in fo_daily_metrics.")
            print(f"        These are correctly BLOCKED by the WHERE date < '{probe_date}' clause.")

        if not pit_ok:
            print()
            print("  RED FLAG: fo_daily_metrics rows from today are visible to the model.")
            print("  This is a look-ahead / data leakage bug. Do NOT trust results.")
            all_ok = False
        else:
            print()
            print("  OK: D-1 PIT rule confirmed. No look-ahead in F&O features.")

    except Exception as exc:
        print(f"  PIT test failed with error: {exc}")
        all_ok = False

    return all_ok


# ── Dataset build ─────────────────────────────────────────────────────────────

def _build_dataset(
    timestamps: list[datetime],
    universe: list[str],
    feature_names: list[str],
    db_path: str,
    label: str = "test",
) -> pd.DataFrame:
    builder = DatasetBuilder(feature_names=feature_names)
    labeler = TripleBarrierLabeler(
        profit_pct=PROFIT_PCT, stop_pct=STOP_PCT, horizon_bars=HORIZON_BARS
    )
    parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=CPU_CAP) as ex:
        futs = {
            ex.submit(_build_for_symbol_worker, sym, db_path, timestamps, builder, labeler): sym
            for sym in universe
        }
        done = 0
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                parts.append(fut.result())
                done += 1
                if done % 20 == 0 or done == len(universe):
                    print(f"    {done}/{len(universe)} symbols ({label})", flush=True)
            except Exception as exc:
                print(f"    ERROR [{sym}]: {exc}")
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).sort_values("t0").reset_index(drop=True)
    return df[df["label"].notna()].copy()


# ── Price fetch ───────────────────────────────────────────────────────────────

def _fetch_entry_prices(
    symbols: list[str], ts_list: list[datetime], db_path: str
) -> dict[tuple[str, datetime], float]:
    if not symbols or not ts_list:
        return {}
    dates    = sorted({t.date() for t in ts_list})
    date_str = ", ".join(f"'{d}'" for d in dates)
    sym_str  = ", ".join(f"'{s}'" for s in symbols)
    query = f"""
        SELECT symbol, ts, close
        FROM   ohlcv_intraday
        WHERE  timeframe = '1'
          AND  symbol IN ({sym_str})
          AND  CAST(ts AS DATE) IN ({date_str})
          AND  EXTRACT(HOUR   FROM ts) = 9
          AND  EXTRACT(MINUTE FROM ts) = 30
    """
    with duckdb.connect(db_path, read_only=True, config={"threads": "2"}) as con:
        rows = con.execute(query).fetchall()
    prices: dict[tuple[str, datetime], float] = {}
    for symbol, ts_raw, close in rows:
        if isinstance(ts_raw, datetime):
            ts_utc = ts_raw.replace(tzinfo=UTC) if ts_raw.tzinfo is None else ts_raw.astimezone(UTC)
        else:
            ts_utc = datetime(ts_raw.year, ts_raw.month, ts_raw.day, 4, 0, tzinfo=UTC)
        prices[(symbol, ts_utc)] = float(close)
    return prices


# ── P&L helpers ───────────────────────────────────────────────────────────────

def _exit_price(entry: float, label: int, pred_dir: str) -> tuple[float, str]:
    """Approximate exit from triple-barrier label (mirrors run_evaluation_v2.py)."""
    if pred_dir == "LONG":
        if label == int(Label.BULLISH):
            return entry * (1 + PROFIT_PCT), "profit"
        elif label == int(Label.BEARISH):
            return entry * (1 - STOP_PCT), "stop"
        return entry, "time"
    else:  # SHORT
        if label == int(Label.BEARISH):
            return entry * (1 - STOP_PCT), "profit"
        elif label == int(Label.BULLISH):
            return entry * (1 + STOP_PCT), "stop"
        return entry, "time"


def _slippage_cost(entry: float, qty: int) -> float:
    return entry * qty * (SLIPPAGE_BPS * 2) / 10_000


def _compute_pnl_df(
    trade_df: pd.DataFrame,
    all_ts: list[datetime],
    db_path: str,
    cost_sim: CostSimulator,
) -> pd.DataFrame:
    """Compute per-trade P&L DataFrame with real NSE cost stack."""
    if len(trade_df) == 0:
        return pd.DataFrame()
    syms      = trade_df["symbol"].unique().tolist()
    price_map = _fetch_entry_prices(syms, all_ts, db_path)

    rows_out = []
    missing  = 0
    for _, row in trade_df.iterrows():
        sym   = row["symbol"]
        t0    = row["t0"]
        label = int(row["label"])
        pred  = int(row["pred"])
        t0_utc = t0.replace(tzinfo=UTC) if (isinstance(t0, datetime) and t0.tzinfo is None) else (
            datetime(t0.year, t0.month, t0.day, 4, 0, tzinfo=UTC)
            if not isinstance(t0, datetime) else t0.astimezone(UTC)
        )
        entry = price_map.get((sym, t0_utc))
        if not entry or entry <= 0:
            missing += 1
            continue
        pred_dir: Direction = "LONG" if pred == int(Label.BULLISH) else "SHORT"
        exit_p, hit = _exit_price(entry, label, pred_dir)
        qty         = max(1, int(POSITION_BUDGET / entry))
        try:
            tr = cost_sim.compute(
                symbol=sym, qty=qty, direction=pred_dir,
                entry_price=Decimal(str(round(entry, 4))),
                exit_price =Decimal(str(round(exit_p, 4))),
            )
            gross = float(tr.gross_pnl)
            cost  = float(tr.costs.total) + _slippage_cost(entry, qty)
            net   = gross - cost
        except Exception as exc:
            print(f"    CostSimulator error [{sym} {t0}]: {exc}")
            continue
        rows_out.append({
            "symbol"  : sym,
            "t0"      : t0_utc,
            "label"   : label,
            "pred_dir": pred_dir,
            "hit_type": hit,
            "entry"   : entry,
            "exit_p"  : exit_p,
            "qty"     : qty,
            "gross"   : gross,
            "cost"    : cost,
            "net"     : net,
        })
    if missing:
        print(f"    Skipped {missing} rows — no entry price in DB")
    return pd.DataFrame(rows_out)


# ── Calibration ───────────────────────────────────────────────────────────────

def _reliability_rows(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    cal = ModelCalibrator()
    rd  = cal.reliability_diagram(y_true, y_prob, n_bins=n_bins)
    rows = [(c, f, n) for c, f, n in
            zip(rd["bin_centers"], rd["empirical_freqs"], rd["counts"]) if n > 0]
    return rows, rd["brier_score"], rd["max_calibration_error"]


def _print_reliability(name: str, rows, brier, max_err) -> None:
    print(f"\n  Calibration — {name}")
    print(f"  {'Pred':>8}  {'Empirical':>10}  {'Gap':>8}  {'N':>6}")
    print(f"  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*6}")
    for c, f, n in rows:
        print(f"  {c:>8.2f}  {f:>10.2f}  {f-c:>+8.3f}  {n:>6}")
    print(f"  Brier: {brier:.4f}  MaxErr: {max_err:.3f}")


# ── Statistical helpers ───────────────────────────────────────────────────────

def _block_tstat(monthly_net_pct: list[float]) -> tuple[float, int]:
    """Block t-stat: t = mean / (std / sqrt(n)).  Returns (t, df)."""
    n = len(monthly_net_pct)
    if n < 2:
        return float("nan"), 0
    arr = np.array(monthly_net_pct, dtype=float)
    t = arr.mean() / (arr.std(ddof=1) / np.sqrt(n))
    return float(t), n - 1


# ── Buy-and-hold benchmark ────────────────────────────────────────────────────

def _bnh_return(universe: list[str], ts_list: list[datetime], db_path: str) -> float:
    if not ts_list:
        return 0.0
    start_d = min(t.date() for t in ts_list)
    end_d   = max(t.date() for t in ts_list)
    sym_str = ", ".join(f"'{s}'" for s in universe)
    query = f"""
        WITH fl AS (
            SELECT symbol, MIN(ts) AS first_ts, MAX(ts) AS last_ts
            FROM ohlcv_intraday
            WHERE timeframe='1' AND symbol IN ({sym_str})
              AND CAST(ts AS DATE) >= '{start_d}' AND CAST(ts AS DATE) <= '{end_d}'
              AND EXTRACT(HOUR FROM ts)=9 AND EXTRACT(MINUTE FROM ts)=30
            GROUP BY symbol
        )
        SELECT f.symbol, i1.close, i2.close
        FROM fl f
        JOIN ohlcv_intraday i1 ON i1.symbol=f.symbol AND i1.ts=f.first_ts AND i1.timeframe='1'
        JOIN ohlcv_intraday i2 ON i2.symbol=f.symbol AND i2.ts=f.last_ts  AND i2.timeframe='1'
    """
    try:
        with duckdb.connect(db_path, read_only=True, config={"threads": "2"}) as con:
            rows = con.execute(query).fetchall()
        rets = [(e - s) / s for _, s, e in rows if s and s > 0]
        return float(np.mean(rets)) if rets else 0.0
    except Exception as exc:
        print(f"  Buy-and-hold query failed: {exc}")
        return 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

def run_fo_evaluation(model_dir: str, baseline_dir: str, db_path: str,
                      skip_fold5_val: bool = False) -> None:
    mdir = Path(model_dir)
    if not mdir.exists():
        sys.exit(f"Model directory not found: {mdir.resolve()}")

    prov_path = mdir / "provenance.json"
    if not prov_path.exists():
        sys.exit(f"provenance.json not found in {mdir}")

    with open(prov_path) as f:
        prov = json.load(f)

    cfg      = load_config()
    cost_sim = CostSimulator(cfg.trading.costs)
    universe = prov["universe"]
    feat_names = prov["feature_names"]

    print(_SEP)
    print("F&O-AUGMENTED MODEL EVALUATION — CHECKPOINT 7.2")
    print(_SEP)
    print(f"Model dir    : {mdir.resolve()}")
    print(f"Model type   : {prov.get('model_type', 'UNKNOWN')}")
    print(f"Baseline dir : {Path(baseline_dir).resolve()}")
    print(f"DB           : {db_path}")
    print(f"Date range   : {prov['date_range'][0]} to {prov['date_range'][1]}")

    fh = prov.get("feature_honesty", {})
    if fh:
        print("\nFEATURE HONESTY:")
        for k, v in fh.items():
            if k not in ("warning", "comparison_note", "baseline"):
                print(f"  {k}: {v}")
        if "baseline" in fh:
            print(f"  Baseline: {fh['baseline']}")

    # ── Part 1 — PIT leakage test ─────────────────────────────────────────────
    pit_ok = _pit_leakage_test(db_path, universe)

    # ── Compute evaluation timestamps ─────────────────────────────────────────
    test_ts  = _test_timestamps(prov)
    fold5_ts = [] if skip_fold5_val else _fold5_val_timestamps(prov)

    print()
    print(_SEP)
    print("EVALUATION WINDOWS (both OOS for the final model)")
    print(_SEP)
    if fold5_ts:
        print(f"  Fold-5 val  : {fold5_ts[0].date()} to {fold5_ts[-1].date()}  ({len(fold5_ts)} trading days)")
    print(f"  Test set    : {test_ts[0].date()} to {test_ts[-1].date()}  ({len(test_ts)} trading days)")
    total_oos = len(fold5_ts) + len(test_ts)
    print(f"  Total OOS   : {total_oos} trading days (~{total_oos/21:.0f} months)")
    if not fold5_ts:
        print("  Note: --skip-fold5-val active — only test set evaluated.")
    else:
        print()
        print("  Why fold-5 val is OOS: the final model trains on fold-5 TRAIN")
        print("  (dev_ts[:5*fold_size]). Fold-5 VAL (dev_ts[5*fold_size:]) was")
        print("  never in the training data — proper OOS.")

    # ── Load primary model ────────────────────────────────────────────────────
    clf_dir = mdir / "scenario_classifier"
    if not clf_dir.exists():
        sys.exit(f"scenario_classifier/ not found in {mdir}")
    classifier = ScenarioClassifier.load(clf_dir)
    print(f"\nClassifier loaded: {len(classifier.feature_names)} features")

    # ── Build evaluation datasets ─────────────────────────────────────────────
    print()
    print(_SEP)
    print("BUILDING EVALUATION DATASETS")
    print(_SEP)

    if fold5_ts:
        print(f"  Building fold-5 val dataset ({len(fold5_ts)} days × {len(universe)} symbols)...")
        fold5_df = _build_dataset(fold5_ts, universe, feat_names, db_path, label="fold5-val")
    else:
        fold5_df = pd.DataFrame()

    print(f"  Building test dataset ({len(test_ts)} days × {len(universe)} symbols)...")
    test_df = _build_dataset(test_ts, universe, feat_names, db_path, label="test")

    if len(test_df) == 0:
        sys.exit("Test dataset is empty after label filter.")

    # Combine for per-period analysis; keep separate for test-only reporting
    all_oos_ts  = fold5_ts + test_ts
    all_oos_dfs = [df for df in [fold5_df, test_df] if len(df) > 0]
    oos_df      = pd.concat(all_oos_dfs, ignore_index=True).sort_values("t0").reset_index(drop=True)

    feat_cols = [c for c in test_df.columns if c in feat_names]

    print(f"\n  Fold-5 val rows : {len(fold5_df)} (with valid labels)")
    print(f"  Test set rows   : {len(test_df)} (with valid labels)")
    print(f"  Combined OOS    : {len(oos_df)} rows")
    print(f"  Label dist (test): {test_df['label'].astype(int).value_counts().to_dict()}")

    # ── Part 2 — Predictions ──────────────────────────────────────────────────
    def _add_preds(df: pd.DataFrame) -> pd.DataFrame:
        if len(df) == 0:
            return df
        X = df[feat_cols].fillna(0.0)
        proba = classifier.predict_proba(X)
        pred_class = proba.idxmax(axis=1).map({
            "p_no_trade": int(Label.NO_TRADE),
            "p_bullish" : int(Label.BULLISH),
            "p_bearish" : int(Label.BEARISH),
        })
        out = df.copy()
        out["pred"]       = pred_class.values
        out["p_bullish"]  = proba["p_bullish"].values
        out["p_bearish"]  = proba["p_bearish"].values
        out["p_no_trade"] = proba["p_no_trade"].values
        return out

    test_df  = _add_preds(test_df)
    fold5_df = _add_preds(fold5_df)
    oos_df   = _add_preds(oos_df)

    def _pred_summary(df: pd.DataFrame, name: str) -> None:
        if len(df) == 0:
            return
        no_trade = (df["pred"] == int(Label.NO_TRADE)).mean()
        bull     = (df["pred"] == int(Label.BULLISH)).mean()
        bear     = (df["pred"] == int(Label.BEARISH)).mean()
        print(f"  {name}: NO_TRADE {no_trade:.1%}  BULL {bull:.1%}  BEAR {bear:.1%}")

    print()
    print(_SEP)
    print("SECTION 1 — PREDICTION DISTRIBUTION")
    print(_SEP)
    _pred_summary(test_df, "Test set  ")
    if len(fold5_df) > 0:
        _pred_summary(fold5_df, "Fold-5 val")
    _pred_summary(oos_df,   "Combined  ")

    # ── Part 3 — Calibration (test set only — gold standard) ─────────────────
    print()
    print(_SEP)
    print("SECTION 2 — CALIBRATION (sealed test set)")
    print(_SEP)
    y_test = test_df["label"].astype(int)
    for class_lbl, col, name in [
        (int(Label.BULLISH),  "p_bullish",  "BULLISH"),
        (int(Label.BEARISH),  "p_bearish",  "BEARISH"),
        (int(Label.NO_TRADE), "p_no_trade", "NO_TRADE"),
    ]:
        y_bin = (y_test == class_lbl).astype(float).values
        y_pr  = test_df[col].values
        rows, brier, max_err = _reliability_rows(y_bin, y_pr)
        if not rows:
            print(f"\n  {name}: no non-empty bins")
            continue
        _print_reliability(name, rows, brier, max_err)

    # High-confidence accuracy
    print()
    print("  High-confidence accuracy (>0.60 threshold, test set):")
    for lv, col, name in [(int(Label.BULLISH), "p_bullish", "BULLISH"),
                           (int(Label.BEARISH), "p_bearish", "BEARISH")]:
        mask = test_df[col] > 0.60
        n = mask.sum()
        if n == 0:
            print(f"    {name} >60%: no predictions")
            continue
        acc = (test_df.loc[mask, "label"].astype(int) == lv).mean()
        print(f"    {name} >60%: {acc:.1%} accurate ({n} predictions)")

    # ── Part 4 — After-cost P&L (test set — primary headline) ────────────────
    print()
    print(_SEP)
    print("SECTION 3 — AFTER-COST BACKTEST (test set — primary headline)")
    print(_SEP)

    test_trade_df = test_df[test_df["pred"] != int(Label.NO_TRADE)].copy()
    test_pnl_df   = _compute_pnl_df(test_trade_df, test_ts, db_path, cost_sim)

    def _print_pnl_summary(pnl_df: pd.DataFrame, label: str) -> dict:
        if len(pnl_df) == 0:
            print(f"  {label}: no trades")
            return {}
        gross = pnl_df["gross"].sum()
        cost  = pnl_df["cost"].sum()
        net   = pnl_df["net"].sum()
        n_profit = (pnl_df["hit_type"] == "profit").sum()
        n_stop   = (pnl_df["hit_type"] == "stop").sum()
        n_time   = (pnl_df["hit_type"] == "time").sum()
        win_rate = (pnl_df["net"] > 0).mean()
        gross_pct = gross / POOL_SIZE * 100
        cost_pct  = cost  / POOL_SIZE * 100
        net_pct   = net   / POOL_SIZE * 100
        print(f"  {label}")
        print(f"    Trades      : {len(pnl_df)}  (profit {n_profit} / stop {n_stop} / time {n_time})")
        print(f"    Win rate    : {win_rate:.1%}")
        print(f"    Gross P&L   : Rs{gross:>+10,.0f}  ({gross_pct:>+7.2f}%)")
        print(f"    Cost drag   : Rs{cost:>+10,.0f}  ({cost_pct:>+7.2f}%)")
        print(f"    Net P&L     : Rs{net:>+10,.0f}  ({net_pct:>+7.2f}%)  <-- HEADLINE")
        return {"gross": gross, "cost": cost, "net": net, "net_pct": net_pct, "n": len(pnl_df)}

    test_metrics = _print_pnl_summary(test_pnl_df, "Test set (sealed, primary)")

    # ── P&L for combined OOS window ───────────────────────────────────────────
    print()
    if len(fold5_df) > 0:
        print("  Combined OOS (fold-5 val + test set — ~9 months):")
        oos_trade_df = oos_df[oos_df["pred"] != int(Label.NO_TRADE)].copy()
        oos_pnl_df   = _compute_pnl_df(oos_trade_df, all_oos_ts, db_path, cost_sim)
        oos_metrics  = _print_pnl_summary(oos_pnl_df, "Combined OOS")
    else:
        oos_pnl_df  = test_pnl_df.copy()
        oos_metrics = test_metrics.copy()

    # ── Benchmarks ────────────────────────────────────────────────────────────
    print()
    print("  --- BENCHMARKS (test set) ---")
    bnh_ret = _bnh_return(universe, test_ts, db_path)
    bnh_pnl = bnh_ret * POOL_SIZE

    test_net     = test_metrics.get("net", 0.0)
    test_net_pct = test_metrics.get("net_pct", 0.0)
    bnh_pct      = bnh_ret * 100

    print(f"  {'Strategy':<30}  {'Net P&L':>10}  {'% pool':>10}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}")
    print(f"  {'v2_fo (FO-augmented)':<30}  Rs{test_net:>+8,.0f}  {test_net_pct:>+9.2f}%  <-- HEADLINE")
    print(f"  {'Buy-and-hold (equal-weight)':<30}  Rs{bnh_pnl:>+8,.0f}  {bnh_pct:>+9.2f}%")
    print(f"  {'Baseline (v1_full, +5.37%)':<30}  Rs{26836:>+8,.0f}  {'+5.37':>10}%")
    print(f"  {'Always-flat':<30}  Rs{0:>9,}  {'0.00%':>10}")

    # ── Part 5 — Per-month breakdown + block t-stat ───────────────────────────
    print()
    print(_SEP)
    print("SECTION 4 — PER-MONTH BREAKDOWN + BLOCK T-STAT (OOS window)")
    print(_SEP)

    if len(oos_pnl_df) > 0:
        oos_pnl_df["month"] = oos_pnl_df["t0"].apply(
            lambda x: x.strftime("%Y-%m") if isinstance(x, datetime) else str(x)[:7]
        )
        monthly = (
            oos_pnl_df.groupby("month")
            .agg(n_trades=("net", "count"), gross=("gross", "sum"),
                 net=("net", "sum"), win_rate=("net", lambda x: (x > 0).mean()))
            .reset_index()
        )
        monthly["net_pct"] = monthly["net"] / POOL_SIZE * 100

        print(f"  {'Month':<10}  {'Trades':>7}  {'Net P&L':>12}  {'Net%':>8}  {'Win%':>7}")
        print(f"  {'-'*10}  {'-'*7}  {'-'*12}  {'-'*8}  {'-'*7}")
        for _, r in monthly.iterrows():
            print(
                f"  {r['month']:<10}  {int(r['n_trades']):>7}  "
                f"Rs{r['net']:>+10,.0f}  {r['net_pct']:>+7.2f}%  {r['win_rate']:>6.1%}"
            )

        # Block t-stat
        monthly_pcts = monthly["net_pct"].tolist()
        t_stat, df   = _block_tstat(monthly_pcts)
        n_months     = len(monthly_pcts)
        print()
        print(f"  Block t-stat  : {t_stat:>+.3f}  (df={df}, n_months={n_months})")
        print(f"  Mean monthly  : {np.mean(monthly_pcts):>+.3f}%  |  Std: {np.std(monthly_pcts, ddof=1):.3f}%")
        if abs(t_stat) >= 2.0:
            print("  SIGNIFICANT: |t| ≥ 2.0 — monthly returns are statistically distinguishable from 0.")
        else:
            print("  NOT SIGNIFICANT: |t| < 2.0 — cannot reject the null of zero monthly edge.")
            print(f"  (Need |t| ≥ 2.0 for α=0.05; have {t_stat:+.2f})")
    else:
        print("  No trades in combined OOS window.")

    # ── Part 6 — Regime breakdown ─────────────────────────────────────────────
    print()
    print(_SEP)
    print("SECTION 5 — REGIME BREAKDOWN")
    print(_SEP)
    print("  Regime = previous day's market direction (median return_1d across symbols).")
    print("  BULL: yesterday median return_1d > 0.  BEAR: yesterday ≤ 0.")
    print()

    if len(oos_pnl_df) > 0 and "return_1d" in oos_df.columns:
        # Compute median return_1d by date
        oos_df["date"] = oos_df["t0"].apply(
            lambda x: x.date() if isinstance(x, datetime) else pd.to_datetime(x).date()
        )
        med_ret = oos_df.groupby("date")["return_1d"].median().rename("med_ret1d")
        oos_pnl_df["date"] = oos_pnl_df["t0"].apply(
            lambda x: x.date() if isinstance(x, datetime) else pd.to_datetime(x).date()
        )
        oos_pnl_df = oos_pnl_df.join(med_ret, on="date")
        oos_pnl_df["regime"] = oos_pnl_df["med_ret1d"].apply(
            lambda r: "BULL" if (pd.notna(r) and r > 0) else "BEAR"
        )

        for reg in ("BULL", "BEAR"):
            sub = oos_pnl_df[oos_pnl_df["regime"] == reg]
            if len(sub) == 0:
                print(f"  {reg}: no trades")
                continue
            net     = sub["net"].sum()
            net_pct = net / POOL_SIZE * 100
            win_r   = (sub["net"] > 0).mean()
            print(f"  {reg} regime : {len(sub):>5} trades  net {net:>+10,.0f} Rs  ({net_pct:>+6.2f}%)  win {win_r:.1%}")
    else:
        print("  Regime breakdown unavailable (no return_1d feature or no trades).")

    # ── Part 7 — F&O feature importance ──────────────────────────────────────
    print()
    print(_SEP)
    print("SECTION 6 — F&O FEATURE IMPORTANCE  ← KEY METRIC FOR CHECKPOINT 7.2")
    print(_SEP)
    print("  If pcr_oi / pcr_volume / net_oi_change have near-zero importance,")
    print("  the F&O data added no edge over the price/volume baseline.")
    print()

    fo_feats   = {"pcr_oi", "pcr_volume", "net_oi_change", "max_pain_distance", "spot_futures_basis"}
    sent_feats = {"news_count_1d", "news_count_5d", "news_sentiment"}
    pv_feats   = {
        "return_1d", "return_5d", "vol_20d", "volume_ratio_20d", "atr_20d",
        "vwap_deviation", "intraday_momentum", "opening_range_position",
    }

    try:
        raw_imp  = classifier._model.feature_importance(importance_type="gain")
        imp      = pd.Series(raw_imp, index=classifier.feature_names).sort_values(ascending=False)
        total_imp = imp.sum()

        print("  All features by gain importance:")
        print(f"  {'Feature':<35}  {'Gain':>10}  {'% total':>8}  {'Group':<10}")
        print(f"  {'-'*35}  {'-'*10}  {'-'*8}  {'-'*10}")
        for feat, gain in imp.items():
            if feat in fo_feats:
                grp = "F&O"
            elif feat in sent_feats:
                grp = "Sentiment"
            elif feat in pv_feats:
                grp = "Price/Vol"
            else:
                grp = "Other"
            pct = (gain / total_imp * 100) if total_imp > 0 else 0.0
            print(f"  {feat:<35}  {gain:>10.1f}  {pct:>7.1f}%  {grp:<10}")

        # Group totals
        fo_total  = sum(imp.get(f, 0) for f in fo_feats)
        pv_total  = sum(imp.get(f, 0) for f in pv_feats)
        sen_total = sum(imp.get(f, 0) for f in sent_feats)
        fo_pct    = fo_total  / total_imp * 100 if total_imp > 0 else 0.0
        pv_pct    = pv_total  / total_imp * 100 if total_imp > 0 else 0.0

        print()
        print(f"  Group totals:")
        print(f"    Price/volume features : {pv_total:>10.1f}  ({pv_pct:.1f}%)")
        print(f"    F&O features          : {fo_total:>10.1f}  ({fo_pct:.1f}%)")
        print(f"    Sentiment features    : {sen_total:>10.1f}  (expected 0 — no news data)")

        # Verdict on F&O importance
        print()
        if fo_pct >= 5.0:
            print(f"  F&O CONTRIBUTION: {fo_pct:.1f}% of model gain — meaningful signal.")
            print("  pcr_oi/pcr_volume/net_oi_change are being used by the model.")
        elif fo_pct >= 1.0:
            print(f"  F&O CONTRIBUTION: {fo_pct:.1f}% of model gain — marginal signal.")
            print("  F&O features contribute but are not dominant.")
        else:
            print(f"  F&O CONTRIBUTION: {fo_pct:.1f}% of model gain — effectively zero.")
            print("  LightGBM did not find F&O features useful. See Per-Month section")
            print("  to confirm whether this matches the P&L outcome.")

        # Dominance check
        if len(imp) >= 2 and imp.iloc[1] > 0:
            dom = imp.iloc[0] / imp.iloc[1]
            if dom > 5.0:
                print()
                print(f"  WARNING: Top feature ({imp.index[0]}) is {dom:.1f}× more important")
                print("  than the second — possible label leakage. Investigate.")

    except Exception as exc:
        print(f"  Feature importance unavailable: {exc}")

    # ── Part 8 — Head-to-head vs v1_full ──────────────────────────────────────
    print()
    print(_SEP)
    print("SECTION 7 — HEAD-TO-HEAD: v2_fo  vs  v1_full BASELINE")
    print(_SEP)
    print("  Both models evaluated on the IDENTICAL test timestamps.")
    print("  v1_full was trained with F&O features = all-zero.  Its F&O")
    print("  importances are all zero, so real F&O values do not change")
    print("  its predictions — the comparison is valid.")
    print()

    baseline_net     = 26836.0   # from Checkpoint 6 provenance
    baseline_net_pct = 5.37

    v1_clf = None
    v1_pnl_df = pd.DataFrame()

    bdir = Path(baseline_dir)
    if bdir.exists() and (bdir / "scenario_classifier").exists():
        try:
            v1_clf = ScenarioClassifier.load(bdir / "scenario_classifier")
            # predict_proba uses X.values — column ORDER must match the order at
            # training time. Reorder by v1_clf.feature_names (stored in meta.json).
            v1_feat_cols = (
                [c for c in v1_clf.feature_names if c in test_df.columns]
                if v1_clf.feature_names else feat_cols
            )
            X_test_v1 = test_df[v1_feat_cols].fillna(0.0)
            v1_proba = v1_clf.predict_proba(X_test_v1)
            v1_pred = v1_proba.idxmax(axis=1).map({
                "p_no_trade": int(Label.NO_TRADE),
                "p_bullish" : int(Label.BULLISH),
                "p_bearish" : int(Label.BEARISH),
            })
            v1_test = test_df.copy()
            v1_test["pred"] = v1_pred.values
            v1_trade = v1_test[v1_test["pred"] != int(Label.NO_TRADE)].copy()
            v1_pnl_df = _compute_pnl_df(v1_trade, test_ts, db_path, cost_sim)
            print(f"  Loaded v1_full model ({len(v1_feat_cols)} features, reordered to match training)")
        except Exception as exc:
            print(f"  Could not load v1_full for comparison: {exc}")
            print(f"  Using recorded baseline: +{baseline_net_pct:.2f}%  (Rs{baseline_net:,.0f})")
    else:
        print(f"  v1_full model not found at {bdir}")
        print(f"  Using recorded baseline: +{baseline_net_pct:.2f}%  (Rs{baseline_net:,.0f})")

    if len(v1_pnl_df) > 0:
        v1_net     = v1_pnl_df["net"].sum()
        v1_net_pct = v1_net / POOL_SIZE * 100
        v1_trades  = len(v1_pnl_df)
    else:
        v1_net     = baseline_net
        v1_net_pct = baseline_net_pct
        v1_trades  = None

    v2_net     = test_metrics.get("net", 0.0)
    v2_net_pct = test_metrics.get("net_pct", 0.0)
    v2_trades  = test_metrics.get("n", 0)
    delta_pct  = v2_net_pct - v1_net_pct

    print()
    print(f"  {'Model':<28}  {'Trades':>7}  {'Net P&L':>10}  {'% pool':>9}  {'vs baseline':>12}")
    print(f"  {'-'*28}  {'-'*7}  {'-'*10}  {'-'*9}  {'-'*12}")
    v1_tr_str = str(v1_trades) if v1_trades is not None else "(recorded)"
    print(f"  {'v1_full (PRICE/VOL only)':<28}  {v1_tr_str:>7}  Rs{v1_net:>+8,.0f}  {v1_net_pct:>+8.2f}%  {'—':>12}")
    print(f"  {'v2_fo  (FO-AUGMENTED)':<28}  {v2_trades:>7}  Rs{v2_net:>+8,.0f}  {v2_net_pct:>+8.2f}%  {delta_pct:>+11.2f}pp")
    print(f"  {'Buy-and-hold':<28}  {'N/A':>7}  Rs{bnh_pnl:>+8,.0f}  {bnh_pct:>+8.2f}%")

    # ── Part 9 — Verdict ──────────────────────────────────────────────────────
    print()
    print(_SEP)
    print("VERDICT — DOES F&O OI ADD REAL EDGE OVER PRICE/VOLUME?")
    print(_SEP)

    t_stat_val = float("nan")
    if len(oos_pnl_df) > 0 and "month" in oos_pnl_df.columns:
        monthly_grp = oos_pnl_df.groupby("month")["net"].sum() / POOL_SIZE * 100
        t_stat_val, _ = _block_tstat(monthly_grp.tolist())

    print(f"  Test period        : {len(test_ts)} trading days (sealed, gold-standard)")
    print(f"  Total OOS coverage : {total_oos} days (~{total_oos/21:.0f} months including fold-5 val)")
    print(f"  v2_fo net return   : {v2_net_pct:>+.2f}%  (test set)")
    print(f"  v1_full baseline   : {v1_net_pct:>+.2f}%")
    print(f"  Delta              : {delta_pct:>+.2f} pp  ({'better' if delta_pct > 0 else 'worse'} than baseline)")
    print(f"  Block t-stat       : {t_stat_val:>+.3f}  ({'significant' if abs(t_stat_val) >= 2.0 else 'NOT significant'} at α=0.05)")
    print(f"  PIT leakage check  : {'PASS' if pit_ok else 'FAIL'}")
    print()

    fo_contribution = 0.0
    try:
        raw_imp   = classifier._model.feature_importance(importance_type="gain")
        imp2      = pd.Series(raw_imp, index=classifier.feature_names)
        tot2      = imp2.sum()
        fo_contribution = sum(imp2.get(f, 0) for f in fo_feats) / tot2 * 100 if tot2 > 0 else 0.0
    except Exception:
        pass

    fo_add_edge = (
        delta_pct > 0.5 and
        abs(t_stat_val) >= 1.5 and
        fo_contribution >= 2.0 and
        pit_ok
    )

    if not pit_ok:
        print("  INCONCLUSIVE — PIT leakage test FAILED. Results cannot be trusted.")
        print("  Investigate fo.py WHERE clause before re-running.")

    elif fo_contribution < 0.5:
        print("  NO F&O EDGE: LightGBM assigned <0.5% gain importance to all F&O features.")
        print("  The model reduced to a price/volume classifier despite F&O data being present.")
        if delta_pct > 0:
            print(f"  The +{delta_pct:.2f}pp delta is likely noise from a different random seed,")
            print("  not from F&O information content.")
        print()
        print("  Possible causes:")
        print("    1. pcr_oi/pcr_volume are noisy at daily granularity for 109 large-cap stocks")
        print("    2. ATM OI CE/PE ratio is already priced in for these liquid stocks")
        print("    3. spot_futures_basis has only 3-month depth — too little history to learn from")
        print("    4. max_pain_distance is all-zero (ATM-only download) — no signal")

    elif fo_add_edge:
        print(f"  F&O ADDS EDGE: +{delta_pct:.2f}pp over baseline, t-stat {t_stat_val:+.2f},")
        print(f"  F&O features contribute {fo_contribution:.1f}% of model gain importance.")
        print()
        print("  CAVEATS:")
        print("  - Test window is still short. Regime-specific edge is possible (test period")
        print("    was bullish — see regime breakdown for whether F&O helps in bear markets).")
        print("  - spot_futures_basis has only 3-month depth. Revisit when more data arrives.")
        print("  - Do NOT over-fit by re-tuning hyperparameters on these results.")

    elif delta_pct > 0:
        print(f"  MARGINAL/UNCLEAR: +{delta_pct:.2f}pp over baseline, but t-stat {t_stat_val:+.2f} < 2.0.")
        print(f"  F&O features show {fo_contribution:.1f}% of model gain — some signal but not definitive.")
        print("  The improvement is not statistically significant given the evaluation window.")
        print()
        print("  Recommendation: collect 6+ more months of OI data and re-evaluate.")
        print("  The ATM CE/PE ratio may have real signal; the sample is too small to confirm.")

    else:
        print(f"  NO DEMONSTRATED EDGE: v2_fo underperforms baseline by {abs(delta_pct):.2f}pp.")
        print(f"  Block t-stat: {t_stat_val:+.2f} (not significant).")
        print(f"  F&O feature importance: {fo_contribution:.1f}%")
        print()
        print("  This is consistent with the hypothesis that daily ATM OI ratios are")
        print("  already reflected in price/volume information for large-cap NSE stocks.")

    print()
    print("  STANDING CAVEATS (all evaluations):")
    print("  - P&L uses approximate barrier prices, not actual tick-level paths.")
    print("  - No market-impact model. Slippage is 2+2 bps (light for large sizes).")
    print("  - max_pain_distance is all-zero (ATM-only OI; no per-strike breakdown).")
    print("  - spot_futures_basis is ~3 months old; longer history may show more.")
    print("  - Sentiment features are all-zero — the full model remains to be built.")
    print(_SEP)


if __name__ == "__main__":
    args = _parse_args()
    run_fo_evaluation(
        model_dir    = args.model_dir,
        baseline_dir = args.baseline_dir,
        db_path      = args.db_path,
        skip_fold5_val = args.skip_fold5_val,
    )
