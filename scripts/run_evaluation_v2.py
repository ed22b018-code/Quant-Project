"""Honest evaluation of the v1_full model on the sealed test set.

Evaluation methodology
----------------------
1. Calibration reliability: does P(bullish)=0.7 mean 70% accuracy?
   - Reliability diagram on the SEALED TEST SET (never seen during training
     or calibration — gold-standard evaluation).

2. After-cost benchmark using the REAL NSE cost stack:
   - CostSimulator (brokerage, STT, exchange, SEBI fees, GST, stamp duty)
   - Position budget: 10% of ₹5,00,000 = ₹50,000 per trade
   - Compared to: buy-and-hold (equal-weight), random entry, always-flat

3. Per-month performance across the test period.

4. Feature importance & leakage checks.

LIMITATIONS (state explicitly before interpreting results)
----------------------------------------------------------
- This is a PRICE/VOLUME-ONLY model (F&O and news features are all zero).
- Test period is ~10% of the date range (~78 trading days).
- Triple-barrier P&L uses APPROXIMATE exit prices (entry × barrier fraction),
  not the actual mid-price path. This is fair — same approximation for all
  strategies — but slightly overstates gross P&L when the barrier is not hit
  exactly at the edge (NO_TRADE rows assume zero gross P&L).
- Bid-ask spread is captured via slippage (2 bps entry + 2 bps exit).
  Market impact for larger sizes is NOT modelled.

Usage
-----
    python -m scripts.run_evaluation_v2              # uses models/v1_full
    python -m scripts.run_evaluation_v2 --model-dir models/my_run
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.WARNING)

import duckdb
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.config import load_config
from src.models.calibration import ModelCalibrator
from src.models.dataset import DatasetBuilder, WalkForwardSplitter
from src.models.labels import Label, TripleBarrierLabeler
from src.models.scenario_classifier import ScenarioClassifier
from src.models.training_pipeline import _build_for_symbol_worker
from src.trading._types import Direction
from src.trading.costs import CostSimulator

UTC = timezone.utc
_SEP = "=" * 72
_SEC = "-" * 60

# ── Config ────────────────────────────────────────────────────────────────────
POOL_SIZE      = 500_000    # ₹5L — from config.toml [trading]
MAX_POS_PCT    = 0.10       # 10% max position
POSITION_BUDGET = POOL_SIZE * MAX_POS_PCT  # ₹50,000 per trade
PROFIT_PCT     = 0.010      # triple-barrier profit target (+1%)
STOP_PCT       = 0.005      # triple-barrier stop (-0.5%)
HORIZON_BARS   = 30         # 30 × 1-min bars = 30-min horizon
SLIPPAGE_BPS   = 2.0        # 2 bps entry + 2 bps exit (from config)
CPU_CAP        = 6


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models/v1_full")
    p.add_argument("--db-path", default="data/market.duckdb")
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _test_timestamps(prov: dict) -> list[datetime]:
    """Replicate WalkForwardSplitter.get_final_test_set() to get test t0s."""
    start = pd.to_datetime(prov["date_range"][0]).date()
    end   = pd.to_datetime(prov["date_range"][1]).date()
    bdays = pd.bdate_range(start=start, end=end)
    all_ts = [datetime(d.year, d.month, d.day, 4, 0, tzinfo=UTC) for d in bdays]
    n_test = max(1, int(len(all_ts) * 0.10))
    return all_ts[-n_test:]


def _fetch_entry_prices(
    symbols: list[str], test_ts: list[datetime], db_path: str
) -> dict[tuple[str, datetime], float]:
    """Get the 1-min close at 9:30 IST (market open) for each (symbol, date).

    The DB stores timestamps in IST (Asia/Calcutta, UTC+5:30).
    EXTRACT(HOUR FROM ts) returns the IST hour, so the market-open bar
    at 9:30 IST (= 4:00 UTC) appears as HOUR=9, MINUTE=30.
    """
    if not symbols or not test_ts:
        return {}

    # DuckDB cannot take a Python list of tz-aware datetimes directly;
    # cast to date (IST date matches the UTC date for daytime bars).
    test_dates = sorted({t.date() for t in test_ts})
    date_strs  = ", ".join(f"'{d}'" for d in test_dates)
    sym_list   = ", ".join(f"'{s}'" for s in symbols)

    query = f"""
        SELECT symbol, ts, close
        FROM   ohlcv_intraday
        WHERE  timeframe = '1'
          AND  symbol IN ({sym_list})
          AND  CAST(ts AS DATE) IN ({date_strs})
          AND  EXTRACT(HOUR   FROM ts) = 9
          AND  EXTRACT(MINUTE FROM ts) = 30
    """
    with duckdb.connect(db_path, read_only=True, config={"threads": "2"}) as con:
        rows = con.execute(query).fetchall()

    prices: dict[tuple[str, datetime], float] = {}
    for symbol, ts_raw, close in rows:
        # Normalise ts to tz-aware UTC
        if isinstance(ts_raw, datetime):
            ts_utc = ts_raw.replace(tzinfo=UTC) if ts_raw.tzinfo is None else ts_raw.astimezone(UTC)
        else:
            ts_utc = datetime(ts_raw.year, ts_raw.month, ts_raw.day, 4, 0, tzinfo=UTC)
        prices[(symbol, ts_utc)] = float(close)
    return prices


def _exit_price(entry: float, label: int, pred_dir: str) -> tuple[float, str]:
    """Approximate exit price from triple-barrier label.

    Returns (exit_price, hit_type) where hit_type in {'profit','stop','time'}.

    The label fires at the ACTUAL barrier price:
      BULLISH: price hit entry * (1 + PROFIT_PCT)  — upper barrier
      BEARISH: price hit entry * (1 - STOP_PCT)    — lower barrier
      NO_TRADE: neither barrier hit within horizon_bars

    For LONG:  profit at +PROFIT_PCT (+1%), stop at -STOP_PCT (-0.5%)
    For SHORT: profit at -STOP_PCT (-0.5%), stop at +STOP_PCT (+0.5%)

    SHORT profit uses STOP_PCT (not PROFIT_PCT) because the BEARISH label fires
    when price falls by STOP_PCT. Using PROFIT_PCT would claim a -1% exit that
    never happened — the label only guarantees -0.5%.

    SHORT stop uses STOP_PCT (0.5% adverse) because when BULLISH fires (+1%),
    the SHORT stop at +0.5% was already triggered on the way there.
    """
    if pred_dir == "LONG":
        if label == int(Label.BULLISH):
            return entry * (1 + PROFIT_PCT), "profit"
        elif label == int(Label.BEARISH):
            return entry * (1 - STOP_PCT), "stop"
        else:  # NO_TRADE / time barrier
            return entry, "time"
    else:  # SHORT
        if label == int(Label.BEARISH):
            # BEARISH label fires at -STOP_PCT (-0.5%). That IS the SHORT profit.
            return entry * (1 - STOP_PCT), "profit"
        elif label == int(Label.BULLISH):
            # BULLISH fires at +PROFIT_PCT (+1%); SHORT stop at +STOP_PCT (+0.5%)
            # was hit on the way — exit at +0.5% adverse.
            return entry * (1 + STOP_PCT), "stop"
        else:
            return entry, "time"


def _slippage_cost(entry: float, qty: int) -> float:
    """4 bps round-trip slippage (2 bps each way)."""
    return entry * qty * (SLIPPAGE_BPS * 2) / 10_000


def _reliability_rows(
    y_true_binary: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> list[tuple[float, float, int]]:
    """Return list of (bin_center, empirical_freq, count) for non-empty bins."""
    cal = ModelCalibrator()
    rd  = cal.reliability_diagram(y_true_binary, y_prob, n_bins=n_bins)
    rows = []
    for center, freq, count in zip(
        rd["bin_centers"], rd["empirical_freqs"], rd["counts"]
    ):
        if count > 0:
            rows.append((center, freq, count))
    return rows, rd["brier_score"], rd["max_calibration_error"]


def _print_reliability(label_name: str, rows, brier, max_err) -> None:
    print(f"\n  Calibration — {label_name}")
    print(f"  {'Pred prob':>10}  {'Empirical':>10}  {'Gap':>8}  {'N':>6}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*6}")
    for center, freq, count in rows:
        gap = freq - center
        print(f"  {center:>10.2f}  {freq:>10.2f}  {gap:>+8.3f}  {count:>6}")
    print(f"  Brier score: {brier:.4f}  (lower=better; naive=0.25 for balanced binary)")
    print(f"  Max |pred-empirical| across bins: {max_err:.3f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation(model_dir: str, db_path: str) -> None:
    mdir = Path(model_dir)
    if not mdir.exists():
        sys.exit(f"Model directory not found: {mdir.resolve()}")

    prov_path = mdir / "provenance.json"
    if not prov_path.exists():
        sys.exit(f"provenance.json not found in {mdir}")

    with open(prov_path) as f:
        prov = json.load(f)

    cfg = load_config()
    cost_sim = CostSimulator(cfg.trading.costs)

    print(_SEP)
    print("MODEL EVALUATION — SEALED TEST SET")
    print(_SEP)
    print(f"Model dir    : {mdir.resolve()}")
    print(f"Model type   : {prov.get('model_type', 'UNKNOWN')}")
    print(f"DB           : {db_path}")
    print(f"Date range   : {prov['date_range'][0]} to {prov['date_range'][1]}")

    fh = prov.get("feature_honesty", {})
    if fh:
        print("\nFEATURE HONESTY:")
        for k, v in fh.items():
            if k != "warning":
                print(f"  {k}: {v}")
        if "warning" in fh:
            print(f"  WARNING: {fh['warning']}")

    # ── 1. Test timestamps ────────────────────────────────────────────────────
    test_ts = _test_timestamps(prov)
    print(f"\nTest period  : {test_ts[0].date()} to {test_ts[-1].date()} ({len(test_ts)} days)")

    universe = prov["universe"]
    feature_names = prov["feature_names"]

    # ── 2. Load classifier ────────────────────────────────────────────────────
    clf_dir = mdir / "scenario_classifier"
    if not clf_dir.exists():
        sys.exit(f"scenario_classifier/ not found in {mdir}")
    classifier = ScenarioClassifier.load(clf_dir)
    print(f"Classifier   : loaded ({len(classifier.feature_names)} features)")

    # ── 3. Rebuild test dataset ───────────────────────────────────────────────
    print(f"\nBuilding test dataset for {len(universe)} symbols × {len(test_ts)} days...")
    builder = DatasetBuilder(feature_names=feature_names)
    labeler = TripleBarrierLabeler(
        profit_pct=PROFIT_PCT, stop_pct=STOP_PCT, horizon_bars=HORIZON_BARS
    )

    parts: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=CPU_CAP) as ex:
        futs = {
            ex.submit(_build_for_symbol_worker, sym, db_path, test_ts, builder, labeler): sym
            for sym in universe
        }
        done = 0
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                parts.append(fut.result())
                done += 1
                if done % 20 == 0 or done == len(universe):
                    print(f"  {done}/{len(universe)} symbols built")
            except Exception as exc:
                print(f"  ERROR [{sym}]: {exc}")

    if not parts:
        sys.exit("No data built for test set.")

    test_df = pd.concat(parts, ignore_index=True).sort_values("t0").reset_index(drop=True)
    # Keep only rows with a label
    test_df = test_df[test_df["label"].notna()].copy()
    feature_cols = [c for c in test_df.columns if c in feature_names]
    X_test = test_df[feature_cols].fillna(0.0)
    y_test = test_df["label"].astype(int)

    print(f"Test rows    : {len(test_df)} (with valid labels)")
    print(f"Label dist   : {y_test.value_counts().to_dict()}")

    if len(test_df) == 0:
        sys.exit("Test dataset is empty after label filter.")

    # ── 4. Predictions ────────────────────────────────────────────────────────
    proba = classifier.predict_proba(X_test)
    # Predicted class: argmax of probabilities
    pred_class = proba.idxmax(axis=1).map({
        "p_no_trade": int(Label.NO_TRADE),
        "p_bullish" : int(Label.BULLISH),
        "p_bearish" : int(Label.BEARISH),
    })
    test_df["pred"]       = pred_class.values
    test_df["p_bullish"]  = proba["p_bullish"].values
    test_df["p_bearish"]  = proba["p_bearish"].values
    test_df["p_no_trade"] = proba["p_no_trade"].values

    n_trades = (test_df["pred"] != int(Label.NO_TRADE)).sum()
    no_trade_pct = (test_df["pred"] == int(Label.NO_TRADE)).mean()
    bull_pct     = (test_df["pred"] == int(Label.BULLISH)).mean()
    bear_pct     = (test_df["pred"] == int(Label.BEARISH)).mean()

    print()
    print(_SEP)
    print("SECTION 1 — PREDICTION DISTRIBUTION")
    print(_SEP)
    print(f"  NO_TRADE predicted : {no_trade_pct:.1%} of rows  ({int(no_trade_pct*len(test_df))} rows)")
    print(f"  BULLISH  predicted : {bull_pct:.1%}  ({int(bull_pct*len(test_df))} rows)")
    print(f"  BEARISH  predicted : {bear_pct:.1%}  ({int(bear_pct*len(test_df))} rows)")
    print(f"  Total trades taken : {n_trades}")

    if no_trade_pct > 0.98:
        print()
        print("  RED FLAG: Model predicts NO_TRADE on >98% of test bars.")
        print("  This means the model has learnt to abstain almost always.")
        print("  It may have edge in the few trades it does take, or it may")
        print("  simply be defaulting to the majority class. Check label dist.")

    # ── 5. Calibration ────────────────────────────────────────────────────────
    print()
    print(_SEP)
    print("SECTION 2 — CALIBRATION (RELIABILITY DIAGRAM on sealed test set)")
    print(_SEP)
    print("  Interpretation: if bin shows pred=0.70, empirical=0.45, the model")
    print("  is OVERCONFIDENT — it says 70% likely but is only right 45% of")
    print("  the time. Perfect calibration: empirical = pred along the diagonal.")
    print()
    print("  NOTE: calibration was fitted on the LAST VALIDATION FOLD only,")
    print("  not on the test set. This reliability check is on the test set,")
    print("  so it reflects true out-of-sample calibration quality.")

    # Binary reliability for each class
    for class_label, col, name in [
        (int(Label.BULLISH),  "p_bullish",  "BULLISH"),
        (int(Label.BEARISH),  "p_bearish",  "BEARISH"),
        (int(Label.NO_TRADE), "p_no_trade", "NO_TRADE"),
    ]:
        y_bin = (y_test == class_label).astype(float).values
        y_pr  = proba[col].values
        rows, brier, max_err = _reliability_rows(y_bin, y_pr, n_bins=10)
        if len(rows) == 0:
            print(f"\n  {name}: no non-empty bins (class may be absent from test set)")
            continue
        _print_reliability(name, rows, brier, max_err)

    # High-confidence accuracy check
    print()
    print("  High-confidence accuracy check (predictions > 0.60 threshold):")
    for label_val, col, name in [
        (int(Label.BULLISH), "p_bullish", "BULLISH"),
        (int(Label.BEARISH), "p_bearish", "BEARISH"),
    ]:
        mask = test_df[col] > 0.60
        n = mask.sum()
        if n == 0:
            print(f"    {name} >60%: no predictions above threshold")
            continue
        acc = (test_df.loc[mask, "label"].astype(int) == label_val).mean()
        print(f"    {name} >60%: {acc:.1%} accurate ({n} predictions)")

    # ── 6. Fetch entry prices for cost simulation ─────────────────────────────
    print()
    print(_SEP)
    print("SECTION 3 — AFTER-COST BACKTEST")
    print(_SEP)
    print(f"  Cost engine: real NSE cost stack (CostSimulator)")
    print(f"    Brokerage: min(0.03% × turnover, Rs20) per leg")
    print(f"    STT:       0.025% on sell leg")
    print(f"    Exchange:  0.00297% per leg (NSE post-Oct-2024)")
    print(f"    SEBI:      0.0001% on total turnover")
    print(f"    GST:       18% on brokerage+exchange+SEBI")
    print(f"    Stamp:     0.003% on buy leg")
    print(f"    Slippage:  {SLIPPAGE_BPS:.0f} bps each way ({SLIPPAGE_BPS*2:.0f} bps round-trip)")
    print(f"  Position budget: Rs{POSITION_BUDGET:,.0f} (10% of Rs{POOL_SIZE:,.0f} pool)")
    print()

    # Symbols with trades
    trade_mask = test_df["pred"] != int(Label.NO_TRADE)
    trade_df   = test_df[trade_mask].copy()

    if len(trade_df) == 0:
        print("  No trades taken by model — skipping P&L computation.")
        gross_total = net_total = 0.0
        cost_total = 0.0
        n_profitable = n_loss = n_time = 0
    else:
        print(f"  Fetching entry prices for {len(trade_df)} trade rows...")
        trade_symbols = trade_df["symbol"].unique().tolist()
        price_map = _fetch_entry_prices(trade_symbols, test_ts, db_path)

        rows_out = []
        missing_prices = 0
        for _, row in trade_df.iterrows():
            sym    = row["symbol"]
            t0     = row["t0"]
            label  = int(row["label"])
            pred   = int(row["pred"])

            # Normalise t0 to tz-aware UTC for dict lookup
            if isinstance(t0, datetime):
                t0_utc = t0.replace(tzinfo=UTC) if t0.tzinfo is None else t0.astimezone(UTC)
            else:
                t0_utc = datetime(t0.year, t0.month, t0.day, 4, 0, tzinfo=UTC)

            key = (sym, t0_utc)
            entry = price_map.get(key)
            if entry is None or entry <= 0:
                missing_prices += 1
                continue

            pred_dir: Direction = "LONG" if pred == int(Label.BULLISH) else "SHORT"
            exit_p, hit_type = _exit_price(entry, label, pred_dir)

            qty = max(1, int(POSITION_BUDGET / entry))

            # CostSimulator (Decimal)
            try:
                tr = cost_sim.compute(
                    symbol=sym, qty=qty, direction=pred_dir,
                    entry_price=Decimal(str(round(entry, 4))),
                    exit_price =Decimal(str(round(exit_p, 4))),
                )
                gross_pnl = float(tr.gross_pnl)
                cost_pnl  = float(tr.costs.total)
                slip_cost = _slippage_cost(entry, qty)
                net_pnl   = gross_pnl - cost_pnl - slip_cost
            except Exception as exc:
                print(f"  CostSimulator error [{sym} {t0}]: {exc}")
                continue

            rows_out.append({
                "symbol"   : sym,
                "t0"       : t0_utc,
                "label"    : label,
                "pred_dir" : pred_dir,
                "hit_type" : hit_type,
                "entry"    : entry,
                "exit_p"   : exit_p,
                "qty"      : qty,
                "gross_pnl": gross_pnl,
                "cost_total": cost_pnl + slip_cost,
                "net_pnl"  : net_pnl,
            })

        if missing_prices > 0:
            print(f"  Warning: {missing_prices} trade rows skipped (no entry price in DB)")

        if not rows_out:
            print("  No valid trade P&L computed.")
            gross_total = net_total = cost_total = 0.0
            n_profitable = n_loss = n_time = 0
        else:
            pnl_df = pd.DataFrame(rows_out)
            gross_total = pnl_df["gross_pnl"].sum()
            cost_total  = pnl_df["cost_total"].sum()
            net_total   = pnl_df["net_pnl"].sum()

            n_profitable = (pnl_df["net_pnl"] > 0).sum()
            n_loss       = (pnl_df["net_pnl"] < 0).sum()
            n_time       = (pnl_df["hit_type"] == "time").sum()
            n_profit_hit = (pnl_df["hit_type"] == "profit").sum()
            n_stop_hit   = (pnl_df["hit_type"] == "stop").sum()

            gross_pct = gross_total / POOL_SIZE * 100
            net_pct   = net_total   / POOL_SIZE * 100
            cost_drag = cost_total  / POOL_SIZE * 100

            print(f"  Trades computed  : {len(pnl_df)}")
            print(f"  Profit bar hit   : {n_profit_hit}  ({n_profit_hit/len(pnl_df):.1%})")
            print(f"  Stop hit         : {n_stop_hit}   ({n_stop_hit/len(pnl_df):.1%})")
            print(f"  Time bar (flat)  : {n_time}  ({n_time/len(pnl_df):.1%})")
            print()
            print(f"  Gross P&L        : Rs{gross_total:+,.0f}  ({gross_pct:+.2f}% of pool)")
            print(f"  Cost drag        : Rs{cost_total:,.0f}   ({cost_drag:.2f}% of pool)")
            print(f"  Net P&L          : Rs{net_total:+,.0f}   ({net_pct:+.2f}% of pool)  <-- HEADLINE")
            print(f"  Profitable trades: {n_profitable} / {len(pnl_df)}  ({n_profitable/len(pnl_df):.1%})")

    # ── 7. Buy-and-hold benchmark ─────────────────────────────────────────────
    print()
    print("  --- BENCHMARKS ---")

    test_start_date = test_ts[0].date()
    test_end_date   = test_ts[-1].date()
    sym_list_str    = ", ".join(f"'{s}'" for s in universe)

    # Equal-weight buy-and-hold over the test period
    # DB stores IST timestamps: market open = 9:30 IST (HOUR=9, MINUTE=30)
    bnh_query = f"""
        WITH first_last AS (
            SELECT symbol,
                   MIN(ts) AS first_ts,
                   MAX(ts) AS last_ts
            FROM ohlcv_intraday
            WHERE timeframe = '1'
              AND symbol IN ({sym_list_str})
              AND CAST(ts AS DATE) >= '{test_start_date}'
              AND CAST(ts AS DATE) <= '{test_end_date}'
              AND EXTRACT(HOUR   FROM ts) = 9
              AND EXTRACT(MINUTE FROM ts) = 30
            GROUP BY symbol
        )
        SELECT f.symbol,
               i1.close AS start_close,
               i2.close AS end_close
        FROM   first_last f
        JOIN   ohlcv_intraday i1 ON i1.symbol = f.symbol AND i1.ts = f.first_ts AND i1.timeframe = '1'
        JOIN   ohlcv_intraday i2 ON i2.symbol = f.symbol AND i2.ts = f.last_ts  AND i2.timeframe = '1'
    """
    try:
        with duckdb.connect(db_path, read_only=True, config={"threads": "2"}) as con:
            bnh_rows = con.execute(bnh_query).fetchall()
        bnh_returns = []
        for sym, s_close, e_close in bnh_rows:
            if s_close and s_close > 0:
                bnh_returns.append((e_close - s_close) / s_close)
        bnh_avg = np.mean(bnh_returns) if bnh_returns else 0.0
        bnh_pnl = bnh_avg * POOL_SIZE
    except Exception as exc:
        print(f"  Buy-and-hold query failed: {exc}")
        bnh_avg = 0.0
        bnh_pnl = 0.0

    # Random baseline (same trade frequency, random direction, same costs)
    np.random.seed(42)
    if len(trade_df) > 0 and rows_out:
        # Simulate: same n_trades, random LONG/SHORT, average cost per trade
        avg_cost_per_trade = cost_total / len(rows_out)
        avg_entry = np.mean([r["entry"] for r in rows_out])
        # Random direction: 50/50
        random_gross_list = []
        for r in rows_out:
            rand_dir = "LONG" if np.random.random() > 0.5 else "SHORT"
            # Use this row's actual label (not a cross-symbol lookup)
            rg, _ = _exit_price(r["entry"], r["label"], rand_dir)
            rand_move = rg - r["entry"]
            if rand_dir == "SHORT":
                rand_move = -rand_move
            random_gross_list.append(rand_move * r["qty"])
        rand_gross = sum(random_gross_list)
        rand_net   = rand_gross - cost_total  # same cost structure
    else:
        rand_gross = rand_net = 0.0

    print()
    model_net_pct = net_total / POOL_SIZE * 100
    bnh_pct       = bnh_avg * 100
    rand_net_pct  = rand_net / POOL_SIZE * 100

    print(f"  {'Strategy':<30}  {'Net P&L':>10}  {'% of pool':>10}")
    print(f"  {'-'*30}  {'-'*10}  {'-'*10}")
    print(f"  {'Model (net of costs)':<30}  Rs{net_total:>+8,.0f}  {model_net_pct:>+9.2f}%  <-- HEADLINE")
    print(f"  {'Model (gross, before costs)':<30}  Rs{gross_total:>+8,.0f}  {gross_total/POOL_SIZE*100:>+9.2f}%")
    print(f"  {'Buy-and-hold (equal-weight)':<30}  Rs{bnh_pnl:>+8,.0f}  {bnh_pct:>+9.2f}%")
    print(f"  {'Random entry (same freq)':<30}  Rs{rand_net:>+8,.0f}  {rand_net_pct:>+9.2f}%")
    print(f"  {'Always-flat':<30}  Rs{0:>9,}  {'0.00%':>10}")

    # ── 8. Per-month breakdown ────────────────────────────────────────────────
    print()
    print(_SEP)
    print("SECTION 4 — PER-MONTH BREAKDOWN (test period)")
    print(_SEP)
    if rows_out:
        pnl_df["month"] = pnl_df["t0"].apply(
            lambda x: x.strftime("%Y-%m") if isinstance(x, datetime) else str(x)[:7]
        )
        monthly = (
            pnl_df.groupby("month")
            .agg(
                n_trades=("net_pnl", "count"),
                gross_pnl=("gross_pnl", "sum"),
                net_pnl=("net_pnl", "sum"),
                win_rate=("net_pnl", lambda x: (x > 0).mean()),
            )
            .reset_index()
        )
        print(f"  {'Month':<10}  {'Trades':>7}  {'Gross P&L':>12}  {'Net P&L':>12}  {'Win%':>7}")
        print(f"  {'-'*10}  {'-'*7}  {'-'*12}  {'-'*12}  {'-'*7}")
        for _, r in monthly.iterrows():
            print(
                f"  {r['month']:<10}  {int(r['n_trades']):>7}  "
                f"Rs{r['gross_pnl']:>+10,.0f}  Rs{r['net_pnl']:>+10,.0f}  "
                f"{r['win_rate']:>6.1%}"
            )
    else:
        print("  No trades to break down by month.")

    # ── 9. Feature importance & leakage ──────────────────────────────────────
    print()
    print(_SEP)
    print("SECTION 5 — FEATURE IMPORTANCE & LEAKAGE CHECKS")
    print(_SEP)
    try:
        raw_imp = classifier._model.feature_importance(importance_type="gain")
        imp_series = pd.Series(raw_imp, index=classifier.feature_names).sort_values(ascending=False)
        top5 = imp_series.head(5)
        print("  Top-5 features by gain importance:")
        for feat, imp in top5.items():
            print(f"    {feat:<35s}: {imp:>8.1f}")

        # Check: F&O/sentiment features should have zero importance (all-zero inputs)
        fo_feats   = {"pcr_oi","pcr_volume","net_oi_change","max_pain_distance","spot_futures_basis"}
        sent_feats = {"news_count_1d","news_count_5d","news_sentiment"}
        zero_feats = fo_feats | sent_feats
        leaking = [(f, imp_series.get(f, 0)) for f in zero_feats if imp_series.get(f, 0) > 0]
        if leaking:
            print()
            print("  RED FLAG: These all-zero features have non-zero importance:")
            print("  (zero-variance features should be completely ignored by LightGBM)")
            for f, imp in leaking:
                print(f"    {f}: {imp:.1f}")
            print("  This suggests either (a) they are not actually all-zero, or")
            print("  (b) there is a feature-construction bug — investigate before trusting results.")
        else:
            print()
            print("  OK: All F&O and sentiment features have zero importance.")
            print("      LightGBM correctly ignored the all-zero columns.")

        # Dominance check
        if len(imp_series) >= 2 and imp_series.iloc[1] > 0:
            dominance = imp_series.iloc[0] / imp_series.iloc[1]
            if dominance > 5.0:
                print()
                print(f"  WARNING: Top feature is {dominance:.1f}x more important than the second.")
                print("  This level of dominance is unusual and may indicate label leakage.")
                print(f"  Top feature: {imp_series.index[0]}")
            else:
                print(f"  OK: No single feature dominates (top/second ratio = {dominance:.1f}x).")

    except Exception as exc:
        print(f"  Feature importance unavailable: {exc}")

    # ── 10. Verdict ───────────────────────────────────────────────────────────
    print()
    print(_SEP)
    print("VERDICT")
    print(_SEP)
    print(f"  Model type   : PRICE/VOLUME ONLY (F&O and sentiment features = 0)")
    print(f"  Test period  : {len(test_ts)} trading days")
    print(f"  Total trades : {len(rows_out) if rows_out else 0}")
    print()

    if not rows_out or len(rows_out) < 5:
        print("  INCONCLUSIVE: Too few trades (<5) to assess edge reliably.")
        print("  The model may be correctly abstaining or simply defaulting to NO_TRADE.")
    elif net_total > 0 and net_total > rand_net and net_total > 0.5 * bnh_pnl:
        print("  POTENTIAL EDGE: Model beats random and shows positive net return.")
        print("  Caveat: price/volume-only; short test window; limited statistical power.")
        print("  Do NOT deploy until F&O + news features are added and re-validated.")
    elif net_total > 0 and net_total <= rand_net:
        print("  WEAK/MARGINAL: Model is profitable but does not beat random entry.")
        print("  This is consistent with no real edge — the wins may be noise.")
    elif net_total > -0.5 * abs(bnh_pnl):
        print("  NO CLEAR EDGE: Model shows negative or minimal net return after costs.")
        print("  This is the expected baseline for a price/volume-only model on liquid NSE")
        print("  stocks. The market has priced in the price/volume signal.")
        print("  Actionable: add F&O/news features and re-train when data is available.")
    else:
        print("  NO EDGE: Negative net return worse than buy-and-hold. Model is hurting.")
        print("  Check for: high trade frequency draining costs, degenerate predictions,")
        print("  or systematic bias (always bullish / always bearish).")

    print()
    print("  IMPORTANT CAVEATS:")
    print(f"  - Test window is only {len(test_ts)} days — statistical power is limited.")
    print("  - P&L uses approximate barrier prices, not actual tick-level exit prices.")
    print("  - No market-impact model; slippage model is 2+2 bps (light for large sizes).")
    print("  - F&O and sentiment features are all-zero; real model will differ.")
    print(_SEP)


if __name__ == "__main__":
    args = _parse_args()
    run_evaluation(args.model_dir, args.db_path)
