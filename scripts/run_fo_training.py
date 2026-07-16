"""Retrain with F&O features (v2_fo model).

MODEL TYPE: FO_AUGMENTED
F&O features are NOW REAL from fo_daily_metrics (Checkpoint 7.1 complete):
  - pcr_oi             REAL — 3-yr depth (CE ATM OI / PE ATM OI ratio)
  - pcr_volume         REAL — 3-yr depth (CE ATM volume / PE ATM volume ratio)
  - net_oi_change      REAL — 3-yr depth (normalised CE-PE net OI change)
  - spot_futures_basis PARTIAL — ~3 months only (active contracts; expired → no data)
  - max_pain_distance  NULL/ZERO — ATM-only OI; no per-strike breakdown available

Date range is IDENTICAL to v1_full (2023-07-05 → 2026-07-03) so the test sets are
byte-for-byte the same timestamps — enabling a direct head-to-head comparison.

Baseline: models/v1_full (PRICE_VOLUME_ONLY) achieved +5.37% net on the sealed test
set in Checkpoint 6.  The question is: do pcr_oi / pcr_volume / net_oi_change add
real edge on top of price/volume information?

Usage
-----
    python -m scripts.run_fo_training
    # or from project root:
    .venv\\Scripts\\python.exe scripts\\run_fo_training.py
"""
from __future__ import annotations

import io
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

# Ensure project root is on sys.path so 'src.*' imports resolve when the script
# is launched directly (e.g. via Start-Process python scripts\run_fo_training.py).
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── stdout must be UTF-8 for Windows console ─────────────────────────────────
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.config import load_config
from src.models.training_pipeline import run_training

cfg = load_config()

UNIVERSE = cfg.universe.symbols    # 109 symbols from config.toml [universe]
DB_PATH  = "data/market.duckdb"
OUT_DIR  = "models/v2_fo"
START    = date(2023, 7, 5)        # SAME as v1_full — identical test set for comparison
END      = date(2026, 7, 3)        # SAME as v1_full
CPU_CAP  = 6                       # 6/8 cores — safe cap for dev laptop

_SEP = "=" * 72

print(_SEP, flush=True)
print(f"FO-AUGMENTED TRAINING — {len(UNIVERSE)} symbols, {START} to {END}", flush=True)
print(f"CPU cap  : {CPU_CAP} / 8 threads", flush=True)
print(f"Output   : {OUT_DIR}", flush=True)
print(flush=True)
print("FEATURE HONESTY STAMP", flush=True)
print("  Price/volume features (REAL, 8 features, 3-yr depth):", flush=True)
print("    return_1d, return_5d, vol_20d, volume_ratio_20d, atr_20d,", flush=True)
print("    vwap_deviation, intraday_momentum, opening_range_position", flush=True)
print("  F&O features:", flush=True)
print("    pcr_oi          — REAL, 3-yr depth (CE ATM OI / PE ATM OI)", flush=True)
print("    pcr_volume      — REAL, 3-yr depth (CE ATM vol / PE ATM vol)", flush=True)
print("    net_oi_change   — REAL, 3-yr depth (norm. CE-PE net OI change)", flush=True)
print("    spot_futures_basis — PARTIAL, ~3 months (active contracts only)", flush=True)
print("    max_pain_distance  — NULL/ZERO (ATM-only download; no per-strike)", flush=True)
print("  Sentiment features (ALL ZERO — no historical news feed):", flush=True)
print("    news_count_1d, news_count_5d, news_sentiment", flush=True)
print(flush=True)
print("BASELINE FOR COMPARISON: models/v1_full (+5.37% net, Checkpoint 6)", flush=True)
print(_SEP, flush=True)

t_start = time.perf_counter()

prov = run_training(
    db_path=DB_PATH,
    output_dir=OUT_DIR,
    universe=UNIVERSE,
    start_date=START,
    end_date=END,
    max_cpu_cores=CPU_CAP,
    dry_run=False,
)

elapsed = time.perf_counter() - t_start

# ── Stamp model_type + feature honesty in provenance ─────────────────────────
prov_path = Path(OUT_DIR) / "provenance.json"
with open(prov_path) as f:
    prov_full = json.load(f)

prov_full["model_type"] = "FO_AUGMENTED"
prov_full["feature_honesty"] = {
    "price_volume_features": "REAL — 8 features from Dhan intraday OHLCV (3-yr depth)",
    "fo_pcr_oi":             "REAL — CE ATM OI / PE ATM OI.  3-yr depth. 92% CE coverage.",
    "fo_pcr_volume":         "REAL — CE ATM vol / PE ATM vol. 3-yr depth. 92% coverage.",
    "fo_net_oi_change":      "REAL — norm. (CE OI Δ - PE OI Δ). 3-yr depth. 93% PE coverage.",
    "fo_spot_futures_basis": "PARTIAL — ~3 months only. Active contracts; expired → NULL → 0.",
    "fo_max_pain_distance":  "NULL/ZERO — ATM-only OI downloaded. No per-strike breakdown.",
    "sentiment_features":    "ALL_ZERO — news table empty (no historical news feed)",
    "baseline":              "models/v1_full (PRICE_VOLUME_ONLY) +5.37% net — Checkpoint 6",
    "comparison_note": (
        "Date range identical to v1_full → identical test timestamps. "
        "Head-to-head comparison in run_fo_evaluation.py is apples-to-apples."
    ),
}

with open(prov_path, "w") as f:
    json.dump(prov_full, f, indent=2, default=str)

# ── Summary ───────────────────────────────────────────────────────────────────
print()
print(_SEP)
print(f"COMPLETED in {elapsed:.1f}s  ({elapsed / 60:.1f} min)")
print(_SEP)

cal = prov_full.get("calibration_provenance", {})
print("TRAINING SUMMARY")
print(f"  Symbols trained : {len(UNIVERSE)}")
print(f"  Dataset rows    : {prov_full.get('n_dataset_rows', '?')}")
print(f"  Test set rows   : {prov_full.get('test_set_rows', '?')}")
print(f"  Walk-fwd folds  : {prov_full.get('n_folds', '?')}")
print(f"  Calibrated      : {'YES' if cal.get('is_calibrated') else 'NO'}")
if cal.get("is_calibrated"):
    print(f"    Method        : {cal.get('calibration_method')}")
    print(f"    Val samples   : {cal.get('n_calibration_samples')}")
    print(f"    Val period    : {cal.get('validation_period')}")

print()
print(f"Model files : {Path(OUT_DIR).resolve()}")
print("Next step   : python -m scripts.run_fo_evaluation")
