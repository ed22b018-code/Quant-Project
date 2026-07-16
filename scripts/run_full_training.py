"""Full 109-symbol training run.

MODEL TYPE: PRICE_VOLUME_ONLY
- 8 price/volume features are REAL (3 years of Dhan intraday data)
- F&O features (pcr_oi, pcr_volume, net_oi_change, max_pain_distance,
  spot_futures_basis): ALL ZERO — fo_daily_metrics table is empty
- Sentiment features (news_count_1d, news_count_5d, news_sentiment):
  ALL ZERO — news table is empty

This is the price/volume baseline. DO NOT interpret results as a full-feature
model. F&O and sentiment will be added when that data is available.

Usage
-----
    python -m scripts.run_full_training
"""
from __future__ import annotations

import io
import json
import logging
import sys
import time
from datetime import date
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from src.config import load_config
from src.models.training_pipeline import run_training

cfg = load_config()

UNIVERSE = cfg.universe.symbols   # 109 symbols — config.toml [universe]
DB_PATH  = "data/market.duckdb"
OUT_DIR  = "models/v1_full"
START    = date(2023, 7, 5)
END      = date(2026, 7, 3)
CPU_CAP  = 6    # 6 of 8 cores (75 %) — safe cap for laptop

_SEP = "=" * 72

print(_SEP, flush=True)
print(f"FULL TRAINING — {len(UNIVERSE)} symbols, {START} to {END}", flush=True)
print(f"CPU cap  : {CPU_CAP} / 8 threads", flush=True)
print(f"Output   : {OUT_DIR}", flush=True)
print(flush=True)
print("FEATURE HONESTY STAMP", flush=True)
print("  Price/volume (REAL, 8 features):", flush=True)
print("    return_1d, return_5d, vol_20d, volume_ratio_20d, atr_20d,", flush=True)
print("    vwap_deviation, intraday_momentum, opening_range_position", flush=True)
print("  F&O (ALL ZERO — fo_daily_metrics table empty, not yet downloaded):", flush=True)
print("    pcr_oi, pcr_volume, net_oi_change, max_pain_distance, spot_futures_basis", flush=True)
print("  Sentiment (ALL ZERO — news table empty, no historical feed):", flush=True)
print("    news_count_1d, news_count_5d, news_sentiment", flush=True)
print(flush=True)
print("THIS IS A PRICE/VOLUME-ONLY MODEL.", flush=True)
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

# ── Stamp model_type in provenance ───────────────────────────────────────────
prov_path = Path(OUT_DIR) / "provenance.json"
with open(prov_path) as f:
    prov_full = json.load(f)

prov_full["model_type"] = "PRICE_VOLUME_ONLY"
prov_full["feature_honesty"] = {
    "price_volume_features": "REAL — 8 features from Dhan intraday OHLCV",
    "fo_features": "ALL_ZERO — fo_daily_metrics table empty (not downloaded)",
    "sentiment_features": "ALL_ZERO — news table empty (no historical news feed)",
    "warning": (
        "Results reflect price/volume information only. "
        "Do not compare against a model that uses F&O or sentiment data."
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

print(f"\nModel files: {Path(OUT_DIR).resolve()}")
print("Next step  : python -m scripts.run_evaluation_v2")
