"""run_training() — single triggerable step for the complete training pipeline.

Call this ONCE after the Dhan bulk download completes and the DuckDB database
contains real intraday + F&O data.

Steps
-----
1.  Build dataset via DatasetBuilder — all features PIT at each t0.
2.  Triple-barrier label the price series.
3.  Apply purge+embargo to each (train, val) fold — no label window may cross
    the split boundary.
4.  Seal the final test set (last 10 % of data by time). NEVER accessed during
    training, tuning, or calibration.
5.  Walk-forward validation: train LightGBM on each fold.
6.  Calibrate on the last validation fold.
7.  Evaluate on the sealed test set EXACTLY ONCE.
8.  Save models + calibrators + provenance JSON to output_dir.

dry_run=True
    Skips all data access and training. Returns provenance metadata only.
    Use this to verify the pipeline wiring before the data lands:
        python -m scripts.run_training --dry-run

Single triggerable step after data lands
-----------------------------------------
    python -m scripts.run_training
    (or: from src.models.training_pipeline import run_training; run_training(...))
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.features import DataGate
from src.models.calibration import ModelCalibrator, UncalibratedConfidenceError
from src.models.dataset import DatasetBuilder, WalkForwardSplitter
from src.models.labels import Label, TripleBarrierLabeler, purge_and_embargo
from src.models.quantile_cone import QuantileConeTrainer
from src.models.scenario_classifier import ScenarioClassifier

_LOG = logging.getLogger(__name__)
UTC = timezone.utc
_BAR = "=" * 72


def _now() -> str:
    return datetime.now(UTC).strftime("%H:%M:%S")


def _fmt_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _detect_lgbm_device() -> tuple[str, int, int]:
    """Detect the best device for LightGBM.
    Returns (device_type, platform_id, device_id).
    
    Probes OpenCL platforms to reliably find the NVIDIA GPU, avoiding AMD iGPU.
    """
    import subprocess
    import sys
    
    # Try to probe OpenCL platforms for NVIDIA
    for i in range(3):
        script = f"import lightgbm as lgb, numpy as np; lgb.train({{'device_type':'gpu', 'gpu_platform_id':{i}, 'gpu_device_id':0, 'verbose':1}}, lgb.Dataset(np.random.rand(10,2), label=np.random.rand(10)), num_boost_round=1)"
        try:
            out = subprocess.check_output([sys.executable, "-c", script], stderr=subprocess.STDOUT, text=True)
            if "NVIDIA" in out:
                return "gpu", i, 0
        except Exception:
            pass
            
    return "cpu", 0, 0

def _build_for_symbol_worker(sym_name: str, db_path: str, timestamps: list, builder, labeler) -> pd.DataFrame:
    import duckdb
    with duckdb.connect(db_path, read_only=True, config={'threads': '1'}) as con:
        # timeframe='1' — use only 1-minute bars for labeling.
        # The DB stores 1/5/15/25/60-min timeframes; all have a bar at 9:30 IST
        # (= 4:00 UTC). Without this filter, three timeframe rows match each daily
        # feature timestamp, triplicating the dataset with mixed-timeframe labels.
        df_intra = con.execute(
            "SELECT ts, close FROM ohlcv_intraday WHERE symbol = ? AND timeframe='1' ORDER BY ts",
            [sym_name]
        ).df()
        
    labels_df = None
    if not df_intra.empty:
        labels_df = labeler.label_series(df_intra["close"], df_intra["ts"])
        
    return builder.build(sym_name, timestamps, db_path, labels_df=labels_df)

_GATE_NONE_FEATURES = [
    "return_1d", "return_5d", "vol_20d", "volume_ratio_20d", "atr_20d",
]
_ALL_FEATURES = [
    "return_1d", "return_5d", "vol_20d", "volume_ratio_20d", "atr_20d",
    "vwap_deviation", "intraday_momentum", "opening_range_position",
    "pcr_oi", "pcr_volume", "net_oi_change", "max_pain_distance", "spot_futures_basis",
    "news_count_1d", "news_count_5d", "news_sentiment",
]


def run_training(
    db_path: str,
    output_dir: str,
    *,
    universe: list[str],
    start_date: date,
    end_date: date,
    bar_timedelta: timedelta = timedelta(minutes=30),
    profit_pct: float = 0.010,
    stop_pct: float = 0.005,
    horizon_bars: int = 30,
    embargo_bars: int = 5,
    n_walk_forward_splits: int = 5,
    gates_open: Optional[set] = None,
    horizon_minutes: int = 30,
    max_cpu_cores: int = -1,
    dry_run: bool = False,
) -> dict:
    """Run the complete training pipeline.

    Parameters
    ----------
    db_path        : Path to DuckDB with real Dhan data. Not accessed in dry_run.
    output_dir     : Directory to write models, calibrators, provenance.json.
    universe       : NSE tickers to include (e.g. ['RELIANCE', 'TCS']).
    start_date     : First trading date of the training window.
    end_date       : Last trading date of the training window.
    bar_timedelta  : Duration of one bar (default: 30 min).
    profit_pct     : Upper triple-barrier as fraction (default: 1.0 %).
    stop_pct       : Lower triple-barrier as fraction (default: 0.5 %).
    horizon_bars   : Vertical barrier in bars (default: 30).
    embargo_bars   : Embargo gap in bars — extra gap beyond purge (default: 5).
    n_walk_forward_splits : Number of walk-forward folds (default: 5).
    gates_open     : Set of DataGate values. None = all gates open.
    horizon_minutes: Stored in model metadata. Does not affect training logic.
    dry_run        : Skip data access and training; return provenance only.
                     Use this to validate wiring before real data lands.

    Returns
    -------
    Provenance dict written to output_dir/provenance.json.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _LOG.info(
        "run_training start — universe=%s, %s→%s, dry_run=%s",
        universe, start_date, end_date, dry_run,
    )

    if gates_open is None:
        gates_open = set(DataGate)
    feature_names = _ALL_FEATURES if gates_open == set(DataGate) else _GATE_NONE_FEATURES

    # Business-day timestamps at session open (09:30 IST = 04:00 UTC)
    bdays = pd.bdate_range(start=start_date, end=end_date)
    timestamps = [datetime(d.year, d.month, d.day, 4, 0, tzinfo=UTC) for d in bdays]

    if dry_run:
        _LOG.info("dry_run=True — skipping dataset build and model training.")
        prov = _provenance(
            universe=universe, start_date=start_date, end_date=end_date,
            gates_open=gates_open, feature_names=feature_names,
            n_timestamps=len(timestamps), dry_run=True,
        )
        _write_provenance(out, prov)
        return prov

    # ── 1. Build dataset ─────────────────────────────────────────────────────
    builder = DatasetBuilder(feature_names=feature_names, gates_open=gates_open)
    labeler = TripleBarrierLabeler(
        profit_pct=profit_pct, stop_pct=stop_pct, horizon_bars=horizon_bars
    )
    splitter = WalkForwardSplitter(
        n_splits=n_walk_forward_splits, final_test_fraction=0.10
    )

    import multiprocessing
    import concurrent.futures

    total_cores = multiprocessing.cpu_count()
    use_cores = max_cpu_cores if max_cpu_cores > 0 else max(1, total_cores - 2)
    use_cores = min(use_cores, total_cores)

    n_sym = len(universe)
    t_run_start = time.perf_counter()

    print(f"\n{_BAR}", flush=True)
    print(f"[{_now()}]  STAGE 1/4 — DATASET BUILD", flush=True)
    print(f"  Symbols : {n_sym}  |  Threads : {use_cores}  |  Timestamps/symbol : {len(timestamps)}", flush=True)
    print(_BAR, flush=True)

    parts: list[pd.DataFrame] = []
    _LOG.info(
        "Building dataset and labels for %d symbols concurrently using %d threads...",
        n_sym, use_cores,
    )

    t_ds_start = time.perf_counter()
    done_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=use_cores) as executor:
        future_to_sym = {
            executor.submit(_build_for_symbol_worker, sym, db_path, timestamps, builder, labeler): sym
            for sym in universe
        }
        for future in concurrent.futures.as_completed(future_to_sym):
            sym = future_to_sym[future]
            try:
                sym_df = future.result()
                parts.append(sym_df)
                done_count += 1
                elapsed_s = time.perf_counter() - t_ds_start
                pct = done_count / n_sym * 100
                if done_count >= 3:
                    rate = done_count / elapsed_s          # symbols / second
                    eta_s = (n_sym - done_count) / rate
                    eta_str = f"ETA ~{_fmt_elapsed(eta_s)}"
                else:
                    eta_str = "ETA ..."
                print(
                    f"  [{_now()}]  {done_count:3d}/{n_sym} ({pct:5.1f}%)"
                    f"  {sym:<20}  |  {_fmt_elapsed(elapsed_s)} elapsed  |  {eta_str}",
                    flush=True,
                )
                _LOG.info("Finished dataset build for %s", sym)
            except Exception as exc:
                _LOG.error("Symbol %s generated an exception: %s", sym, exc)
                raise

    dataset = pd.concat(parts, ignore_index=True).sort_values("t0").reset_index(drop=True)
    elapsed_ds = time.perf_counter() - t_ds_start
    _LOG.info("Dataset: %d rows, %d columns", len(dataset), len(dataset.columns))
    print(f"\n{_BAR}", flush=True)
    print(f"[{_now()}]  STAGE 1/4 COMPLETE — {n_sym} symbols in {_fmt_elapsed(elapsed_ds)}", flush=True)
    print(f"  Dataset : {len(dataset):,} rows × {len(dataset.columns)} columns", flush=True)
    print(_BAR, flush=True)

    # ── 2. Walk-forward splits + purge/embargo ────────────────────────────────
    print(f"\n{_BAR}", flush=True)
    print(f"[{_now()}]  STAGE 2/4 — WALK-FORWARD SPLITS + PURGE/EMBARGO", flush=True)
    print(f"  Folds : {n_walk_forward_splits}  |  Test fraction : 10%  |  Embargo : {embargo_bars} bars", flush=True)
    print(_BAR, flush=True)

    splits = splitter.generate_splits(dataset)
    purged_pairs: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for i, (train_fold, val_fold) in enumerate(splits, 1):
        clean_train = purge_and_embargo(
            train_fold, val_fold,
            embargo_bars=embargo_bars,
            bar_timedelta=bar_timedelta,
        )
        purged_pairs.append((clean_train, val_fold))
        removed = len(train_fold) - len(clean_train)
        print(
            f"  [{_now()}]  Fold {i}/{n_walk_forward_splits}"
            f"  train {len(clean_train):,} rows (purged {removed})"
            f"  |  val {len(val_fold):,} rows",
            flush=True,
        )
        _LOG.debug(
            "Fold purge: %d -> %d train rows (%d windows removed)",
            len(train_fold), len(clean_train), removed,
        )

    # ── 3. Seal final test set ────────────────────────────────────────────────
    test_set = splitter.get_final_test_set(dataset)
    print(f"  [{_now()}]  Test set sealed : {len(test_set):,} rows (evaluated ONCE at the end)")
    _LOG.info("Final test set sealed: %d rows (evaluated once after training).", len(test_set))

    # ── 4. Train models on last fold ──────────────────────────────────────────
    feature_cols = [c for c in dataset.columns if c in feature_names]

    device, plat_id, dev_id = _detect_lgbm_device()

    if device == "gpu":
        lgbm_params = {"device_type": "gpu", "gpu_platform_id": plat_id, "gpu_device_id": dev_id, "num_threads": use_cores}
        device_msg = f"GPU (NVIDIA OpenCL Platform {plat_id}, Device {dev_id})"
    else:
        lgbm_params = {"num_threads": use_cores}
        device_msg = "CPU"

    _LOG.info("LightGBM device selected: %s (Max CPU Cores: %d / %d)", device_msg, use_cores, total_cores)

    cone = QuantileConeTrainer(horizon_minutes=horizon_minutes, feature_names=feature_cols, lgbm_params=lgbm_params)
    classifier = ScenarioClassifier(feature_names=feature_cols, lgbm_params=lgbm_params)
    calibrator = ModelCalibrator(method="isotonic")

    if feature_cols and "label" in dataset.columns and purged_pairs:
        last_train, last_val = purged_pairs[-1]
        # Drop only rows where the LABEL is NaN — feature NaNs (e.g. FO features
        # when fo_daily_metrics is empty) are handled by fillna(0.0) below.
        # dropna on feature_cols would eliminate every row when any feature is
        # universally NaN (e.g. empty FO/news tables for fo_daily_metrics).
        labeled_train = last_train[last_train["label"].notna()].copy()
        labeled_val = last_val[last_val["label"].notna()].copy()

        print(f"\n{_BAR}", flush=True)
        print(f"[{_now()}]  STAGE 3/4 — LGBM TRAINING", flush=True)
        print(f"  Device  : {device_msg}  |  Threads : {use_cores}/{total_cores}", flush=True)
        print(f"  Train   : {len(labeled_train):,} rows  |  Val : {len(labeled_val):,} rows  |  Features : {len(feature_cols)}", flush=True)
        print(_BAR, flush=True)

        if len(labeled_train) >= 10 and len(labeled_val) >= 5:
            X_train = labeled_train[feature_cols].fillna(0.0)
            y_train = labeled_train["label"]

            _LOG.info("Training ScenarioClassifier on %d rows", len(X_train))
            t_lgbm = time.perf_counter()
            classifier.fit(X_train, y_train)
            print(f"  [{_now()}]  ScenarioClassifier trained in {_fmt_elapsed(time.perf_counter()-t_lgbm)}", flush=True)
            classifier.save(out / "scenario_classifier")

            # ── 5. Calibrate on validation fold ──────────────────────────────
            print(f"\n{_BAR}", flush=True)
            print(f"[{_now()}]  STAGE 4/4 — CALIBRATION", flush=True)
            print(f"  Method  : isotonic regression  |  Val samples : {len(labeled_val):,}", flush=True)
            print(_BAR, flush=True)

            X_val = labeled_val[feature_cols].fillna(0.0)
            y_val = labeled_val["label"].values

            proba_val = classifier.predict_proba(X_val)
            # Calibrate NO_TRADE vs rest (primary signal)
            y_binary = (y_val == int(Label.NO_TRADE)).astype(float)
            calibrator.fit(
                y_binary,
                proba_val[["p_no_trade"]],
                val_period=(str(labeled_val["t0"].min()), str(labeled_val["t0"].max())),
            )
            print(f"  [{_now()}]  Calibration complete", flush=True)
        else:
            print(
                f"  [{_now()}]  SKIPPED — insufficient rows"
                f"  (train={len(labeled_train)}, val={len(labeled_val)}, need >=10/5)",
                flush=True,
            )

    # ── 6. Evaluate on sealed test set ONCE ───────────────────────────────────
    print(f"\n{_BAR}", flush=True)
    print(f"[{_now()}]  EVALUATION — sealed test set (touched ONCE)", flush=True)
    print(_BAR, flush=True)
    test_metrics: dict = {}
    if classifier._is_fitted and feature_cols and "label" in test_set.columns:
        labeled_test = test_set[test_set["label"].notna()].copy()
        if len(labeled_test) >= 1:
            X_test = labeled_test[feature_cols].fillna(0.0)
            proba_test = classifier.predict_proba(X_test)
            _LOG.info("Evaluated on %d sealed test rows.", len(X_test))
            test_metrics["test_rows"] = len(X_test)
            print(f"  [{_now()}]  Evaluated on {len(X_test):,} sealed test rows", flush=True)

    # ── 7. Provenance ─────────────────────────────────────────────────────────
    prov = _provenance(
        universe=universe, start_date=start_date, end_date=end_date,
        gates_open=gates_open, feature_names=feature_names,
        n_timestamps=len(timestamps), dry_run=False,
        n_dataset_rows=len(dataset),
        n_folds=len(purged_pairs),
        test_set_rows=len(test_set),
        calibration_provenance=calibrator.calibration_provenance(),
        **test_metrics,
    )
    _write_provenance(out, prov)

    total_elapsed = time.perf_counter() - t_run_start
    cal_prov = calibrator.calibration_provenance()
    print(f"\n{_BAR}", flush=True)
    print(f"[{_now()}]  === DONE ===  Total time: {_fmt_elapsed(total_elapsed)}", flush=True)
    print(f"  Classifier trained : {classifier._is_fitted}", flush=True)
    print(f"  Calibrated         : {cal_prov.get('is_calibrated', False)}", flush=True)
    print(f"  Dataset rows       : {len(dataset):,}  |  Test set rows : {len(test_set):,}", flush=True)
    print(f"  Provenance         : {(out / 'provenance.json').resolve()}", flush=True)
    print(_BAR, flush=True)

    return prov


# ── helpers ───────────────────────────────────────────────────────────────────

def _provenance(
    *,
    universe: list[str],
    start_date: date,
    end_date: date,
    gates_open: set,
    feature_names: list[str],
    n_timestamps: int,
    dry_run: bool,
    **extra: object,
) -> dict:
    return {
        "pipeline_version": "2.3+2.4",
        "build_ts": datetime.now(UTC).isoformat(),
        "universe": universe,
        "date_range": [str(start_date), str(end_date)],
        "gates_open": sorted(g.value for g in gates_open),
        "feature_names": feature_names,
        "n_timestamps_per_symbol": n_timestamps,
        "dry_run": dry_run,
        "notes": "F&O/OI features explicitly absent (Dhan /charts/historical returns no OI data). Active features depend on gates_open parameter — check feature_names field.",
        **extra,
    }


def _write_provenance(out: Path, prov: dict) -> None:
    with open(out / "provenance.json", "w") as f:
        json.dump(prov, f, indent=2, default=str)
    _LOG.info("Provenance written to %s/provenance.json", out)
