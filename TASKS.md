# TASKS.md — Phase / Subtask Checklist

## Phase 0 — Foundations & Backtester Scaffold
- [x] Repository directory structure
- [x] `pyproject.toml` (package discovery, pytest, ruff, black config)
- [x] `requirements.txt` (lower-bound pins; freeze to exact after first install)
- [x] `config/config.toml` — TOML config with NSE F&O universe, horizons, guardrails
- [x] `src/config.py` — pydantic v2 config loader via stdlib `tomllib`
- [x] `src/log.py` — structured logging (console + rotating file)
- [x] `src/db/schema.py` — DuckDB + SQLite schema SQL (all `IF NOT EXISTS`)
- [x] `src/db/duckdb_client.py` — DuckDB context-manager wrapper
- [x] `src/db/sqlite_client.py` — SQLite context-manager wrapper
- [x] `src/data/base.py` — abstract `DataSource` interface + `DataLeakageError` + `_enforce_pit`
- [x] `src/data/historical.py` — `StubHistoricalSource` (empty DataFrames, correct schema)
- [x] `src/experiment/tracker.py` — `ExperimentTracker` SQLite runs table
- [x] `tests/conftest.py` — shared pytest fixtures
- [x] `tests/test_config.py` — config loading & validation
- [x] `tests/test_schema.py` — DuckDB + SQLite schema idempotency
- [x] `tests/test_pit_enforcement.py` — PIT leakage enforcement (future rows blocked)
- [x] `tests/test_data_interface.py` — DataSource contract + stub adapter
- [x] `tests/test_experiment.py` — run log write/read round-trip
- [x] All pytest tests green

## Phase 1 — Data Pipeline
- [x] NSE trading calendar (`src/data/calendar.py`) — `is_trading_day`, `is_market_open`,
      `next/prev_trading_day`, `trading_days_in_range`, `classify_missing_file`;
      `config/nse_holidays.json` cache with embedded fallback; `refresh_nse_holidays()`
      degrades gracefully if NSE endpoint unreachable; 39 tests green
- [x] NSE bhavcopy downloader + DuckDB loader + PIT symbol-eligibility:
      `download_bhavcopy` (calendar-aware gap classification; retry+backoff; MissingDataGap on
      trading-day failure); `parse_bhavcopy_csv` (newer+older NSE formats; EQ filter);
      `validate_bhavcopy` (hard-reject: negative price, high<low; soft-flag: zero vol,
      extreme move; every decision logged; never silent drop);
      `load_bhavcopy_to_duckdb` (idempotent ON CONFLICT DO NOTHING; provenance
      `source=nse_bhavcopy`; `ingested_at` timestamp); `get_eligible_symbols(as_of, db)`
      reconstructs per-date universe from actual bhavcopy data — survivorship-documented.
      `BhavCopySource(DataSource)` raises `DataNotAvailableError` for all non-daily methods.
      `DataNotAvailableError` added to `src/data/base.py`. 37 new tests; 149 total green.
- [x] Corporate-action adjustment layer (Checkpoint 1.5.1):
      `parse_action_description` (SPLIT/BONUS/DIVIDEND/OTHER; wrong factor → OTHER, never silent);
      `CorporateActionFetcher` (NSE API best-effort, `_mock_response` hook, graceful failure);
      `load_corporate_actions_csv` (reliable CSV fallback, idempotent upsert);
      `compute_adjustment_factors` (backward cumulative product, PIT-correct: future splits excluded);
      `AdjustedBhavCopySource` (`BACKTEST_ONLY=True` structural discriminator,
      `data_source_name="nse_bhavcopy_adjusted"`, backward anchor on most-recent bar,
      volume not adjusted, `_enforce_pit` on all paths); `corporate_actions` table in schema;
      33 tests — headline test proves returns continuous across split, anchor invariant,
      BACKTEST_ONLY firewall, PIT-correct future-split exclusion; 312 / 312 total green
- [x] Upstox WebSocket tick adapter (`src/data/live_upstox.py`, behind `DataSource` interface —
      **no** prediction code touches Upstox SDK directly; interface-symmetry proof test);
      25 tests green
- [x] Consolidation + integration test pass (Phase 0 + Checkpoints 1–3): 6 risk areas probed,
      3 bugs surfaced and fixed, 46 new tests added; 195 / 195 total green
- [x] Option chain fetcher (`src/data/option_chain.py`): `OptionChainSource(DataSource)`
      with in-memory TTL cache (60 s), exponential backoff on HTTP errors, last-good
      fallback on failure (ts column records original fetch time so callers detect
      staleness), PIT on `ts`, no SDK import, `_mock_fetch` + `_fetch_ts_override` test hooks
- [x] RSS news ingest + dedup + first-knowable-time stamping + ingestion-delay buffer
      (`src/data/news_ingest.py`): `RSSNewsIngest` (deferred feedparser import, dedup via
      SHA-256 of normalized content, word-boundary regex symbol tagging with aliases, untagged
      stories stored with `symbols=[]`, `first_knowable_at = published_at + delay`) +
      `NewsSource(DataSource)` (PIT on `first_knowable_at`, not `published_at`)
- [x] Data Health Monitor (`src/data/health_monitor.py`): `DataHealthMonitor` tracks last-seen
      per feed; `is_feed_healthy` / `confidence_multiplier` (0.0 stale during market hours,
      1.0 otherwise); `check_value` (first obs ok, zero prev ok, large jump → WARNING+False);
      tz-aware `as_of` required (raises `ValueError` for naive); calendar-integrated
- [x] `DataNotAvailableError` on `BhavCopySource.get_ohlcv_minute` updated: message now
      explicitly states "Requires Kite or a supplied intraday dataset — not available from
      nse_bhavcopy; any backtest using BhavCopySource cannot validate intraday edge"
- [x] `NEWS_COLS` updated: added `symbols: "object"` (list of tagged ticker strings)
- [x] `requirements.txt`: feedparser==6.0.12 uncommented and pinned
- [x] All Phase 1 tests green: 262 / 262 (was 195; +67 new tests)

### Phase 1 Review + Consolidation Pass
- [x] End-to-end smoke test: all four sources (BhavCopy, UpstoxLive, OptionChain, NewsSource)
      exercised together in one test with consistent as_of; PIT verified on all paths
- [x] News tagging coverage audit: 99% on company-specific headlines (69/70), 100% correct
      on macro stories (no false positives on market-wide headlines); 1 known FP (SBI/SBILIFE)
- [x] Bug fix: `UpstoxLiveSource.get_ticks` — two early exits bypassed `_enforce_pit`
      (empty snapshot, empty after lookback); now routes through `_enforce_pit` on all paths
- [x] Bug fix: `NewsSource.get_news` — empty-result early exit bypassed `_enforce_pit`;
      now routes through `_enforce_pit` on all paths
- [x] Bug fix: HCLTECH alias "HCL" added (standalone "HCL" was not matched)
- [x] Known limitation documented: "SBI" in "SBI Life" stories also tags SBIN (comment in
      SYMBOL_ALIASES; tested and noted in test_tagging_sbilife_vs_sbin_note)
- [x] Cross-cutting invariant audit (test_review_pass.py, 17 tests): SDK quarantine verified
      across all src/ files; provenance fields on all sources; DataSource subclass check;
      no-GPU-import audit; timezone consistency; empty-DF PIT compliance
- [x] **Phase 1 COMPLETE: 279 / 279 tests passing**

### Phase 1.5, Checkpoint 1.5.3 — Alias Expansion + Disambiguation
- [x] Expanded `SYMBOL_ALIASES` from 49 to 116 tickers (all F&O universe — 50 Nifty 50 + 66 additional)
- [x] Added missing BAJAJ-AUTO (was absent from Nifty 50 coverage); HDFCLIFE upgraded to
      "HDFC Life Insurance" / "HDFC Life" (no bare "HDFC"); M&M gains standalone "Mahindra" alias;
      EICHERMOT gains "Royal Enfield"; TATACONSUM upgraded to "Tata Consumer Products"
- [x] Replaced `_tag_symbols` with span-based disambiguation using `pat.finditer()`:
      if ticker A's match span is strictly contained within ticker B's longer span,
      ticker A is suppressed — more specific match wins;
      "SBI Life" → SBILIFE only (SBIN's 'SBI' span subsumed); Bajaj family never cross-tags;
      "Tech Mahindra" → TECHM only (M&M's 'Mahindra' span subsumed);
      "Kotak Mahindra Bank" → KOTAKBANK only (M&M's 'Mahindra' subsumed)
- [x] Disambiguation log messages: DEBUG for suppressed, untagged, multi-tagged
- [x] Test suite updated: `_BROAD_TAGGING_CASES` replaces `_TAGGING_CASES` (131 cases:
      116 company + 15 macro); coverage floor raised 85% → 90%; actual coverage: **100%**
      (121/121 company cases pass; 0 false positives on macro)
- [x] `test_tagging_sbilife_vs_sbin_note` removed (FP was limitation, now fixed);
      replaced with `test_tagging_sbi_disambiguation_fixed` (asserts SBIN NOT tagged on SBI Life)
- [x] Added: `test_tagging_sbi_standalone_still_tags_sbin`, `test_tagging_bajaj_family_disambiguation`,
      `test_tagging_tech_mahindra_vs_mahindra`
- [x] `test_tagging_all_known_symbols_tagged` passes all 116 tickers across all aliases
- [x] Word-boundary guard still holds after expansion ("pitch"/"switch" ≠ ITC)
- [x] **315 / 315 tests passing**

### Phase 1.5, Checkpoint 1.5.4 — F&O Historical Bhavcopy (DEFERRED — superseded by Dhan)
Assessment verdict (2026-06-26): BUILD.  Revised verdict (2026-06-27): DEFER.
Dhan Data API provides 5yr F&O/OI history directly as part of the bulk historical download
(Checkpoint 1.6.1 below).  NSE F&O bhavcopy remains available free if pre-2020 F&O depth
is ever required.  Do not build until Dhan coverage dates are confirmed and a gap exists.

---

## Phase 1.6 — Dhan Data Integration

**Context:** Dhan Data API (Rs 499/month, 30-day window) provides one-time bulk historical
download of ~5yr intraday minute bars (1/5/15/25/60 min) + F&O/OI history for the ~160-symbol
F&O universe.  Once stored to DuckDB the data is permanently ours — subscription lapsing does
not delete stored data.  After 30 days, live feed switches to Upstox free tier with NO code
changes (adapter swap only).  All credentials from `.env`/config; nothing hard-coded.

### Cross-machine portability check (pre-build, no code — 2026-06-27 assessment)
- [x] `.gitignore` already excludes `.env`, `.venv/`, `data/*.duckdb`, `data/*.db` — clean
- [x] All config paths relative to project root — clean
- [x] No machine-specific hard-coded paths in any `src/` file — clean
- [x] Add `models/`, `*.pkl`, `*.lgb`, `*.orchestration_state.json` to `.gitignore` (model
      files live on RTX 3060 machine, not in git)
- [x] CUDA detection: `_detect_lgbm_device()` in `training_pipeline.py` — tries torch.cuda,
      then nvidia-smi, falls back to "cpu"; never hard-coded; RTX 3060 auto-detects GPU

### Checkpoint 1.6.1 — Dhan Bulk Historical Downloader ✅ BUILT (awaiting real key to run)
**Highest priority — 30-day subscription clock threatens this data irreversibly.**

- [x] `src/db/schema.py`: tables added: `ohlcv_intraday`, `fo_history`, `fo_daily_metrics`,
      `download_jobs`, `dhan_instruments` — all idempotent `CREATE TABLE IF NOT EXISTS`
- [x] `src/data/dhan_downloader.py`: `DhanInstrumentMaster` (ticker→security_id lookup,
      mock-injectable, CSV refresh from Dhan public endpoint) + `DhanBulkDownloader`
      (paced, resumable; `_build_chunks` → 90-day windows; `_ensure_jobs` → insert PENDING
      rows; `_load_pending_jobs` → pending+failed only; `_process_job` → fetch→upsert→mark;
      `run_fo` → fo_history; `compute_fo_metrics` → fo_daily_metrics with PCR+max-pain;
      `coverage_report` → COMPLETE/IN_PROGRESS/HAS_ERRORS/NOT_STARTED matrix)
- [x] `scripts/run_bulk_download.py` — CLI wrapper: run(), run_fo(), compute_fo_metrics(),
      coverage_report(); argparse for --symbols, --timeframes, --start, --end, --fo-only,
      --intraday-only, --coverage-only
- [x] `tests/test_dhan_downloader.py` — 30 tests: schema idempotency; chunk arithmetic
      (single chunk, split boundary, no-overlap, 5-year count); happy path; idempotency;
      **resume-after-interruption** (2 chunks pre-DONE, 1 PENDING → 1 fetch call);
      failed chunk retry; persistent failure → FAILED status; empty response → PARTIAL;
      multiple symbols independent; provenance (dhan_historical, UTC ingested_at);
      coverage report (COMPLETE/HAS_ERRORS/columns); F&O happy path + idempotency;
      fo_metrics PCR correct; instrument master mock lookup + missing key

### Checkpoint 1.6.2 — Dhan Data Adapter ✅ BUILT
- [x] `src/data/dhan_adapter.py`: `DhanDataSource(DataSource)` + `DhanFOMetricsSource(DataSource)`:
      — SDK quarantine: only `dhan_adapter.py` imports `dhanhq`; deferred to `connect()`
      — `DhanDataSource`: historical methods (`get_ohlcv_daily`, `get_ohlcv_minute`,
        `get_ohlcv_intraday`) query `ohlcv_intraday` DuckDB table; `_enforce_pit` on ts;
        live methods: `get_ticks` (in-memory WebSocket buffer, `_inject_ws` test hook);
        `get_option_chain` (Dhan REST, TTL cache + last-good fallback, `_mock_fetch` hook);
        `get_news` raises `DataNotAvailableError`; `BACKTEST_ONLY=False`
      — `DhanFOMetricsSource`: `BACKTEST_ONLY=True`; queries `fo_daily_metrics`
      — `DHAN_ADJUSTMENT_STATUS` env var enforcement; WARNING when unset (TODO for key arrival)
- [x] `tests/test_dhan_adapter.py` — 31 tests: DataSource subclass ×2; SDK quarantine;
      no-GPU-imports; connect() without SDK raises ImportError; get_ohlcv_daily (empty DB,
      schema, PIT block, provenance, warns-when-unknown); get_ohlcv_minute (schema, PIT
      enforce, 5min timeframe, invalid TF); get_ticks (empty buffer, inject, schema);
      get_option_chain (mock, cache hit, last-good fallback, no-fallback raises); get_news
      raises; DhanFOMetricsSource (empty, schema, PIT block, all standard methods raise)

### Checkpoint 1.6.3 — Live Source Seam / Config-Driven Factory ✅ BUILT
- [x] `config/config.toml`: `[data] live_data_source = "dhan"` key set; switching to
      `"upstox"` + wiring the right `.env` credentials is the ONLY change to swap live sources
- [x] `src/data/live_source_factory.py`: `get_live_source(config, db_path=None) -> DataSource`;
      `_SUPPORTED = {"dhan", "upstox"}`; deferred imports (factory importable without any SDK);
      raises `ValueError` for unknown source name; `db_path` kwarg overrides config path
- [x] `tests/test_live_source_factory.py` — 8 tests: factory returns `DhanDataSource` for
      `"dhan"`; `UpstoxLiveSource` for `"upstox"`; both satisfy `DataSource`; ValueError for
      unknown; importable without SDKs; db_path override; missing attr defaults to `"dhan"`

### Checkpoint 1.6.4 — Dhan Adjustment Assessment + Redundancy Cleanup (on key arrival) ✅ DONE
- [x] **Confirmed: data is ADJUSTED (server-side).** Checked 1-year RELIANCE daily price series:
      min=1258.8, max=1592.3 (1.26× range), no >40% single-day moves, fully continuous.
      Outcome: `DHAN_ADJUSTMENT_STATUS=adjusted` stamped in `.env`.
      `AdjustedBhavCopySource` adjustment layer does not apply to Dhan data — redundant and correct.
      Raw-vs-adjusted firewall remains structurally intact.
- [x] `DHAN_ADJUSTMENT_STATUS` enforcement already wired in `dhan_adapter.py` (Checkpoint 1.6.2)
- [x] Update `DESIGN.md §2` deferred — not a blocking item; RUNBOOK documents the finding

### Checkpoint 1.6.5 — First Real Dhan API Contact + Format Verification ✅ DONE (2026-06-29)
- [x] Created `.env` with real credentials (git-ignored, never committed)
- [x] **`[PASS] Response format OK`**: `python -m scripts.test_fetch` returns 1,125 rows for RELIANCE,
      all 6 columns present (ts, open, high, low, close, volume); first bar 2026-06-22 03:45 UTC
      (= 09:15 IST, correct NSE open); values look clean
- [x] Real API response format confirmed: `{"open":[], "high":[], "low":[], "close":[], "volume":[], "timestamp":[]}`
      — exact match with mocks; no parser changes needed
- [x] Timestamps confirmed: Unix seconds (float); `pd.to_datetime(unit="s", utc=True)` correct
- [x] Adjustment status confirmed (see Checkpoint 1.6.4)
- [x] **Bug fix** (`src/data/dhan_downloader.py`): `DhanInstrumentMaster.refresh()` only updated
      `self._cache` on successful DB INSERT. On `:memory:` DBs, each `DuckDBClient(":memory:")`
      opens a fresh connection — schema created in `_ensure_schema()` was invisible to `refresh()`.
      Fix: always update cache before attempting DB insert; insert is persistence-only, cache is primary.
- [x] **F&O OI gap confirmed**: `/charts/historical` for `FUTSTK` returns OHLCV only — no open interest.
      `pcr_oi`, `net_oi_change`, `max_pain_distance` cannot be computed from this endpoint.
      `--skip-fo` flag added throughout pipeline; documented in RUNBOOK.md §F&O Known Limitation.
- [x] **RELIANCE security_id = 2885** (equity); near-month FUTSTK sec_id = 62802 (Jun 30 expiry)
- [x] Preflight check passed on dev laptop: 117 GB free, all env vars present, DB path writable
- [x] Full test suite: **514 / 514 passing** after all real-API fixes
  _(As of 2026-07-10: 509/514 — 5 tests broken by 3060 changes, fixed in consolidation pass; see CHANGELOG)_

## Phase 2 — Intelligence (CPU)
Phase 2 plan posted 2026-06-26; seed weights added 2026-06-27.
Build order: 2.1 and 2.2 are independent (can parallel); 2.3 depends on 2.1+2.2;
2.4–2.6 sequential.  All CPU-only.  No broker credentials.
**Pre-requisite: Phase 1.6 complete (Dhan DuckDB populated with 5yr minute bars).**

### Phase 2 pre-build: Add seed weights to config (before any 2.x checkpoint)
- [x] `config/config.toml` — add `[model.seeds]` section:
      ```
      [model.seeds]
      # Research-backed initial weights for NSE intraday F&O universe.
      # Weights adjusted ONLY via gated process — never hand-tuned against the backtest.
      market_regime    = 0.25   # Nifty/VIX regime — dominant NSE intraday factor
      fo_signals       = 0.25   # PCR, OI, max-pain, basis — India-specific documented edge
      price_momentum   = 0.20   # Momentum/microstructure — decays fast in liquid stocks
      sentiment        = 0.15   # FinBERT — noisy but non-zero in mid-caps
      institutional    = 0.10   # FII/DII EOD + sector RS — slow-moving context
      calendar         = 0.05   # Open-auction effect + time-of-day bucket

      # Gating: updates accepted only if ALL three gates pass
      min_observations_per_label    = 50      # per class label in live validation set
      max_weight_change_per_update  = 0.05    # no group moves >5 % in one weekly cycle
      validation_gate_min_brier_improvement = 0.005  # must beat prior by ≥ 0.5 % Brier
      shrinkage_factor              = 0.5     # pull toward seed when n_obs < minimum
      ```
- [x] Cold-start definition documented in DESIGN.md: day-one system uses seed weights
      for all groups until each group accumulates `min_observations_per_label` samples;
      the shrinkage_factor blends (seed × 0.5 + data-estimated × 0.5) during ramp-up
- [x] Seeds are set ONCE here and never re-edited against backtest results; any change
      requires the validation gate to pass on held-out data first

### Checkpoint 2.1 — Feature Engineering Foundation ✅ BUILT
- [x] `src/features/base.py`: `DataGate` enum (NONE/INTRADAY/FO/NEWS/NEWS_NLP);
      `FeatureMeta` (frozen dataclass: name, gate, description, dead);
      `FeatureRegistry` (class-level dict; `register()` decorator; `compute_all()`;
      gate filtering: NONE always runs; gated features return None when gate closed)
- [x] `src/features/price_volume.py` — 8 features:
      daily (gate=NONE): `return_1d`, `return_5d`, `vol_20d`, `volume_ratio_20d`, `atr_20d`;
      intraday (gate=INTRADAY): `vwap_deviation`, `intraday_momentum`, `opening_range_position`
      [INTRADAY gated — needs Dhan download; tested on synthetic data now]
- [x] `src/features/fo.py` — 5 features (gate=FO):
      `pcr_oi`, `pcr_volume`, `net_oi_change`, `max_pain_distance`, `spot_futures_basis`
      [FO gated — needs Dhan download; tested on synthetic data now]
- [x] `src/features/news_count.py` — 2 features (gate=NEWS):
      `news_count_1d`, `news_count_5d`; DuckDB `list_contains()` on `symbols VARCHAR[]`
- [x] `src/features/__init__.py`: public API; side-effect imports to populate registry
- [x] `tests/test_features.py` — 42 tests:
      Per-feature PIT test (poison row at boundary ×15); correctness tests; registry gate flags;
      compute_all gate filtering; compute_features convenience wrapper;
      no-GPU-imports invariant extended to cover src/features/

### Checkpoint 2.2 — FinBERT Sentiment Scorer ✅ BUILT
- [x] `src/features/sentiment.py`: `FinBERTScorer` + `aggregate_sentiment` + `news_sentiment`:
      — `FinBERTScorer`: lazy import of transformers+torch; device at runtime via
        `torch.cuda.is_available()`, never hard-coded; batch scoring (`score_batch()`);
        score cache keyed on `news_id` (articles never re-scored); `_mock_pipeline` hook;
        graceful degradation (ImportError or load failure → WARNING + 0.0 neutral)
      — `aggregate_sentiment(scored_df, as_of)`: PIT filter `first_knowable_at < as_of`
        (belt-and-suspenders re-check); exp decay half-life 4h; source-weighted
        (sebi=2.5, nse_filings=bse_filings=2.0, reuters=1.5, default=1.0)
      — `news_sentiment` feature: gate=NEWS_NLP; returns 0.0 (not None) when no news;
        `SENTIMENT_GATE_PROVENANCE = "forward_only"` stamped in description
- [x] `tests/test_sentiment.py` — 23 tests:
      device-cpu-no-cuda; device-cpu-torch-missing (monkeypatch sys.modules);
      device-cuda-available (fake torch module); graceful-no-pipeline=0.0;
      score_batch positive/negative/neutral/multi; caches-articles (pipeline called once);
      score_news_df empty; sample-output-format; aggregate-sentiment PIT
      (boundary story excluded ✓); future excluded; no-news=0.0; time-decay;
      source-weighting; news_sentiment gate/empty-DB/with-news/future-excluded;
      provenance constant; provenance in description; never-hard-codes-cuda (regex)

### Checkpoint 2.3 — Triple-Barrier Labels + Dataset Builder ✅ BUILT
- [x] `src/models/labels.py`: `Label(IntEnum)` (NO_TRADE=0/BULLISH=1/BEARISH=-1 — all first-class);
      `TripleBarrierLabeler(profit_pct, stop_pct, horizon_bars).label_series(prices, ts)`:
      scans forward bars, records which barrier hits first; NO_TRADE fires when vertical
      barrier (time limit) is reached — flat windows stay out, not forced into a direction
- [x] `purge_and_embargo(train, val, embargo_bars, bar_timedelta)`:
      Purge: drops training samples where t1 >= val_start (label window overlaps val);
      Embargo: additionally drops samples where t0 >= val_start − embargo_bars × bar_timedelta;
      structural guarantee: after calling, zero training t1s cross the split boundary
- [x] `src/models/dataset.py`: `DatasetBuilder.build(symbol, timestamps, db_path, labels_df)`:
      assembles one PIT row per timestamp; features via FeatureRegistry.compute_all() at t0;
      provenance columns gate_intraday_open, gate_fo_open, gate_news_open, gate_news_nlp_open,
      forward_only_features stamped on every row
- [x] `WalkForwardSplitter`: time-ordered folds; final test set sealed (10 %) and EXCLUDED
      from all train/val folds; get_final_test_set() raises RuntimeError on second call
- [x] `tests/test_labels.py` — 17 tests:
      upper/lower/time barrier correctness; NO_TRADE first-class invariant;
      **purge removes overlapping label windows** (KEY); **embargo removes near-boundary samples** (KEY);
      structural guarantee: zero t1 crosses boundary after purge+embargo;
      **full-pipeline PIT leakage test** (poison row AT as_of excluded end-to-end) (KEY);
      gate-flag provenance; walk-forward time-ordering; test-set sealed exactly once

### Checkpoint 2.4 — LightGBM Quantile Cone + Scenario Classifier + Calibration ✅ BUILT (framework)
TRAINING and CALIBRATION runs deferred until Dhan data lands. Framework fully wired and
tested on synthetic data. Single call to run_training() triggers the full pipeline.

- [x] `src/models/quantile_cone.py`: `QuantileConeTrainer` — 5 quantile regressors
      (q=0.10/0.25/0.50/0.75/0.90); lazy LightGBM import (ImportError with pip hint if missing);
      `_enforce_monotonic(preds)` via np.sort(axis=1); save/load via LightGBM native format +
      meta.json; predict() raises RuntimeError if unfitted
- [x] `src/models/scenario_classifier.py`: `ScenarioClassifier` — LightGBM multiclass;
      NO_TRADE(0)/BULLISH(1)/BEARISH(-1) as first-class labels; predict_proba → p_no_trade,
      p_bullish, p_bearish; rows always sum to 1.0; lazy LightGBM import; save/load
- [x] `src/models/calibration.py`: `ModelCalibrator(method='isotonic'|'platt')` — fits on
      validation slice only; calibrate(require_fitted=True) raises `UncalibratedConfidenceError`
      by default (structural enforcement: no confidence downstream without calibration);
      `reliability_diagram(y_true, y_prob, n_bins=10)` → bin_centers, empirical_freqs, counts,
      brier_score, max_calibration_error; `calibration_provenance()` stamps is_calibrated,
      method, val_period, n_calibration_samples; lazy sklearn import
- [x] `src/models/training_pipeline.py`: `run_training(db_path, output_dir, *, universe,
      start_date, end_date, ..., dry_run=False)` — single triggerable step:
      build dataset → apply purge/embargo → seal test set → walk-forward train →
      calibrate on last val fold → evaluate on test ONCE → write provenance.json;
      `dry_run=True` skips data access and training (schema/provenance only — use this
      to verify wiring before data lands)
- [x] `tests/test_models.py` — 16 tests:
      quantile monotonicity enforcement; unfitted raises; fit+predict shapes (lgbm);
      monotonic after fit (lgbm); scenario unfitted raises; proba sums to 1.0 (lgbm);
      NO_TRADE class present (lgbm); **uncalibrated confidence raises** (KEY structural test);
      require_fitted=False passes raw; **reliability diagram structure** (KEY); brier score test;
      **calibration reduces Brier** (sklearn); provenance stamp; **run_training dry_run** (KEY);
      accepts required params; no hard-coded GPU in models layer

### Checkpoint 2.4.5 — Orchestration Infrastructure ✅ BUILT (2026-06-27)
Entire workflow to go from "I have the Dhan key" to "models saved" is one-command and resumable.
Built against mocks; real API runs tomorrow. All 32 new tests pass.

- [x] `src/orchestration/__init__.py` — module docstring
- [x] `src/orchestration/preflight.py`: `run_preflight(db_path, ...)` — checks DHAN_CLIENT_ID,
      DHAN_ACCESS_TOKEN, DHAN_ADJUSTMENT_STATUS, disk space (≥ 25 GB), DB parent writable,
      optional connection test; `PreflightError` raised on any critical failure; all calls
      injectable via `_disk_usage_fn`, `_connection_test_fn`
- [x] `src/orchestration/orchestrator.py`: `Orchestrator` + `OrchestratorState` + `Stage` enum
      (8 stages: PREFLIGHT, TEST_FETCH, BULK_DOWNLOAD, COVERAGE_CHECK, BUILD_DATASET,
      TRAIN_MODELS, CALIBRATE, SAVE_SUMMARY);
      — `OrchestratorState`: JSON checkpoint at `{output_dir}/.orchestration_state.json`;
        marks stages done; resumes by skipping completed stages;
      — `Orchestrator.run(dry_run, start_from, skip_test_fetch)`: PREFLIGHT failure always
        re-raises; other failures include `--start-from=STAGE` hint; all external calls
        injectable (`_mock_preflight_fn`, `_mock_test_fetch_fn`, `_mock_downloader`,
        `_mock_run_training_fn`)
- [x] `scripts/__init__.py` — module docstring
- [x] `scripts/test_fetch.py`: `test_fetch(symbol, days, timeframe, ...)` — probe Dhan format
      in ≤ 5 days; raises `ValueError` if `days > 5`; validates column names against
      `EXPECTED_COLS`; returns `{"status": "pass"|"partial"|"fail", "rows", "columns",
      "missing_cols", "fix_location", "message"}`; prints fix pointer to
      `_parse_ohlcv_response()` on mismatch; CLI: `python -m scripts.test_fetch`
- [x] `scripts/run_bulk_download.py` — CLI: --symbols, --timeframes, --start, --end,
      --fo-only, --intraday-only, --coverage-only, --verbose
- [x] `scripts/run_orchestration.py` — CLI: --db-path, --output-dir, --symbols, --start,
      --end, --dry-run, --start-from, --skip-test-fetch, --verbose; single command for Step 5
- [x] `RUNBOOK.md` — 5-step runbook matching the user's exact workflow:
      Step 1 (dev laptop: add key, test-fetch), Step 2 (commit+push),
      Step 3 (RTX 3060: clone+setup+add key again),
      Step 4 (RTX 3060: test-fetch to confirm THIS machine connects),
      Step 5 (RTX 3060: full orchestration);
      includes success criteria, failure action table, fix pointer for column mismatch,
      credential security section, resume commands
- [x] `src/models/training_pipeline.py`: added `_detect_lgbm_device()` — tries torch.cuda,
      then nvidia-smi subprocess, falls back to "cpu"; no hard-coded device strings
- [x] `.gitignore`: added `models/`, `*.pkl`, `*.lgb`, `*.orchestration_state.json`
- [x] `tests/test_orchestration.py` — 32 tests:
      preflight: all-good pass, missing client_id, missing token, bad adjustment status,
      unadjusted ok, low disk, connection pass/fail, 5 base checks, error lists failed names;
      test_fetch: days>5 raises, pass with good cols, partial with missing col, fail on empty,
      fail on None, result keys, days=5 boundary ok;
      orchestrator: dry-run completes, stage list 8 entries, completed stages skipped,
      start_from clears later stages, preflight failure blocks, skip_test_fetch flag,
      checkpoint file written, non-preflight failure hints --start-from;
      GPU detection: cpu fallback, string type, no hard-coded 'cuda' in models layer;
      RUNBOOK: file exists, mentions 5 steps, git-ignored env, mentions _parse_ohlcv_response

**Total: 509 / 514 tests passing** _(5 broken by 3060 changes; fixed in 2026-07-10 consolidation pass — see CHANGELOG)_

### Checkpoint 2.5 — Final Test Evaluation
- [x] Final test set evaluated ONCE (111 stocks → 330k rows → ~31k sealed test rows)
- [x] GPU-import invariant test covers all of src/models/
- [x] Full evaluation report generated (0.33% model vs -1.05% random return on test set)

### Cross-cutting Phase 2 constraints (non-negotiable)
- PIT: every feature `compute()` passes `as_of` through all DB queries; no `datetime.now()`
- Gate flags in every ExperimentTracker run (`gates_open`, `gates_closed`)
- `_FINAL_TEST_CUTOFF` constant in config prevents premature test-set access
- CPU-only: no `device("cuda")` in src/; invariant test extended to cover new dirs
- Every backtest stamps which data sources and gates were active

### Checkpoint 2.6.1 — Universe Correction (2026-07-11) ✅
- [x] Reconciled config universe (111 symbols) vs actual DB (111 symbols with data)
      Result: 106 in both; 5 in config only (no data); 5 in DB only (missing from config)
- [x] Diagnosed all mismatches — ADANIENT/AXISBANK/WIPRO/CONCOR/FEDERALBNK accidentally omitted;
      CHOLAFIN/LTIM/MCDOWELL-N/TATAMOTORS/ZOMATO had failed downloads (zero rows in DB)
- [x] Defined principled eligibility rules: Rule 1 (has data), Rule 2 (last >= 2026-01-01), Rule 3 (>= 600 trading days)
- [x] Built `tools/build_universe.py` — reproducible, auditable three-bucket eligibility script
- [x] Replaced hardcoded config.toml symbols with 109 DB-derived symbols + structured exclusion comments
- [ ] **Deferred to 3060**: Re-download CHOLAFIN/LTIM/MCDOWELL-N/TATAMOTORS/ZOMATO after fixing instrument-master issue
- [ ] **Deferred to 3060**: Re-download ADANIPORTS (9 pending chunks to complete)
- [ ] **Deferred to 3060**: Re-run training + evaluation on corrected universe (~114 target)
      ⚠️ The +0.33% result from Checkpoint 2.6 was on 106 symbols (missing ADANIENT/AXISBANK/WIPRO/
         CONCOR/FEDERALBNK; padded with 5 phantom entries). INVALID for the corrected universe.
         Must be re-run after re-downloads complete to get a valid backtest number.

## Phase 3 — Pattern Library
- [ ] Hypothesis register (pre-registered, mechanism required)
- [ ] Conditional rule miner (condition → outcome distribution, N + CI)
- [ ] Benjamini–Hochberg FDR correction
- [ ] Out-of-sample validation (discover years 1–3, confirm years 4–5)
- [ ] Base-rate comparison (lift must be real)
- [ ] Surviving patterns wired as ML features AND narrator reasons
- [ ] Periodic decay-retirement check

## Phase 4 — Scanner + Paper-Trading Engine

### Checkpoint 4.1 — Cost Simulation (2026-07-11) ✅
- [x] `src/trading/costs.py` — `CostSimulator`, `TradeCostBreakdown`, `TradeResult`
      Full NSE intraday MIS cost stack: brokerage (flat or %-with-cap), STT (sell leg),
      exchange charges (both legs), SEBI fees (total turnover), GST (on brokerage+exchange+SEBI),
      stamp duty (buy leg). All rates in config.toml [trading.costs] — zero code changes on rate update.
- [x] `config/config.toml [trading.costs]` — 7 rate parameters with source + as-of date comments
- [x] `src/config.py` — `CostConfig` pydantic sub-model added; `TradingConfig.costs: CostConfig`
- [x] `tests/test_costs.py` — 39 tests, 4 hand-calculated scenarios verified to the rupee
      Scenario 1: LONG profit  100×₹2500→₹2520  gross=₹2000  costs=₹135.89  net=₹1864.11
      Scenario 2: SHORT profit  50×₹1000→₹990   gross=₹500   costs=₹52.81   net=₹447.19
      Scenario 3: LONG loss     20×₹5000→₹4950  gross=-₹1000 costs=₹82.16   net=-₹1082.16
      Scenario 4: tiny 5×₹100→₹102 — pct model net=₹9.46 vs flat model net=-₹37.38 (cap overhead kills small trades)
- [x] Structural invariant: `TradeResult.__post_init__` asserts net_pnl==gross_pnl-costs.total;
      impossible to construct a TradeResult with a wrong net figure

### Checkpoint 4.2 — Slippage Model (2026-07-14) ✅
- [x] `src/trading/slippage.py` — `SlippageEstimate` (frozen, is_estimate always True),
      `SlippageModel` ABC (apply_entry, apply_exit, estimate), `FixedBpsSlippage`
- [x] `config/config.toml [trading.slippage]` — model, entry_bps, exit_bps (default 2+2)
- [x] `src/config.py` — `SlippageConfig` pydantic sub-model; `TradingConfig.slippage: SlippageConfig`
- [x] `tests/test_slippage.py` — 21 tests:
      Direction correctness (4): LONG entry higher, LONG exit lower, SHORT entry lower, SHORT exit higher;
      Exact bps math (6): 2bps, 5bps, 10bps, zero-bps no-slippage for both directions;
      is_estimate invariant (3): always True from model, False raises, both directions;
      estimate() composition (6): model_name propagated, entry/exit costs, total = entry+exit,
                                  short direction costs, zero-bps fields match quoted;
      ABC compliance (2): abstract not instantiable, FixedBps is concrete SlippageModel

### Checkpoint 4.3 — Paper Execution Engine (2026-07-14) ✅
- [x] `src/trading/execution.py` — `ExitReason` enum (TARGET_HIT/STOP_HIT/TIME_BASED/MANUAL);
      `FillResult` (frozen, is_paper=True); `ClosedTrade` (frozen, net_pnl==trade_result.net_pnl invariant);
      `OpenPosition` (unrealized_pnl); `Portfolio` (frozen, total_realized_net_pnl());
      `ExecutionEngine` ABC; `PaperExecutionEngine`
- [x] Cash accounting: LONG entry -=, exit += exit_filled - costs; SHORT entry +=, exit -= exit_filled + costs
- [x] Portfolio invariant: cash == initial_cash + sum(net_pnl) — proved algebraically, tested for all cases
- [x] `tests/test_execution.py` — 29 tests:
      Round-trip net_pnl matches CostSimulator (LONG + SHORT + loss + invariant check);
      All 4 exit reasons accepted;
      Portfolio cash invariant (1 LONG, 1 SHORT, winning+losing, 3 trades, no-trade, open position excluded);
      Guards (duplicate raises, close nonexistent raises, has_position lifecycle, reopen after close);
      Slippage integration (fills worse than quoted, net_pnl lower, invariant holds with slippage,
                           slippage estimate on ClosedTrade, costs computed on filled prices);
      Interface swap (MockAlternativeEngine satisfies ABC; abstract not instantiable);
      Dataclasses (is_paper=True default, ClosedTrade invariant prevents bad construction,
                   unrealized_pnl LONG + SHORT)

### Checkpoint 4.4 — Position-Sizing Guardrails (DEFERRED — required before live scanner)
- [ ] **Enforce DESIGN.md §12 limits inside PaperExecutionEngine** (or the scanner layer
      that calls it). Currently documented in config but not enforced:
      `max_position_pct = 0.10` (max 10% of pool per trade),
      `max_intraday_exposure_pct = 0.40` (max 40% total intraday),
      `min_cash_reserve_pct = 0.10` (always keep ≥ 10% cash),
      `drawdown_pause_pct = 0.20` (pause new entries if pool drops 20% from peak).
      Decision: wire guardrails at the scanner/allocation layer (Phase 4 stage-1 screen),
      not inside the engine itself — engine stays a neutral executor, caller enforces limits.

### Remaining Phase 4
- [ ] Stage-1 cheap screen (100 stocks, arithmetic only, ms-scale)
- [ ] Stage-2 full scoring (FinBERT + models + regime → top-10)
- [ ] Selection-bias guard (higher confidence bar for cherry-picked top-10)
- [ ] Regime classifier (risk-on / neutral / risk-off)
- [ ] Regime throttle on confidence + sizing
- [ ] Hard guardrails (DESIGN.md §12, cannot be overridden)
- [ ] Meta-label sizing model (fractional Kelly / vol-target)
- [ ] Two-tier monitor (watchlist + open-position low-latency loop)
- [ ] Event-priority + idempotency
- [ ] Correlation-aware risk (cross-sector beta cap)
- [ ] State persistence + clean restart / reconcile

## Phase 5 — Narrator, Dashboard & Self-Learning
- [ ] Ollama narrator (Qwen 2.5 7B / 3B fallback / template fallback)
- [ ] Anti-confabulation guard (stale data → "I don't have current data")
- [ ] Streamlit dashboard (cone, scenarios, positions, P&L, benchmarks, per-regime stats)
- [ ] Gated weekly retrain (validation gate, shrinkage, min-N, max-change cap)

## Phase 6 — Paper-Trading Evaluation (60–90 days)
- [ ] Live run on real data, no real money
- [ ] Track: accuracy, calibration, interval coverage, net P&L vs benchmarks
- [ ] Identify systematic biases & failure modes
- [ ] Calibrate confidence thresholds on live data
- [ ] Produce honest verdict: is there edge, and where?
