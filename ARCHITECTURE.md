# ARCHITECTURE.md — System Design

This document captures the **engineering** built for this project. The trading *strategy* has
no demonstrated edge (see [FINDINGS.md](FINDINGS.md)), but the **system design is reusable**
for a different strategy, horizon, or market. It is documented here so it can be picked up
later without re-reading the whole codebase.

**Stack:** Python 3.14, DuckDB (analytics store) + SQLite (state), LightGBM, Pydantic config,
pytest (670 tests). All timestamps stored in IST (Asia/Calcutta, UTC+5:30).

---

## 1. High-level pipeline

```
                 ┌─────────────────────────────────────────────────────────────┐
                 │                     CONFIG (Pydantic)  src/config.py           │
                 │  universe · costs · slippage · scanner · allocator · model     │
                 └─────────────────────────────────────────────────────────────┘
                                          │
   DATA LAYER                             ▼
   ┌──────────────┐   ┌────────────────────────────┐   ┌──────────────────────────┐
   │ Dhan / Upstox│──▶│ DataSource (ABC) interface  │──▶│ DuckDB market.duckdb      │
   │ Bhavcopy/News│   │ PIT-enforced, leakage-safe  │   │ (OHLCV, F&O, instruments) │
   └──────────────┘   └────────────────────────────┘   └──────────────────────────┘
                                          │
   FEATURE LAYER                          ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ FeatureRegistry + DataGate    price_volume · fo · news_count · sentiment     │
   │ every feature query: WHERE date < t0   (D−1 point-in-time rule)             │
   └───────────────────────────────────────────────────────────────────────────┘
                                          │
   LABELS + DATASET                       ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ TripleBarrierLabeler (±profit/stop/horizon) → Label{NO_TRADE,BULLISH,BEARISH}│
   │ DatasetBuilder · WalkForwardSplitter (5 folds) · purge_and_embargo()        │
   └───────────────────────────────────────────────────────────────────────────┘
                                          │
   MODEL LAYER                            ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ ScenarioClassifier (LightGBM 3-class) + QuantileConeTrainer (quantile bands) │
   │ ModelCalibrator (isotonic)  →  models/<name>/ + provenance.json             │
   └───────────────────────────────────────────────────────────────────────────┘
                                          │
   TRADING LAYER                          ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ Scanner (100→10 funnel) → CapitalAllocator (fractional Kelly + guardrails)   │
   │ → PaperExecutionEngine (Portfolio) → CostSimulator + SlippageModel          │
   └───────────────────────────────────────────────────────────────────────────┘
                                          │
   ORCHESTRATION                          ▼
   ┌───────────────────────────────────────────────────────────────────────────┐
   │ Orchestrator (staged run + resumable state) · Preflight checks · Tracker    │
   └───────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data layer (`src/data/`, `src/db/`)

**The `DataSource` abstraction is the spine of the system.** Every source of market data
implements one interface, so the feature pipeline is agnostic to provider.

- **`base.py`** — `DataSource(ABC)`; `DataLeakageError`, `DataNotAvailableError`. The ABC's
  contract is that any historical query is **point-in-time**: asking for data "as of t0"
  never returns rows dated ≥ t0. Violations raise `DataLeakageError`.
- **Concrete sources** (all implement `DataSource`):
  - `bhavcopy_source.BhavCopySource` — NSE end-of-day bhavcopy.
  - `corporate_actions.AdjustedBhavCopySource` — bhavcopy with corporate-action adjustment
    applied (splits/bonuses/dividends) via `CorporateActionFetcher`.
  - `dhan_adapter.DhanDataSource` — Dhan intraday/historical OHLCV.
  - `dhan_adapter.DhanFOMetricsSource` — Dhan F&O daily metrics (PCR, OI).
  - `live_upstox.UpstoxLiveSource` — Upstox live WebSocket adapter.
  - `news_ingest.NewsSource` / `RSSNewsIngest` — RSS news ingestion.
  - `option_chain.OptionChainSource` — option-chain snapshots.
  - `historical.StubHistoricalSource` — test/dev stub.
- **`dhan_downloader.py`** — `DhanInstrumentMaster` (177k-row instrument master; resolves
  symbol → security_id, incl. `RELIANCE-Jul2026-FUT` futures naming) and `DhanBulkDownloader`
  (chunked bulk history with a discovered 46-day span limit; job-queue driven).
- **`calendar.NSETradingCalendar`** — NSE holidays/sessions; the source of truth for
  "is this a trading day / trading minute."
- **`eligibility.py`** — F&O-eligibility logic → the survivorship-corrected universe.
- **`health_monitor.DataHealthMonitor`** — data completeness/gap monitoring.
- **Stores:** `db/duckdb_client.DuckDBClient` (bulk analytics — OHLCV, F&O, instruments) and
  `db/sqlite_client.SQLiteClient` (lightweight state); `db/schema.py` defines both.

**Why reusable:** point a new `DataSource` implementation at any provider/market and the
entire downstream pipeline works unchanged, with leakage protection inherited for free.

---

## 3. Feature layer (`src/features/`)

- **`base.py`** — `FeatureRegistry`, `FeatureMeta`, and `DataGate` (an enum marking which
  data dependency a feature needs: `intraday`, `fo`, `news`, `news_nlp`, `none`). Gates let
  the pipeline run with subsets of data and record which features were live in provenance.
- **`price_volume.py`** — 8 features: `return_1d`, `return_5d`, `vol_20d`, `volume_ratio_20d`,
  `atr_20d`, `vwap_deviation`, `intraday_momentum`, `opening_range_position`.
- **`fo.py`** — F&O features: `pcr_oi`, `pcr_volume`, `net_oi_change`, `max_pain_distance`,
  `spot_futures_basis`. **All queries enforce `WHERE date < CAST(t0 AS DATE)` (D−1 rule).**
- **`news_count.py`** — `news_count_1d`, `news_count_5d`.
- **`sentiment.py`** — `FinBERTScorer` → `news_sentiment` (built + tested, no historical data).

**Key invariant:** features are computed as-of t0 with only strictly-prior data. This is what
makes every backtest result trustworthy.

---

## 4. Labels + dataset (`src/models/dataset.py`, `labels.py`)

- **`labels.Label(IntEnum)`** — `NO_TRADE=0`, `BULLISH=1`, `BEARISH=-1`. `NO_TRADE` is a
  first-class label: flat windows stay out, and the model is explicitly rewarded for
  abstaining rather than forced to pick a direction.
- **`labels.TripleBarrierLabeler`** — labels each entry by which barrier is hit first within
  the horizon: profit (`+PROFIT_PCT`), stop (`−STOP_PCT`), or time (horizon bars elapsed).
- **`dataset.DatasetBuilder`** — assembles features + labels per symbol/timestamp.
- **`dataset.WalkForwardSplitter`** — expanding walk-forward splits. Fold logic:
  `fold_size = n_dev // (n_splits + 1)`; fold *k* validation = `dev[(k+1)·fold_size :
  (k+2)·fold_size]`. The **final** model trains on fold-5 train only, so fold-5 validation
  and the sealed test set are the true OOS periods. `get_final_test_set()` raises on a second
  call (seal guard).
- **`purge_and_embargo()`** — removes training rows whose label horizon (`t1`) overlaps the
  validation window start, plus an embargo buffer. Proven by test to leave zero leakage.

---

## 5. Model layer (`src/models/`)

- **`scenario_classifier.ScenarioClassifier`** — LightGBM 3-class classifier
  (`p_no_trade`, `p_bullish`, `p_bearish`). **Note:** `predict_proba` operates on `X.values`,
  so **column order must match training order** — `feature_names` is stored in `meta.json` and
  callers must reorder. (This bit us once in the head-to-head; the eval reorders explicitly.)
- **`quantile_cone.QuantileConeTrainer`** — LightGBM quantile regressors producing a
  price-cone (uncertainty band) around the horizon.
- **`calibration.ModelCalibrator`** — isotonic probability calibration with a reliability
  diagram, Brier score, and max-calibration-error; `UncalibratedConfidenceError` raised if
  confidence is consumed before calibration.
- **`training_pipeline.run_training()`** — the 4-stage driver: (1) concurrent dataset build
  (threaded per-symbol, with progress/ETA banners), (2) walk-forward splits + purge/embargo,
  (3) seal test set, (4) train classifier + cone on the last fold, then calibrate. Writes
  `models/<name>/scenario_classifier/` + `provenance.json` (universe, date range, features,
  gates, calibration, `model_type`, and a `feature_honesty` block).
  - **Critical correctness fix baked in:** the per-symbol query filters `AND timeframe='1'`.
    The DB stores 1/5/15/25/60-min bars that all include a 09:30 IST bar; without the filter,
    each daily feature timestamp matched multiple timeframe rows and triplicated the dataset.

---

## 6. Trading layer (`src/trading/`)

- **`costs.CostSimulator`** — rupee-accurate NSE intraday cost stack: STT, exchange
  transaction charge, SEBI turnover fee, stamp duty, brokerage, GST. Returns
  `TradeCostBreakdown` / `TradeResult`. **Tested to the rupee** — the trustworthy core of
  every after-cost number.
- **`slippage.py`** — `SlippageModel(ABC)` with `FixedBpsSlippage` (configurable bps entry +
  exit); returns `SlippageEstimate`. Seam for a future market-impact model.
- **`execution.PaperExecutionEngine`** (implements `ExecutionEngine` ABC) — simulates fills,
  maintains a `Portfolio` of `OpenPosition`s, records `ClosedTrade`s with `ExitReason`
  (profit/stop/time/EOD). Deterministic paper engine for backtests.
- **`scanner.Scanner`** — two-stage candidate funnel (≈100 → 10): cheap filters first, then
  model scoring, producing `ScanResult`s and `FilteredOut` audit records.
- **`allocator.CapitalAllocator`** — position sizing via **fractional Kelly** with hard
  guardrails (max position %, pool caps); emits `SizedOrder`s and `RejectedCandidate`s with
  reasons. Pool model: ₹5,00,000, max 10% per position.

---

## 7. Orchestration + experiment (`src/orchestration/`, `src/experiment/`)

- **`orchestrator.Orchestrator`** — runs the end-to-end pipeline as ordered `Stage`s with a
  persisted, **resumable** `OrchestratorState` (`*.orchestration_state.json`).
- **`preflight.py`** — `PreflightReport` / `CheckResult`; pre-run validation (data present,
  config sane, gates satisfiable) that fails fast with `PreflightError`.
- **`experiment.tracker.ExperimentTracker`** — run metadata/status tracking (`RunStatus`).

---

## 8. Cross-cutting invariants (do not break when reusing)

1. **IST everywhere.** DB timestamps are Asia/Calcutta (UTC+5:30). The 09:30 IST open bar is
   `HOUR=9, MINUTE=30`. (A past bug used `HOUR=4` assuming UTC — wrong.)
2. **PIT or it didn't happen.** Any new feature must query `date < t0`. Add a
   `test_pit_enforcement` case for it.
3. **Seal the test set.** Never evaluate on the final 10% more than once per model.
4. **Always report after-cost and against benchmarks.** Gross numbers are meaningless here —
   costs eat ~87% of gross.
5. **Column order matters for LightGBM `predict`.** Reorder inputs to `feature_names`.
6. **`model_type` + `feature_honesty` in provenance.** Every model records what its features
   really were (REAL / PARTIAL / NULL / ALL_ZERO) so comparisons stay honest.
