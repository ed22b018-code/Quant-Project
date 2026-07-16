# REUSABLE-ASSETS.md — What Has Lasting Value

The trading strategy has no edge (see [FINDINGS.md](FINDINGS.md)), but several components have
value **independent of this result**. This inventory is what to salvage and reuse — for a
different strategy, horizon, market, or an entirely separate project.

---

## 1. Rupee-accurate NSE cost engine ★★★ (highest reuse value)

- **Where:** `src/trading/costs.py` (`CostSimulator`, `TradeCostBreakdown`, `TradeResult`);
  config in `src/config.py` (`CostConfig`). Tests: `tests/test_costs.py` (39 tests).
- **What it is:** the complete NSE intraday cost stack — STT, exchange transaction charge,
  SEBI turnover fee, stamp duty, brokerage, GST — computed to the rupee and unit-tested.
- **Reuse for:** *any* NSE equity strategy or backtest, position-sizing that must be
  cost-aware, broker cost comparison, or a standalone "what will this trade actually cost"
  calculator. This is provider-agnostic and strategy-agnostic.
- **Caveat:** slippage/market-impact is a separate, deliberately light model
  (`FixedBpsSlippage`) — swap in a real impact model for large sizes.

## 2. Leakage-protection (PIT) framework ★★★

- **Where:** `src/data/base.py` (`DataSource` ABC, `DataLeakageError`); the `WHERE date < t0`
  discipline in `src/features/*`; `purge_and_embargo()` in `src/models/dataset.py`. Tests:
  `tests/test_pit_enforcement.py` (28 tests).
- **What it is:** a systematic guarantee that no feature or training row can see the future —
  point-in-time queries, purge/embargo around fold boundaries, and a sealed test set.
- **Reuse for:** any supervised ML on time series where look-ahead is the #1 silent failure.
  The pattern (ABC contract + dedicated exception + enforcement test suite) is portable to any
  domain.

## 3. Honest-evaluation harness ★★★ (the thing that caught the mirage)

- **Where:** `scripts/run_evaluation_v2.py`, `scripts/run_fo_evaluation.py`,
  `scripts/run_stress_test.py`; `src/models/calibration.py` (reliability/Brier).
- **What it is:** calibration reliability diagrams, after-cost backtesting against benchmarks
  (buy-and-hold, always-flat), per-month P&L with a **block t-stat** for significance, regime
  breakdown, feature-importance analysis, and a PIT probe — plus stress tests that found and
  killed the +37.66%→+5.37% exit-price bug.
- **Reuse for:** validating *any* trading model honestly. The block-t-stat + benchmark +
  after-cost pattern is the antidote to backtest self-deception.

## 4. Survivorship-corrected universe logic ★★

- **Where:** `src/data/eligibility.py`; `UniverseConfig` in `src/config.py`.
- **What it is:** the 109-symbol universe derived from F&O-eligibility rules rather than
  current index membership — avoids survivorship bias.
- **Reuse for:** constructing point-in-time-correct universes for any NSE study.

## 5. Multi-provider data layer ★★

- **Where:** `src/data/` — `DataSource` ABC with Dhan, Upstox, bhavcopy, corporate-action-
  adjusted, news, and option-chain implementations; `DhanBulkDownloader` +
  `DhanInstrumentMaster`; `NSETradingCalendar`; `DataHealthMonitor`.
- **What it is:** a provider-agnostic ingestion layer with corporate-action adjustment, an NSE
  calendar, instrument-master resolution (incl. futures naming), and health monitoring.
- **Reuse for:** any project needing clean, adjusted NSE data from Dhan/Upstox. The
  `DhanBulkDownloader` job-queue + discovered API limits (46-day span; minute-OI only for
  active contracts) are hard-won and documented.

## 6. Model + calibration scaffolding ★★

- **Where:** `src/models/` — `ScenarioClassifier`, `QuantileConeTrainer`, `ModelCalibrator`,
  `WalkForwardSplitter`, `TripleBarrierLabeler`, `run_training()`.
- **What it is:** a reusable LightGBM classification + quantile-cone + isotonic-calibration
  pipeline with walk-forward CV, triple-barrier labeling, and provenance stamping.
- **Reuse for:** any tabular time-series classification with abstention (`NO_TRADE`) and
  uncertainty bands. Swap features/labels; the harness stays.

## 7. Trading pipeline components ★

- **Where:** `src/trading/scanner.py` (100→10 funnel), `allocator.py` (fractional Kelly +
  guardrails), `execution.py` (paper engine + portfolio).
- **Reuse for:** any candidate-ranking → sizing → execution flow. Strategy-independent
  plumbing.

## 8. Downloaded data (3 years) ★★

- **Where:** `data/market.duckdb` (~2.8 GB, git-ignored; lives on the training machine).
- **What it is:** 3 years of adjusted NSE intraday OHLCV (40.3M rows across 1/5/15/25/60-min),
  daily OHLCV, and daily F&O ATM open-interest metrics (69k rows). See
  [DATA-INVENTORY.md](DATA-INVENTORY.md).
- **Reuse for:** re-testing new hypotheses without re-downloading. **Not committed to git** —
  must be copied between machines out-of-band (it exceeds sane git sizes).

---

## What is NOT worth keeping

- **The trained models** (`models/v1_full`, `models/v2_fo`) as *deployable strategies* —
  neither has edge. Keep only as reproducibility references.
- **Any expectation that adding more daily F&O granularity will help** — the evidence says it
  won't (it overfits).
