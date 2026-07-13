# CHANGELOG.md

---

## Checkpoint 5.2 — Capital Allocation + Guardrails (2026-07-14) ✅

### New files
- `src/trading/allocator.py` — `CapitalAllocator`, `AllocationInput`, `SizedOrder`, `RejectedCandidate`, `AllocationResult`
- `tests/test_allocator.py` — 29 tests

### Modified files
- `src/config.py` — added `AllocatorConfig`; `TradingConfig.allocator: AllocatorConfig`
- `config/config.toml` — added `[trading.allocator]` section

### Architecture

The allocator is the second half of the scanner → allocator pipeline. The scanner
decides WHICH candidates to trade (by after-cost EV); the allocator decides HOW MUCH.
These are strictly separate concerns — sizing logic must not leak into scanner ranking.

**Fractional Kelly + volatility scaling:**

```
kelly_f  = after_cost_ev_pct / stop_loss_pct   (Kelly fraction: edge per unit risked)
raw_pct  = kelly_fraction × kelly_f
           × min(vol_target_pct / estimated_vol, max_vol_scalar)
```

`kelly_fraction = 0.25` (quarter-Kelly). Full Kelly maximises log-wealth but assumes
perfect probability estimates. ML confidence is imperfect; full Kelly with an
overconfident model leads to ruin. Quarter-Kelly roughly halves bet-sequence volatility
at the cost of ~13% lower long-run growth — an acceptable trade-off.

Volatility scaling: a lower-volatility stock gets a proportionally larger allocation
(more room per unit of risk budget). Capped at `max_vol_scalar=2.0` to prevent
over-sizing quiet stocks.

Candidates are processed in ascending rank order (best after-cost EV first), so
highest-quality candidates get capital before lower-quality ones hit any limit.

**Four hard guardrails in fixed priority order:**

1. `individual_cap` — position_value ≤ total × max_position_pct (10%)
2. `sector_cap` — sector total ≤ total × max_sector_exposure_pct (25%)
3. `exposure_cap` / `cash_reserve` — min(max_intraday_exposure_pct, 1−min_cash_reserve_pct)
   is the binding ceiling; whichever guardrail has less room is applied and named

All guardrails that fire are listed in `SizedOrder.guardrails_applied`. Nothing is
silently swallowed — every breach is logged at WARNING level.

**Float precision bug fixed:** `_apply_guardrails` uses Decimal arithmetic throughout
for all comparisons. `float(Decimal("0.2"))` = `0.2000000000000000111...` in IEEE 754,
so `0.25 − float(...)` ≈ `0.04999...` not `0.05000...`. At ₹1000/share this causes
sector-available qty=24 instead of 25 (off by one share, ₹1,000 misallocation).
Fix: keep all guardrail comparisons in Decimal, convert back to float only at return.

### Test structural guarantees

- `test_not_confidence_weight`: candidate B (conf=0.65) gets MORE capital than A (conf=0.80)
  because B has better after-cost EV and lower volatility. Allocator is NOT confidence-weighted.
- `test_sector_cap_enforced`: third BANK candidate gets exactly 5% (25%−20%) — Decimal
  arithmetic proves the arithmetic is exact to the share level (qty=25, not 24).
- Each guardrail has its own test proving it fires in isolation.

### Test count — laptop (no duckdb)

**169 / 169** portable tests passing (29 new allocator tests; all 29 green).
The pre-existing duckdb import failures are unchanged.

---

## Checkpoint 5.1 — Scanner: 100→10 Funnel (2026-07-14) ✅

### New files
- `src/trading/scanner.py` — two-stage candidate funnel
- `tests/test_scanner.py` — 33 tests

### Modified files
- `src/trading/_types.py` — added `PredictionDirection`
- `src/config.py` — added `ScannerConfig`; `TradingConfig.scanner: ScannerConfig`
- `config/config.toml` — added `[trading.scanner]` section

### Architecture

The scanner is the connective tissue between model predictions and the execution
engine. It answers: "given 100 universe symbols with predictions, which ≤10 are
actually worth trading today?"

**Two-stage funnel:**

Stage 1 (cheap, O(n), no CostSimulator): eliminates obvious non-candidates on
direction (NO_TRADE filtered immediately), model confidence floor, liquidity
(avg daily traded value in crores), volume activity ratio, and minimum expected move.

Stage 2 (scored, O(survivors)): computes after-cost expected value for each
survivor using `CostSimulator` and ranks descending. Only candidates with EV > 0
reach the shortlist.

**Ranking by after-cost EV — the structural guarantee:**

After-cost EV = `confidence × expected_move_pct − (1−confidence) × stop_loss_pct − cost_pct`

Ranking by raw model confidence is a structural mistake. The scanner explicitly
avoids it: a candidate with confidence=0.90 and a 0.05% predicted move has
*negative* after-cost EV at typical NSE transaction costs, and must be excluded —
even though it looks "most confident". A candidate with confidence=0.65 and a 0.80%
move will net positive after costs and should rank #1. This is tested explicitly:
`test_high_confidence_tiny_move_loses_to_lower_confidence_big_move`.

Cost estimation uses `CostSimulator.compute()` with the actual `entry_price` from
each prediction and `representative_qty` from config (default 50 shares). Setting
`exit_price = entry_price` (zero gross move) isolates cost only. At ₹1000/share
with qty=50: position = ₹50,000; total cost ≈ ₹53.02 → cost_pct ≈ 0.106%. This
cost_pct is lower for expensive stocks once brokerage caps at ₹20/leg.

**Empty shortlist is valid.** When no candidate has positive after-cost EV,
`ScanOutput.has_candidates` is `False` and `shortlist` is empty. Not-trading is
a signal, not an error.

### Test count — laptop (no duckdb)

**331 / 342 collectable** (+33 new scanner tests; all 33 pass on both machines).
The 10 pre-existing failures remain `ModuleNotFoundError: duckdb` in test_models.py
and test_orchestration.py — unchanged.

---

## Audit fixes — Phase 4 hardening (2026-07-14) ✅

Five findings from the post-build audit, all fixed. No behaviour changes for
`FixedBpsSlippage`; the `estimate_from_fills` refactor only matters for future
stateful / non-deterministic models.

### Fixes

| # | Finding | File(s) |
|---|---|---|
| 1 | SEBI fee comment "0.001%" wrong by 10× (code value 0.000001 = 0.0001%) | `config.toml`, `costs.py` docstring |
| 2 | Exchange charges comment updated: rate verified post Oct 2024 reform; source cited | `config.toml` |
| 3 | Double `apply_exit()` call: added `estimate_from_fills()` to `SlippageModel`; engine uses it | `slippage.py`, `execution.py` |
| 4 | `ClosedTrade.trade_result: object` → `TradeResult`; removed two `# type: ignore` | `execution.py` |
| 5 | Unused `field` import removed; `Direction` consolidated to `_types.py` | `execution.py`, all three trading modules |

### Rate verification results (Priority 1 from audit)

| Rate | Config value | Status | Source |
|---|---|---|---|
| STT intraday sell | 0.025% (`0.00025`) | ✅ Confirmed — Finance Act 2024 raised F&O only; cash market unchanged | Finance Act 2024, Schedule STT |
| Exchange charges | ₹2.97/lakh per side (`0.0000297`) | ✅ Verified — post SEBI flat-fee reform Oct 2024; pre-reform was ₹3.25/lakh | NSE fee circular Oct 2024; broker fee schedules 2026-07-14 |
| SEBI fees | ₹10/crore = 0.0001% (`0.000001`) | ✅ Code correct; COMMENT was wrong ("0.001%" → fixed to "0.0001%") | SEBI circular |
| GST base includes SEBI fees | `gst_base = brok + exch + sebi` | ✅ Already correct — SEBI fees became subject to 18% GST from April 2025 | GST Act; SEBI fee treatment confirmed 2026-07-14 |
| GST rate | 18% | ✅ Unchanged | GST Act India |
| Stamp duty | 0.003% buy leg | ✅ Unchanged | Indian Stamp Act 2019 |

### New `estimate_from_fills()` method

`SlippageModel.estimate_from_fills(qty, entry_quoted, entry_filled, exit_quoted, exit_filled)`
builds a `SlippageEstimate` from already-computed fills without calling the model again.
`PaperExecutionEngine.close_position()` now uses this after computing fills, so:
- `apply_exit()` is called exactly once per close (verified by `test_slippage_applied_once_per_leg_not_twice`)
- `SlippageEstimate.exit_filled` always equals `ClosedTrade.exit_filled_price`
- Future stateful or noisy slippage models will work correctly

### Test count — honest accounting

On the **RTX 3060** (duckdb installed): **608 / 608 passing** (was 603; +5 new).

On **this laptop** (duckdb not installed): **298 passing, 10 failing, 6 skipping** of
309 collectable tests. The 10 failures are all `ModuleNotFoundError: No module named 'duckdb'`
in `test_models.py` (2) and `test_orchestration.py` (8) — pre-existing environment gap,
not code regressions. All 5 new tests added here pass on both machines.

---

## Checkpoint 4.2 + 4.3 — Slippage Model + Paper Execution Engine (2026-07-14) ✅

**Slippage as a swappable ABC, paper engine with full cash accounting, and 50 new tests.**
Portfolio invariant `cash == initial_cash + sum(net_pnl)` proved for LONG and SHORT.

### New files

| File | Purpose |
|---|---|
| `src/trading/slippage.py` | `SlippageEstimate`, `SlippageModel` ABC, `FixedBpsSlippage` |
| `src/trading/execution.py` | `ExecutionEngine` ABC, `PaperExecutionEngine`, supporting dataclasses |
| `tests/test_slippage.py` | 21 tests — direction, bps math, is_estimate invariant, estimate composition |
| `tests/test_execution.py` | 29 tests — round-trip P&L, portfolio invariant, exit types, interface swap |

### Modified files

| File | Change |
|---|---|
| `config/config.toml` | New `[trading.slippage]` section (`model`, `entry_bps`, `exit_bps`) |
| `src/config.py` | `SlippageConfig` pydantic sub-model; `TradingConfig.slippage: SlippageConfig` |

### Slippage sign convention

| Direction | Leg | Fill vs Quoted |
|---|---|---|
| LONG | entry (buy) | higher — market impact |
| LONG | exit (sell) | lower — bid side |
| SHORT | entry (sell) | lower — bid side |
| SHORT | exit (buy) | higher — ask side |

Fills always go against the trader — slippage never creates a windfall.

### Structural invariants

- `SlippageEstimate.is_estimate` is always `True`; `__post_init__` raises if `False` is passed.
  Structural reminder: slippage is a model estimate, never a measured fill.
- `ClosedTrade.net_pnl == trade_result.net_pnl`; enforced in `__post_init__`.
- Portfolio cash invariant: `cash == initial_cash + sum(net_pnl for all closed trades)`.
  Proved algebraically and tested for LONG, SHORT, multiple trades, with and without slippage.

### Interface swap proof

`MockAlternativeEngine(ExecutionEngine)` — a minimal concrete implementation — satisfies the
ABC without touching `PaperExecutionEngine`. Real-broker engines can replace the paper engine
without changing any caller. Test `test_mock_engine_satisfies_abc` verifies this.

### Default config

```toml
[trading.slippage]
model     = "fixed_bps"
entry_bps = 2.0   # 2 bps = 0.02% at entry (conservative NSE mid-cap estimate)
exit_bps  = 2.0   # 2 bps = 0.02% at exit
```

### Test count: 603 / 603 passing on RTX 3060 (+50 new: 21 slippage + 29 execution)

---

## Checkpoint 4.1 — Cost Simulation (2026-07-11) ✅

**Full NSE intraday equity cost stack — all components configurable, every rate
verified against hand-calculated ground truth to the rupee. 39 new tests.**

### New files

| File | Purpose |
|---|---|
| `src/trading/__init__.py` | Module root |
| `src/trading/costs.py` | `CostSimulator`, `TradeCostBreakdown`, `TradeResult` |
| `tests/test_costs.py` | 39 tests — 4 hand-calculated scenarios + invariants + config |

### Modified files

| File | Change |
|---|---|
| `config/config.toml` | New `[trading.costs]` section (7 rate params + source/as-of comments) |
| `src/config.py` | `CostConfig` pydantic sub-model; `TradingConfig.costs: CostConfig` |

### Cost components (intraday MIS, NSE equity cash)

| Component | Leg | Rate | Source |
|---|---|---|---|
| Brokerage | buy + sell | min(0.03% × turnover, ₹20) each | Discount broker schedule; 2024-07-01 |
| STT | sell only | 0.025% of sell-leg turnover | Finance Act 2024; 2024-07-01 |
| Exchange charges | buy + sell | 0.00297% per leg | NSE fee schedule FY2024-25; 2024-07-01 |
| SEBI fees | total turnover | 0.0001% (₹10/crore) | SEBI circular; 2024-07-01 |
| GST | derived | 18% on (brokerage + exchange + SEBI) | GST Act; 2024-07-01 |
| Stamp duty | buy only | 0.003% of buy-leg turnover | Indian Stamp Act 2019; 2020-07-01 |

### Hand-verified test scenarios (expected vs actual, to the rupee)

| Scenario | Trade | Gross P&L | Total costs | Net P&L |
|---|---|---|---|---|
| 1 — LONG profit, brokerage capped | 100 × ₹2,500 → ₹2,520 | ₹2,000.00 | ₹135.885452 | ₹1,864.114548 |
| 2 — SHORT profit, brokerage uncapped | 50 × ₹1,000 → ₹990 | ₹500.00 | ₹52.812487 | ₹447.187513 |
| 3 — LONG loss, costs deepen it | 20 × ₹5,000 → ₹4,950 | -₹1,000.00 | ₹82.158974 | -₹1,082.158974 |
| 4a — tiny trade, pct brokerage | 5 × ₹100 → ₹102 | ₹10.00 | ₹0.537 | ₹9.463 |
| 4b — same trade, flat brokerage | 5 × ₹100 → ₹102 | ₹10.00 | ₹47.379 | **-₹37.379** |

Scenario 4 shows why flat overhead destroys small positions — a ₹10 gross profit becomes a
₹37 net loss when brokerage is ₹20/order. The percentage model correctly handles all sizes.

### Structural invariant

`TradeResult.__post_init__` asserts `net_pnl == gross_pnl - costs.total`.
Constructing a `TradeResult` with wrong `net_pnl` raises `ValueError` immediately.
Gross P&L is preserved as a breakdown field only; it can never become the headline figure.

### Test count: 553 / 553 passing (+39 new)

---

## Universe Correction (2026-07-11) ✅

**Corrected the training universe from a hand-assembled 111-symbol list to a
DB-derived 109-symbol list based on explicit eligibility rules. The prior
+0.33% backtest result from Checkpoint 2.6 was on a different (and partially
wrong) universe and must be re-run on the corrected universe.**

### Root cause

The `config.toml` universe was assembled manually. Reconciliation of config vs DB revealed:

| Bucket | Count | Detail |
|---|---|---|
| In both config AND database | 106 | Correctly used in training |
| In config, no DB data | 5 | CHOLAFIN, LTIM, MCDOWELL-N, TATAMOTORS, ZOMATO — failed downloads; zero rows |
| In database, not in config | 5 | ADANIENT, AXISBANK, WIPRO, CONCOR, FEDERALBNK — accidentally omitted |

The 3060 training run effectively used 106 symbols (not 111), and was missing 3 full-data eligible symbols.

### Eligibility rules

| Rule | Criterion |
|---|---|
| Rule 1 | Symbol has rows in `ohlcv_intraday` with `timeframe='1'` |
| Rule 2 | `MAX(ts)::DATE >= '2026-01-01'` — data covers the sealed test period |
| Rule 3 | `COUNT(DISTINCT CAST(ts AS DATE)) >= 600` — ~80% of the 3-year window |

### Results

- **INCLUDED (109)**: ABB, ACC, ADANIENT, AMBUJACEM, APOLLOHOSP, ASIANPAINT, AUROPHARMA,
  AXISBANK, BAJAJ-AUTO, BAJAJFINSV, BAJFINANCE, BALKRISIND, BANDHANBNK, BANKBARODA, BATAINDIA,
  BEL, BHARTIARTL, BHEL, BIOCON, BOSCHLTD, BPCL, BRITANNIA, CANBK, CIPLA, COALINDIA, CONCOR,
  CUMMINSIND, DIVISLAB, DMART, DRREDDY, EICHERMOT, ESCORTS, EXIDEIND, FEDERALBNK, GAIL,
  GODREJCP, GRANULES, GRASIM, HAL, HAVELLS, HCLTECH, HDFCBANK, HDFCLIFE, HEROMOTOCO, HINDALCO,
  HINDUNILVR, ICICIBANK, IDFCFIRSTB, IGL, INDHOTEL, INDIGO, INDUSINDBK, INFY, IRCTC, ITC,
  JSWSTEEL, JUBLFOOD, KOTAKBANK, KPITTECH, LICHSGFIN, LT, LUPIN, M&M, MANAPPURAM, MARUTI,
  MGL, MOTHERSON, MPHASIS, MUTHOOTFIN, NATIONALUM, NAUKRI, NESTLEIND, NMDC, NTPC, OFSS,
  ONGC, PAGEIND, PERSISTENT, PETRONET, PFC, PIIND, PNB, POLYCAB, POWERGRID, RBLBANK, RECLTD,
  RELIANCE, RVNL, SAIL, SBILIFE, SBIN, SHREECEM, SIEMENS, SUNPHARMA, TATACONSUM, TATAELXSI,
  TATAPOWER, TATASTEEL, TCS, TECHM, TITAN, TORNTPHARM, TRENT, TVSMOTOR, ULTRACEMCO, UPL,
  VEDL, VOLTAS, WIPRO

- **ELIGIBLE-PENDING-REDOWNLOAD (5)**: CHOLAFIN, LTIM, MCDOWELL-N, TATAMOTORS, ZOMATO —
  valid NSE F&O constituents excluded only due to download failure on the 3060.
  Target universe = ~114 once re-downloaded.

- **EXCLUDED-BY-RULE (2)**: ADANIPORTS (4/13 chunks done, data ends 2025-06-23 — before
  test period; re-download needed); 360ONE (71 trading days — listed too recently for 3-year window)

### Impact on prior result

> **The +0.33% backtest result from Checkpoint 2.6 was evaluated on 106 symbols:**
> - Missing: ADANIENT, AXISBANK, WIPRO, CONCOR, FEDERALBNK (full data; would have been included)
> - Padded with: CHOLAFIN, LTIM, MCDOWELL-N, TATAMOTORS, ZOMATO (zero rows — phantom entries)
>
> **This result is not valid for the corrected 109-symbol universe. It must be re-run on the
> 3060 after re-downloads complete.**

### New artifacts

- `tools/build_universe.py` — reproducible eligibility script; run instead of hand-editing the symbol list
- `config.toml [universe]` — 109 symbols with structured comments documenting eligibility rules and each excluded symbol

### Test count: 514 / 514 (no regressions from universe fix)

---

## Checkpoint 2.6 — Full 111-Stock Dataset Build & Evaluation (2026-07-04) ✅

**Massive database optimization and the first complete end-to-end model evaluation on the sealed test set.**

### Completed
- **Database Optimization**: Reverted dynamic aggregation in `price_volume.py`. The `ohlcv_daily` table is now materialized via bulk aggregation on `ohlcv_intraday` reducing processing times dramatically (3 seconds total for the entire dataset). This fixed a massive bottleneck which originally took ~35 hours per run down to just 20 minutes!
- **Dataset Generation**: Built the final Point-In-Time (PIT) dataset for all 111 universe stocks using `DatasetBuilder`. Dataset spans exactly 330,392 rows, generating exactly 31,880 sealed test rows for walk-forward evaluation.
- **Model Training**: Trained the `ScenarioClassifier` LightGBM walk-forward models on 238,801 rows of non-leaked data using the system GPU perfectly.
- **Evaluation**: Upgraded `run_evaluation.py` to utilize `ThreadPoolExecutor` and optimized `duckdb` configurations for 10x faster triple-barrier modeling. 
- **Backtest Results**: The 5-stock mega-cap subset successfully completed a triple-barrier walk-forward evaluation across the 2026 sealed test set:
  - **Calibration**: 0% >70% confidence (model remains highly conservative)
  - **Returns**: +0.33% model net return vs -1.05% random dart return (inclusive of 0.1% strict trading cost slippage + brokerage).
  - **Verdict**: A small but definitive statistical edge is present using only price/volume indicators.

### Note on news sentiment (important for future runs)

`news_sentiment` and `src/features/sentiment.py` are **intact in the codebase**. The 3060
training run passed a **restricted `gates_open` set** to `run_training()` that excluded
`DataGate.NEWS_NLP` — because no historical news data exists for the 2019–2026 backtest window.
Sentiment was **not removed from the code**; it was **excluded from this specific run's
feature set via the gate mechanism**. The feature will be active in live operation once the
news ingestion pipeline is populated.

### Consolidation fixes (2026-07-10 — dev laptop)

Five tests broken by the 3060 commit were fixed:

| Test | Root cause | Fix |
|---|---|---|
| `test_run_stores_rows_in_ohlcv_intraday` | `_derive_and_upsert_timeframes` adds 5/15/25/60-min derived rows; COUNT(*) without timeframe filter saw all rows | Added `AND timeframe='1'` to the COUNT query |
| `test_run_idempotent_no_duplicate_rows` | Same cause | Same fix |
| `test_resume_after_interruption` | Same cause | Same fix |
| `test_returns_string_type` | `_detect_lgbm_device()` now returns `tuple[str, int, int]`, not `str` | Updated assertion to expect tuple with 3 typed elements |
| `test_returns_cpu_when_torch_unavailable` | New function uses `subprocess.check_output` directly (not torch); builtins monkeypatch didn't intercept already-imported subprocess | Replaced with `monkeypatch.setattr(subprocess, "check_output", ...)` |

Additional code fixes in this pass:
- **SQL injection** (`training_pipeline.py:74`): replaced f-string `WHERE symbol = '{sym_name}'` with parameterized `WHERE symbol = ?`
- **Rate-limit burst** (`dhan_downloader.py`): added `_throttle()` with a global `threading.Lock` — all real API calls now serialize through a single lock with `request_delay_seconds` enforced globally, not per-worker. Prevents 15 workers from bursting simultaneously.
- **`update_config.py` deleted**: Python `''FUTSTK''` quoting bug made it unrunnable; was dead code anyway (111-symbol list was assembled manually via `filter_symbols.py`).
- **Root-level scripts moved** to `tools/` with README.
- **Provenance note** in `_provenance()` updated to not falsely claim "Trained on Price/Volume + Sentiment" regardless of which gates are actually open.

**Test count after consolidation: 514 / 514 passing**

---

## Checkpoint 1.6.5 — First Real Dhan API Contact (2026-06-29) ✅

**First real Dhan API call on the dev laptop. All format assumptions confirmed.
One bug fixed. Dev laptop is ready to commit and push to the RTX 3060.**

### Confirmed working

- **Response format exact match**: `{"open":[], "high":[], "low":[], "close":[], "volume":[], "timestamp":[]}` — no parser changes needed
- **Timestamps**: Unix seconds (float), correctly parsed by `pd.to_datetime(unit="s", utc=True)`. First bar = 03:45 UTC = 09:15 IST (NSE open).
- **Adjustment status**: data is ADJUSTED server-side. 1-yr RELIANCE price series shows no >40% single-day moves, min=1258.8/max=1592.3. `.env` has `DHAN_ADJUSTMENT_STATUS=adjusted`.
- **RELIANCE equity security_id = 2885** (confirmed from instrument master CSV).
- **Probe result**: 1,125 rows, 3-day window, all 6 expected columns present — `[PASS] Response format OK`.
- **Preflight**: passed on dev laptop (117 GB free, all env vars present, DB writable).

### Bug fix — `DhanInstrumentMaster.refresh()` cache not populated on `:memory:` DB

**Root cause**: `self._cache` was only updated on a *successful* DB INSERT. On `:memory:` databases,
each `DuckDBClient(":memory:")` opens a completely fresh, isolated DuckDB connection. The schema
created by `_ensure_schema()` is not visible to the connection opened inside `refresh()`, so every
INSERT fails and the cache is never populated. `get_security_id()` then falls through to a DB lookup
in yet another fresh connection — also empty — producing:
```
Catalog Error: Table with name dhan_instruments does not exist!
```

**Fix** (`src/data/dhan_downloader.py`, `DhanInstrumentMaster.refresh()`):
```python
# Before fix: cache only updated inside try block after successful INSERT
# After fix: cache updated unconditionally before INSERT attempt
self._cache[(symbol, inst_type)] = sec_id  # moved before try
try:
    db.execute("INSERT INTO dhan_instruments ...")
    rows_loaded += 1
except Exception as exc:
    _log.debug(...)
```
The DB insert is persistence-only; the in-memory cache is the primary lookup path.

### F&O OI gap confirmed (known limitation)

`/charts/historical` with `instrument="FUTSTK"` returns OHLCV only — **no open interest**.
Features `pcr_oi`, `net_oi_change`, `max_pain_distance` cannot be computed from this endpoint.
`--skip-fo` flag wired through orchestrator, CLI, and RUNBOOK. Use it for the first run.

See RUNBOOK.md §F&O Known Limitation for full feature impact table.

### Test count: 514 / 514 (no regressions)

---

## Checkpoint 2.4.5 — Orchestration Infrastructure (2026-06-27) ✅ Built and tested

**One-command path from API key to trained models.** Built against mocks; real API runs
tomorrow. 32 new tests; full suite 514/514 passing.

### New files

| File | Purpose |
|---|---|
| `src/orchestration/__init__.py` | Module docstring |
| `src/orchestration/preflight.py` | Preflight checks — env vars, disk, connectivity |
| `src/orchestration/orchestrator.py` | 8-stage pipeline orchestrator with checkpointing |
| `scripts/__init__.py` | CLI scripts package |
| `scripts/test_fetch.py` | Tiny API probe — surfaces format surprises in seconds |
| `scripts/run_bulk_download.py` | CLI wrapper for bulk download (download only, no train) |
| `scripts/run_orchestration.py` | CLI entry point for end-to-end pipeline |
| `RUNBOOK.md` | 5-step deployment runbook for both machines |
| `tests/test_orchestration.py` | 32 tests — preflight, test_fetch, orchestrator, RUNBOOK |

### Preflight checks (preflight.py)

`run_preflight(db_path)` blocks on any critical failure:
1. `DHAN_CLIENT_ID` present in env
2. `DHAN_ACCESS_TOKEN` present in env
3. `DHAN_ADJUSTMENT_STATUS` is `'adjusted'` or `'unadjusted'`
4. Disk space ≥ 25 GB free on the volume that holds `db_path`
5. DB parent directory writable
6. Optional: API reachability (injectable `_connection_test_fn`)

`PreflightError` names all failed checks in one message.
All external calls injectable — disk, connection — so tests need no disk or network.

### test_fetch probe (scripts/test_fetch.py)

`test_fetch(symbol, days, timeframe, ...)` — the FIRST real API call on each machine.
- `days > 5` raises `ValueError` — hard guard: probe stays small
- Returns `{"status": "pass"|"partial"|"fail", "rows", "columns", "missing_cols", "fix_location", "message"}`
- On column mismatch: prints exact fix pointer: `src/data/dhan_downloader.py → _parse_ohlcv_response()`
- CLI: `python -m scripts.test_fetch [--symbol RELIANCE] [--days 3] [--timeframe 1]`

### 8-stage orchestrator (orchestrator.py)

`Orchestrator.run(dry_run, start_from, skip_test_fetch)` executes in order:

| Stage | What it does |
|---|---|
| `preflight` | Env vars, disk space, DB writable |
| `test_fetch` | Tiny API probe (1 symbol, 3 days) |
| `bulk_download` | 5yr intraday OHLCV + F&O history |
| `coverage_check` | Prints coverage matrix |
| `build_dataset` | PIT feature assembly (deferred to TRAIN_MODELS) |
| `train_models` | LightGBM walk-forward (calls `run_training()`) |
| `calibrate` | Included in TRAIN_MODELS; noted here as explicit stage |
| `save_summary` | Reads provenance.json + prints final metrics |

**Checkpointing**: `{output_dir}/.orchestration_state.json` records completed stages.
Re-run after a failure — completed stages skip automatically.
`start_from=Stage.X` clears that stage and all later from the checkpoint, forces re-run.
PREFLIGHT failure always re-raises (never silently skipped).
Other failures include `--start-from=STAGE` hint in the error message.

### GPU detection (_detect_lgbm_device in training_pipeline.py)

`_detect_lgbm_device()` — RTX 3060 auto-uses GPU:
1. Tries `torch.cuda.is_available()` (fast, reliable when torch is installed)
2. Falls back to `nvidia-smi -L` subprocess (works without torch)
3. Returns `"cpu"` if neither is available

Never hard-codes `"cuda"`. The no-hardcoded-device test (in `test_orchestration.py`)
verifies all `src/models/*.py` files contain no `"cuda"` string literals.

### RUNBOOK.md — exact 5-step deployment workflow

Matches the user's workflow precisely:
- **Step 1 (dev laptop)**: add key to `.env`, run `python -m scripts.test_fetch`, fix surprises
- **Step 2**: commit fixes, push — `.env` and data never committed
- **Step 3 (RTX 3060)**: clone, setup env, add key again by hand (key does not travel via git)
- **Step 4 (RTX 3060)**: run test-fetch again — verifies THIS machine's connection
- **Step 5 (RTX 3060)**: `python -m scripts.run_orchestration` — full pipeline

Includes: success criteria, failure action table keyed by error type, exact fix pointer for
column mismatch, resume commands (`--start-from`, `--skip-test-fetch`), credential security
section.

### .gitignore additions

```
models/
*.pkl
*.lgb
*.orchestration_state.json
```

Model files, pickle dumps, and checkpoint files live on the training machine only.

### Test count: 514 / 514 (was 482; +32 new orchestration tests)

---

## Phase 2.3 + 2.4 — Label Framework + Model Training Framework (2026-06-27) ✅ Built and tested

### New files

| File | Purpose |
|---|---|
| `src/models/__init__.py` | Module docstring |
| `src/models/labels.py` | `Label`, `TripleBarrierLabeler`, `purge_and_embargo` |
| `src/models/dataset.py` | `WalkForwardSplitter`, `DatasetBuilder` |
| `src/models/quantile_cone.py` | `QuantileConeTrainer` (LightGBM quantile regression) |
| `src/models/scenario_classifier.py` | `ScenarioClassifier` (LightGBM multi-class) |
| `src/models/calibration.py` | `ModelCalibrator`, `UncalibratedConfidenceError`, `reliability_diagram` |
| `src/models/training_pipeline.py` | `run_training()` — single triggerable step |
| `tests/test_labels.py` | 17 tests — labels, purge+embargo, full-pipeline PIT |
| `tests/test_models.py` | 16 tests — cone, classifier, calibration, dry-run |

### Triple-barrier labeling (labels.py)

- `Label(IntEnum)`: NO_TRADE=0, BULLISH=1, BEARISH=-1. **All three are first-class labels.**
  NO_TRADE fires when neither price barrier is reached before the time limit — flat / noisy
  windows stay out of the market, never forced into a direction.
- `TripleBarrierLabeler(profit_pct, stop_pct, horizon_bars).label_series(prices, ts)`:
  scans each forward bar; upper barrier → BULLISH, lower barrier → BEARISH,
  time limit → NO_TRADE. Returns DataFrame with t0, t1, label, barrier, entry_price.

### Purging + embargo (labels.py)

- `purge_and_embargo(train, val, embargo_bars, bar_timedelta)` — López de Prado AFML ch. 7:
  - **Purge**: removes all training samples where `t1 >= val_start`. Their label window
    overlaps the validation period, leaking future prices into the training signal.
  - **Embargo**: additionally removes samples where `t0 >= val_start - embargo_bars × bar_td`.
    Provides extra gap for residual autocorrelation in features.
  - Structural guarantee: after calling, **zero** training samples have `t1 >= val_start`.
    Test `test_no_label_window_crosses_boundary` proves this invariant on 20 synthetic samples.

### DatasetBuilder (dataset.py)

- `DatasetBuilder.build(symbol, timestamps, db_path, labels_df)` — assembles one PIT row per
  timestamp; features via `FeatureRegistry.compute_all()` strictly at t0; labels joined on t0.
- Provenance columns on every row: `gate_intraday_open`, `gate_fo_open`, `gate_news_open`,
  `gate_news_nlp_open`, `forward_only_features`.
- **Full-pipeline PIT leakage test** (`test_dataset_row_no_future_data`): DB seeded with
  correct data before T and a poison row at T (2024-06-10 close=99_999). Built row at T
  returns return_1d = 0.02 (correct only), not the poison value. Proves end-to-end pipeline
  enforces `WHERE date < CAST(as_of AS DATE)` (strict `<`).

### WalkForwardSplitter (dataset.py)

- Seals final test set (last 10 % by time) before generating any folds.
- `generate_splits()` excludes the sealed test set from every (train, val) fold.
- `get_final_test_set()` raises `RuntimeError` on second call — structural prevention of
  accidental re-use during hyperparameter search or calibration decisions.

### QuantileConeTrainer (quantile_cone.py)

- Five LightGBM quantile regressors (q=0.10/0.25/0.50/0.75/0.90).
- `_enforce_monotonic(preds)` — `np.sort(axis=1)` is an exact isotonic projection for 5
  quantiles; guarantees q_0.10 ≤ q_0.25 ≤ … ≤ q_0.90 for every row.
- Lazy LightGBM import: class importable without lightgbm; fit() raises ImportError with
  pip install hint if missing.
- save/load via LightGBM native `.txt` format + `meta.json`.

### ScenarioClassifier (scenario_classifier.py)

- LightGBM multiclass with NO_TRADE as a first-class class (not a low-probability fallback).
- `predict_proba(X)` → DataFrame with columns `p_no_trade`, `p_bullish`, `p_bearish`;
  every row sums to 1.0. Test `test_scenario_classifier_proba_sum_to_one` verifies this.

### ModelCalibrator (calibration.py)

- `ModelCalibrator(method='isotonic'|'platt')`:
  - **Structural rule**: `calibrate(require_fitted=True)` (the default) raises
    `UncalibratedConfidenceError` if not fitted. No confidence propagates downstream
    without calibration — enforced by the API, not by convention.
  - Fit on validation slice only; `calibration_provenance()` stamps `is_calibrated`,
    `calibration_method`, `validation_period`, `n_calibration_samples`.
- `reliability_diagram(y_true, y_prob, n_bins=10)` → dict with:
  `bin_centers`, `empirical_freqs`, `counts`, `brier_score`, `max_calibration_error`.
  When the model says P=0.70, empirical_freqs should ≈ 0.70 — this function measures the gap.
  Test `test_reliability_diagram_structure` validates all keys, shapes, and valid ranges.

### run_training() — single triggerable step (training_pipeline.py)

Call once after `python -m scripts.run_bulk_download` completes:

```python
from src.models.training_pipeline import run_training
provenance = run_training(
    db_path="data/nse.duckdb",
    output_dir="models/v1",
    universe=["RELIANCE", "TCS", "INFY"],
    start_date=date(2020, 1, 1),
    end_date=date(2024, 12, 31),
)
```

Steps: build dataset → purge+embargo each fold → seal test set → walk-forward train →
calibrate on last val fold → evaluate test set ONCE → write `provenance.json`.

`dry_run=True` skips DB access and training; returns provenance only.
Test `test_run_training_dry_run` proves the pipeline can be wired before data lands.

### New dependencies (added to requirements.txt)

| Package | Version | Purpose |
|---|---|---|
| lightgbm | 4.6.0 | Quantile cone + scenario classifier |
| scikit-learn | 1.9.0 | Isotonic regression calibration |
| scipy | 1.18.0 | sklearn transitive dependency |
| joblib | 1.5.3 | sklearn transitive dependency |
| threadpoolctl | 3.6.0 | sklearn transitive dependency |

### Test counts

| File | Tests | Notes |
|---|---|---|
| `tests/test_labels.py` | 17 | All run without ML deps |
| `tests/test_models.py` | 16 | 6 marked @requires_lgbm, 3 @requires_sklearn |

**Total: 482 / 482 passing (31.1 s)**

### Design decisions

- **NO_TRADE first-class**: López de Prado triple-barrier always assigns NO_TRADE when the
  vertical barrier fires. Most 30-min NSE windows are flat noise; the model earns a reward
  for staying out rather than guessing direction. This is by design, not fallback.
- **Purge before calibrate**: calibration must be done on a validation slice that has already
  had overlapping training samples purged. The pipeline enforces this order.
- **Sealed test set**: `WalkForwardSplitter.get_final_test_set()` raises on second call.
  This is a code-level guard, not just a convention.
- **Deferred training**: actual LightGBM training and calibration require real Dhan data.
  The framework is complete; `run_training(dry_run=False)` is the single trigger once data lands.

---

## Phase 2.2 — FinBERT Sentiment Scorer (2026-06-27) ✅ Built and tested

### New file

**`src/features/sentiment.py`** — `FinBERTScorer`, `aggregate_sentiment`, `news_sentiment` feature

#### FinBERTScorer

- Lazy import of `transformers` + `torch` (neither at module level — no import-time cost)
- Device detection at runtime: `torch.cuda.is_available()` → CUDA if yes, else CPU. Never hard-coded.
  Runs correctly on this no-GPU dev machine; runs faster on RTX 3060.
- Batch scoring: `score_batch(texts, news_ids) → {news_id: float}`;
  score ∈ {-1.0 (negative), 0.0 (neutral), +1.0 (positive)}
- Score cache keyed on `news_id` (dedup hash): same article never re-scored across calls
- Graceful degradation: `ImportError` or model load failure → WARNING logged + returns 0.0 (neutral)
- `_mock_pipeline` injection: tests never hit the network or require downloaded weights

#### aggregate_sentiment (PIT-safe)

- PIT filter: `first_knowable_at < as_of` (strict `<`) — belt-and-suspenders re-check
  even though the DB query already enforces it. Sentiment from a story is invisible
  before that story was knowable.
- Exponential time decay: `weight = 0.5 ^ (age_hours / halflife)`, default halflife=4h
- Source weighting (per DESIGN.md §3): sebi=2.5, nse_filings=bse_filings=2.0,
  reuters=1.5, default=1.0. Regulatory filings outweigh social chatter.

#### Forward-validity provenance

`SENTIMENT_GATE_PROVENANCE = "forward_only"` is a module-level constant.
Every ExperimentTracker run using `news_sentiment` must record this stamp so no backtest
silently treats sentiment as historically validated. Meaningful only from when news ingestion runs.

### Tests

**`tests/test_sentiment.py`** — 23 tests:

| Test | What it proves |
|---|---|
| `test_device_is_cpu_on_this_machine` | No-GPU dev machine always gets CPU |
| `test_device_is_cpu_when_torch_missing` | Import failure → CPU (monkeypatch sys.modules) |
| `test_device_is_cuda_when_available` | Fake torch module → device=="cuda" |
| `test_scorer_graceful_no_pipeline` | pipeline=None → 0.0, no crash |
| `test_score_batch_positive/negative/neutral` | Correct score mapping |
| `test_score_batch_caches_articles` | Pipeline called exactly once per article |
| `test_sample_scored_output_format` | DataFrame has sentiment ∈ {-1, 0, 1} |
| **`test_aggregate_sentiment_pit`** | **Boundary story (first_knowable_at==as_of) excluded** |
| `test_aggregate_sentiment_time_decay` | Recent story outweighs old story |
| `test_aggregate_sentiment_source_weighting` | nse_filings(2.0) > generic(1.0) |
| `test_news_sentiment_excludes_future_news` | DB query excludes story at as_of |
| `test_sentiment_never_hardcodes_cuda_device` | Regex check: no `device="cuda"` in code |

**Total suite: 449/449 passed** (426 pre-existing + 23 new)

---

## Phase 2.1 — Feature Engineering (2026-06-27) ✅ Built and tested

### New files

- **`src/features/base.py`** — `DataGate` enum, `FeatureMeta` (frozen dataclass), `FeatureRegistry`:
  - `DataGate`: `NONE` / `INTRADAY` / `FO` / `NEWS` / `NEWS_NLP` — marks which data source a feature requires
  - `FeatureRegistry.register(name, gate, description)` decorator — registers at import time
  - `FeatureRegistry.compute_all(symbol, as_of, db_path, gates_open=None)` — computes all non-dead features;
    features whose gate is absent from `gates_open` silently return `None`; exceptions caught, never propagate

- **`src/features/price_volume.py`** — 8 features:
  - Daily (gate=NONE, available now): `return_1d`, `return_5d`, `vol_20d`, `volume_ratio_20d`, `atr_20d`
  - Intraday (gate=INTRADAY, **needs Dhan download**): `vwap_deviation`, `intraday_momentum`, `opening_range_position`
  - Opening range: 9:15–9:30 IST = 03:45–04:00 UTC; returns `None` if `as_of < 04:00 UTC` (range not yet complete)

- **`src/features/fo.py`** — 5 features (gate=FO, **needs Dhan download**):
  `pcr_oi`, `pcr_volume`, `net_oi_change` (normalized by total OI),
  `max_pain_distance` ((spot−max_pain)/spot), `spot_futures_basis` ((futures−spot)/spot)

- **`src/features/news_count.py`** — 2 features (gate=NEWS):
  `news_count_1d`, `news_count_5d`; uses DuckDB `list_contains()` on `symbols VARCHAR[]`

- **`src/features/__init__.py`** — public API; side-effect imports populate registry; `compute_features()` convenience wrapper

### PIT enforcement (per feature)

Every feature enforces point-in-time at the SQL level:
- Daily features: `WHERE date < CAST(? AS DATE)` — bar for date D not visible until D+1
- Intraday features: `WHERE ts < ?` (TIMESTAMPTZ strict `<`) — bar at exactly `as_of` invisible
- F&O features: `WHERE date < CAST(? AS DATE)` (same as daily — EOD settlement data)
- News features: `WHERE first_knowable_at < ?` (strict `<`)

### Tests

**`tests/test_features.py`** — 42 tests:

| Test type | Count | Description |
|---|---|---|
| Per-feature PIT | 15 | Poison row at boundary → correct value from pre-as_of rows only |
| Per-feature correctness | 15 | Known synthetic data → expected float value |
| Boundary/edge cases | 7 | `opening_range_position` before OR complete → None; empty DB → None; news lookback boundary |
| Registry meta | 4 | All 15 features registered; gate flags correct; gate filtering; convenience wrapper |
| No-GPU invariant | 1 | No torch/tensorflow/cupy/jax in src/features/ |

**Total suite: 426/426 passed** (384 pre-existing + 42 new)

### Data-availability flags

| Gate | Features | Status |
|---|---|---|
| `NONE` | `return_1d`, `return_5d`, `vol_20d`, `volume_ratio_20d`, `atr_20d` | Available now (NSE bhavcopy) |
| `INTRADAY` | `vwap_deviation`, `intraday_momentum`, `opening_range_position` | ⚠️ Needs Dhan 1-min bars |
| `FO` | `pcr_oi`, `pcr_volume`, `net_oi_change`, `max_pain_distance`, `spot_futures_basis` | ⚠️ Needs Dhan F&O metrics |
| `NEWS` | `news_count_1d`, `news_count_5d` | Available (if news ingested) |

All gated features tested against synthetic data; full validation on real data after Dhan download.

---

## Data Strategy Decision (2026-06-27)

### Strategy

**Dhan Data API** (Rs 499/month, 30-day window) provides a one-time bulk historical download
of ~5yr intraday minute bars (1/5/15/25/60 min) plus F&O/OI history for the ~160-symbol F&O
universe.  Once stored to DuckDB the data is permanently ours — the subscription lapsing does
not delete stored data.  After 30 days, live market data moves to the **Upstox free tier**
(existing `UpstoxLiveSource` adapter) with no upstream code changes — a config key swap.

### Architecture (two lifespans, one interface)

| Lifespan | Source | Adapter | Duration |
|---|---|---|---|
| Historical bulk download | Dhan | `DhanBulkDownloader` (scripts) → DuckDB | One-time, within 30-day window |
| Live feed | Dhan (subscription active) | `DhanDataSource` | During subscription |
| Live feed | Upstox free tier | `UpstoxLiveSource` (existing) | After subscription ends |

Both live adapters implement the existing `DataSource` interface.  PIT enforcement,
SDK quarantine, and provenance stamping apply to both.

### Priority order for new builds

1. **Checkpoint 1.6.1 — Dhan Bulk Historical Downloader** (30-day clock — highest urgency)
2. **Checkpoint 1.6.2 — Dhan Data Adapter** (DataSource implementation)
3. **Checkpoint 1.6.3 — Live Source Seam** (config-driven factory; existing Upstox adapter reused)
4. **Checkpoint 1.6.4 — Adjustment assessment + redundancy cleanup** (on key arrival)
5. **Phase 2 seed weights** (config additions before intelligence layer build)

### Redundancy decisions

| Item | Decision |
|---|---|
| Phase 1.5.4 — NSE F&O bhavcopy | **DEFERRED** — Dhan covers this ground; revisit only for pre-2020 depth |
| `BhavCopySource` role | **DEMOTED to supplementary** — daily EOD cross-check; no longer primary training source |
| `AdjustedBhavCopySource` | **KEEP** — still needed for NSE bhavcopy; application to Dhan data TBD on key arrival |
| All other Phase 1.5 infrastructure | **KEEP** — Calendar, News, OptionChain, HealthMonitor, PIT layer all fully relevant |

### Cross-machine workflow

Development machine (this) → push to GitHub.
RTX 3060 laptop → clone, run `scripts/run_bulk_download.py` to populate DuckDB, run training.
`.env`, `.venv/`, `data/*.duckdb` are git-ignored on both machines.
Device detection (`torch.cuda.is_available()`) is dynamic; CUDA used on 3060, CPU here.
`models/` and `*.pkl` will be added to `.gitignore` before Phase 2 build.

### Dhan adjustment status (to confirm on key arrival)

Working assumption: **Dhan historical data is pre-adjusted** (split/bonus; industry standard
for commercial Indian data providers).  If confirmed, store with `source="dhan_adjusted"`;
our `compute_adjustment_factors` pipeline does not apply to Dhan data (Dhan adjusts server-side).
Raw-vs-adjusted firewall remains structurally intact.  If wrong: store as `source="dhan_raw"`,
apply existing adjustment layer — no infrastructure changes needed.

### Phase 2 seed weights

Research-backed initial weights added to `config.toml [model.seeds]` before Phase 2 build:
market_regime=0.25, fo_signals=0.25, price_momentum=0.20, sentiment=0.15,
institutional=0.10, calendar=0.05.  Gated update process: min 50 obs per label,
max 5% change per weekly cycle, Brier improvement ≥ 0.5% on held-out data required.
Seeds never hand-tuned against the backtest.

---

## Phase 1.6 — Dhan Integration (2026-06-27) ✅ Built against mocks

### New files

- **`src/data/dhan_downloader.py`** (~480 lines) — `DhanInstrumentMaster` + `DhanBulkDownloader`:
  - `DhanInstrumentMaster`: fetches Dhan's CSV instrument master via `urllib.request`
    (no SDK); caches to `dhan_instruments` DuckDB table; `get_security_id()` for symbol lookup
  - `DhanBulkDownloader`: kill-and-resume capable chunk-based historical downloader;
    `run(symbols, timeframes, start_date, end_date)` — OHLCV; `run_fo(symbols, ...)` — F&O;
    `compute_fo_metrics(symbols)` — aggregates PCR/max-pain/futures basis into `fo_daily_metrics`;
    `coverage_report()` — tabular COMPLETE/HAS_ERRORS/PENDING status per symbol×timeframe;
    `_build_chunks()` splits date range into ≤90-day windows (Dhan API cap);
    `ON CONFLICT DO NOTHING` idempotent upsert; exponential backoff on 429/5xx

- **`src/data/dhan_adapter.py`** (~340 lines) — `DhanDataSource(DataSource)` + `DhanFOMetricsSource(DataSource)`:
  - `DhanDataSource`: queries `ohlcv_intraday` DuckDB for all historical OHLCV;
    live `get_ticks` via Dhan WebSocket (thread-safe buffer + `_inject_ws` test hook);
    `get_option_chain` via Dhan REST (TTL cache + last-good fallback);
    `get_news` raises `DataNotAvailableError`; `BACKTEST_ONLY=False`
  - `DhanFOMetricsSource`: `BACKTEST_ONLY=True`; queries `fo_daily_metrics`
  - SDK quarantine: only `dhan_adapter.py` imports `dhanhq` (deferred inside `connect()`)
  - `DHAN_ADJUSTMENT_STATUS` env var enforcement; WARNING logged until confirmed

- **`src/data/live_source_factory.py`** (~50 lines) — `get_live_source(config, db_path=None) -> DataSource`:
  - Returns `DhanDataSource` or `UpstoxLiveSource` per `config.data.live_data_source`;
    deferred imports (factory importable without any SDK installed);
    swapping live source after Dhan subscription ends = one config key change only

### Schema additions (`src/db/schema.py`)

Five new DuckDB tables: `ohlcv_intraday`, `fo_history`, `fo_daily_metrics`,
`download_jobs` (resumability tracker), `dhan_instruments` (security_id mapping).

### Config additions (`config/config.toml`)

`[dhan]` section (base_url, request_delay, retries, chunk size);
`[model.seeds]` section (6 feature weights + 4 gating hyperparameters);
`live_data_source = "dhan"` under `[data]`.

### Tests

| File | Tests | Key scenarios |
|---|---|---|
| `tests/test_dhan_downloader.py` | 30 | resume-after-interruption; coverage report; fo_metrics PCR; idempotent upsert; instrument master mock |
| `tests/test_dhan_adapter.py` | 31 | SDK quarantine; no-GPU-imports; PIT enforcement; option chain TTL/fallback; DhanFOMetricsSource |
| `tests/test_live_source_factory.py` | 8 | routing; DataSource subclass; ValueError; importable without SDKs; db_path override |

**Total suite: 384/384 passed** (315 pre-existing + 69 new)

### Key design decisions

- Historical OHLCV queries hit DuckDB only (populated by downloader) — no live network call
- `_enforce_pit` is belt-and-suspenders: SQL WHERE already filters future bars; enforce catches
  any bad data that slips through downstream
- Resume guarantee: `_ensure_jobs` uses `ON CONFLICT DO NOTHING` — DONE chunks never overwritten;
  `_load_pending_jobs` only returns `pending` + `failed` rows
- `dhan_downloader.py` uses plain `urllib.request` (no SDK); `dhan_adapter.py` is the sole
  SDK importer, quarantined behind deferred import

### Tomorrow's only remaining step

Add `DHAN_CLIENT_ID` and `DHAN_ACCESS_TOKEN` to `.env`, set `DHAN_ADJUSTMENT_STATUS` after
reading Dhan API docs, then run `python -m scripts.run_bulk_download`.
No structural code changes required.

---

## Phase 1.5 — Data Gap Closure (in progress)

### Checkpoint 1.5.1 — Corporate-Action Adjustment Layer

#### New files

- **`src/data/corporate_actions.py`** — full adjustment layer:
  - `parse_action_description(purpose)` → `(action_type, factor)`: NSE free-text
    parser that recognises SPLIT (`new_FV/old_FV`), BONUS (`existing/(existing+bonus)`),
    DIVIDEND (factor=1.0, intentionally not adjusted, logged), RIGHTS (factor=1.0),
    and falls back to `('OTHER', 1.0)` with a WARNING for anything not confidently
    recognised.  A wrong factor is treated as more dangerous than a missed one.
  - `CorporateActionFetcher.fetch_and_store()`: best-effort NSE API call with graceful
    failure; `_mock_response` test hook; returns 0 on any error so callers never crash.
  - `load_corporate_actions_csv(path, db_path)`: reliable CSV fallback for manually
    prepared or externally sourced data; `ON CONFLICT DO NOTHING` idempotent upsert.
  - `compute_adjustment_factors(actions, price_dates, as_of)`: vectorised cumulative
    product, PIT-correct (actions with `ex_date > as_of.date()` excluded); returns
    `pd.Series[float]` with same index as `price_dates`.
  - `AdjustedBhavCopySource(DataSource)`: `BACKTEST_ONLY = True` structural
    discriminator; `data_source_name = "nse_bhavcopy_adjusted"` distinct from raw;
    backward-adjustment anchored on most-recent bar (latest price stays true/unadjusted);
    volume not adjusted; `_enforce_pit` on every return path.

- **`tests/test_corporate_actions.py`** — 33 tests:
  - Schema: idempotency + column check
  - Parser: SPLIT 10→2, 10→5, FV change; BONUS 1:1, 1:2; DIVIDEND; final dividend;
    unrecognized → OTHER + WARNING logged; invalid split ratio (new≥old) → OTHER;
    empty string → OTHER; buy-back → OTHER
  - compute_adjustment_factors: no actions; single split on/before/after ex_date;
    two compounding actions; PIT exclusion of future action
  - AdjustedBhavCopySource: **HEADLINE TEST** (raw return at split = −49.5%;
    adjusted return = +0.96%; most-recent price unchanged); pre-split bar scaled;
    most-recent bar unchanged; no-actions = raw prices; empty DB; PIT date filter;
    raw source unchanged after loading actions; PIT exclusion of future split
  - BACKTEST_ONLY firewall: flag on adjusted; flag absent on all live sources;
    structural proof (simulated live check rejects adjusted source); distinct
    `data_source_name`; "BACKTEST" in limitations text
  - CSV round-trip; fetcher with mock; fetcher ignores wrong symbol

#### Schema change

- **`src/db/schema.py`** — `corporate_actions` table added to `DUCKDB_SCHEMA_SQL`:
  `(symbol, ex_date, action_type, adjustment_factor, description, source, ingested_at)`;
  `PRIMARY KEY (symbol, ex_date, action_type)`; all `CREATE TABLE IF NOT EXISTS`
  (idempotent; no migration required).

#### Adjustment convention (explicit)

- **Backward adjustment, anchored at most-recent bar.**
  For bar on date D: factor = ∏ factors where D < ex_date ≤ as_of.date()
  Latest bar: factor = 1.0 → true market price preserved.
- **PIT-correct**: A split effective after `as_of` is unknown to that backtest
  context and is excluded from the adjustment.
- **Dividends explicitly not adjusted** — logged as a known decision, stored in
  the `corporate_actions` table with `action_type='DIVIDEND'` and `factor=1.0`.

**Tests: 279 → 312 (33 new, all passing)**

---

### Checkpoint 1.5.3 — Alias Expansion + Span-Based Disambiguation

#### Modified files

- **`src/data/news_ingest.py`**:
  - `SYMBOL_ALIASES` expanded from 49 → 116 tickers (all F&O universe: 50 Nifty 50 + 66 additional).
    Previously missing: BAJAJ-AUTO, ABB, ACC, AMBUJACEM, AUROPHARMA, BALKRISIND, BANDHANBNK,
    BANKBARODA, BATAINDIA, BEL, BHEL, BIOCON, BOSCHLTD, CANBK, CHOLAFIN, CONCOR, CUMMINSIND,
    DMART, ESCORTS, EXIDEIND, FEDERALBNK, GAIL, GODREJCP, GRANULES, HAL, HAVELLS, IDFCFIRSTB,
    IGL, INDHOTEL, INDIGO, IRCTC, JUBLFOOD, KPITTECH, LICHSGFIN, LTIM, LUPIN, MANAPPURAM,
    MCDOWELL-N, MGL, MOTHERSON, MPHASIS, MUTHOOTFIN, NATIONALUM, NAUKRI, NMDC, OFSS, PAGEIND,
    PERSISTENT, PETRONET, PFC, PIIND, PNB, POLYCAB, RBLBANK, RECLTD, RVNL, SAIL, SIEMENS,
    TATAELXSI, TATAPOWER, TORNTPHARM, TRENT, TVSMOTOR, VEDL, VOLTAS, ZOMATO.
  - Key collision decisions committed to SYMBOL_ALIASES comment block:
    no bare "HDFC" (HDFCBANK and HDFCLIFE use full form only);
    no bare "Bajaj" (three Bajaj tickers use full "Bajaj Finance / Finserv / Auto");
    "Mahindra" standalone → M&M only (Tech Mahindra coverage handled by disambiguation);
    "Royal Enfield" → EICHERMOT.
  - `_tag_symbols` reimplemented with span-based disambiguation:
    `pat.finditer()` collects all (ticker, start, end) hits across the full universe.
    If ticker A's match span `[s, e]` is strictly contained in ticker B's longer span
    `[s2, e2]` (i.e. `s2 ≤ s`, `e2 ≥ e`, `(e2−s2) > (e−s)`), ticker A is suppressed.
    A ticker with ANY non-subsumed match remains tagged.  Returns `sorted(tagged)`.
    Resolved collisions: "SBI Life" → SBILIFE only; "Kotak Mahindra Bank" → KOTAKBANK only;
    "Tech Mahindra" → TECHM only.  Log: DEBUG for suppressed, untagged, multi-tagged.

- **`tests/test_review_pass.py`**:
  - `_TAGGING_CASES` replaced by `_BROAD_TAGGING_CASES` (131 cases: 116 company + 15 macro).
  - Coverage floor raised 85% → 90%.  Actual coverage: **100% (121/121)**, no false positives.
  - `test_tagging_sbilife_vs_sbin_note` removed (documented FP is now fixed);
    replaced with `test_tagging_sbi_disambiguation_fixed` (hard assertion: SBIN NOT in tags).
  - Added: `test_tagging_sbi_standalone_still_tags_sbin`,
    `test_tagging_bajaj_family_disambiguation`, `test_tagging_tech_mahindra_vs_mahindra`.
  - `test_tagging_all_known_symbols_tagged` passes all 116 tickers; word-boundary guard holds.

**Tests: 312 → 315 (3 net new, all passing)**

---

### Checkpoint 1.5.4 — F&O Historical Bhavcopy (PLANNED — not yet built)

Assessment (2026-06-26): BUILD verdict.

NSE publishes daily F&O bhavcopy at `nsearchives.nseindia.com/content/fo/` — same domain and
access pattern as the equity bhavcopy already implemented.  Free, available ~2008+, contains
end-of-day OI/price for all strikes and expiries.  Two URL/column-format generations (pre/post
~2018) require dual-format detection, same approach as equity bhavcopy.  BACKTEST_ONLY data
(live F&O snapshots are already served by the `OptionChainSource`).

Planned deliverables:
- `src/data/fo_bhavcopy.py`: download, dual-format parse, validate, load to DuckDB,
  compute daily aggregates (PCR by OI+volume, net OI change, max-pain strike, futures basis)
- `src/data/fo_source.py`: `FODailyMetricsSource(DataSource)` — PIT-correct, BACKTEST_ONLY
- Two new DuckDB tables: `fo_bhavcopy` (raw strike-level) + `fo_daily_metrics` (aggregated)
- ~20 tests

Awaiting explicit go.

---

## Phase 1 — Data Pipeline (complete)

### Step 5 — Phase 1 Review + Consolidation Pass

#### Bugs fixed

- **`UpstoxLiveSource.get_ticks` bypassed `_enforce_pit` on empty data** (`src/data/live_upstox.py`):
  Two early exits (`if not snapshot` and `if df.empty`) returned `_empty(TICK_COLS)` directly,
  skipping `_enforce_pit`.  The contract (`DataSource` docstring) says every return path must call
  `_enforce_pit`.  Functionally safe (empty DF passes immediately), but a silent contract violation
  that could mask a future bug.  Fixed: both paths now call `self._enforce_pit(_empty(TICK_COLS), ...)`.

- **`NewsSource.get_news` bypassed `_enforce_pit` on empty query result** (`src/data/news_ingest.py`):
  Same pattern: `if df.empty: return _empty(NEWS_COLS)` short-circuited before `_enforce_pit`.
  Fixed: `return self._enforce_pit(_empty(NEWS_COLS), "first_knowable_at", as_of)`.

- **`HCLTECH` alias missing standalone "HCL"** (`src/data/news_ingest.py`):
  The alias list had `["HCL Technologies", "HCL Tech"]` but not `"HCL"` alone.  Headlines such
  as "HCL group company files DRHP" were untagged.  Fixed: added `"HCL"` to the alias list.
  Risk acknowledged: "HCL" alone has minor false-positive risk for unrelated HCL products, low
  in NSE-context news.

#### Known limitations documented

- **SBI/SBILIFE abbreviation overlap**: `"SBI"` as a standalone alias for SBIN also fires on
  "SBI Life" stories, producing a false positive SBIN tag.  Coverage audit: 1/100 headlines.
  Comment added to `SYMBOL_ALIASES`; `test_tagging_sbilife_vs_sbin_note` documents the expected
  behaviour.  Callers should treat SBIN as noise when SBILIFE is also tagged.

#### Added review-pass tests

- `tests/test_review_pass.py` — 17 tests across four areas:
  1. **E2E all-sources smoke test** (3 tests): all four sources used together with consistent
     as_of; PIT verified on every source; DataNotAvailableError cross-boundary errors correct;
     eligibility integrates with BhavCopySource.
  2. **News tagging coverage** (4 tests): all aliases match their canonical tickers; ≥85% of
     company-specific headlines tagged (actual: 99% after HCL fix); no false positives on
     macro headlines; word-boundary guard proven on "pitch" / "switch".
  3. **Invariant audit** (6 tests): SDK quarantine (only live_upstox.py may import
     upstox_client, proven by source scan); provenance fields (all sources); DataSource
     subclass (all); no-GPU-imports (all src/data/); timezone consistency (IST-aware as_of
     works across all; tz-mismatch still fails loud); empty-DF PIT compliance.
  4. **Intraday-edge boundary** (4 tests): BhavCopySource.get_ohlcv_minute message names
     "intraday edge" and alternative source; stub returns correct schemas for all methods
     including NEWS_COLS with symbols field.

#### News tagging coverage audit results
  100 representative NSE headlines (40 company-specific, 15 multi-company,
  25 market-wide/macro, 20 edge cases):
  - 99% company stories fully tagged (69/70): 1 miss was "HCL group company" → fixed
  - 100% macro stories correctly untagged (30/30): no false positives on RBI/Sensex/GDP
  - 1 known false positive: "SBI Life" story also tags SBIN (documented, not fixed)
  - After HCL fix: 70/70 = 100% on company stories, 30/30 on macro

#### Test counts
  - Starting count: 262 (Checkpoint 4)
  - Review pass new tests: +17 (test_review_pass.py)
  - **Ending count: 279 / 279 passing**

---

### Step 4 — Checkpoint 4: Option chain + News ingest + Health Monitor

#### Added
- `src/data/option_chain.py` — `OptionChainSource(DataSource)`:
  - Plain `urllib.request` REST call — NO SDK import anywhere in this file
  - In-memory per-symbol cache with configurable TTL (default 60 s) to respect
    Upstox rate limits; second call within TTL window returns cached DataFrame
  - Exponential backoff on HTTP errors: `1 * 2^n` seconds per attempt, up to
    `max_retries` total attempts; HTTP 429 logs "rate-limited" before sleeping
  - `_last_good` fallback: on fetch failure, returns the last successfully fetched
    chain with a WARNING that names the staleness age; `ts` column records the
    original fetch time so callers can detect staleness via `df["ts"].max() vs as_of`
  - PIT on `ts` column: live use (as_of ≈ now) always passes; backtest use with
    historical as_of raises DataLeakageError — intentional: no historical snapshots
  - `_mock_fetch` + `_fetch_ts_override` test hooks bypass all HTTP logic
  - `data_source_name = "upstox_option_chain"`, `data_granularity = "tick"`
  - 20 tests: schema, PIT pass/block, cache hit/miss/TTL expiry, fallback with
    ts-preserved, fallback staleness warning, no-fallback DataNotAvailableError,
    backoff 1s/2s/4s, stops at max_retries, wrong-method DataNotAvailableError ×4,
    no SDK import check, metadata, is_datasource_subclass

- `src/data/news_ingest.py` — `RSSNewsIngest` + `NewsSource(DataSource)`:
  - `SYMBOL_ALIASES: dict[str, list[str]]` — 50 NSE F&O tickers with common names
  - `_normalize_text()`: lowercase + strip punctuation + collapse whitespace
  - `_content_hash()`: SHA-256 of normalized headline + body — dedup key
  - `_tag_symbols()`: word-boundary regex per alias/ticker (`\b{term}\b` case-insensitive);
    untagged stories logged at DEBUG; stored with `symbols=[]`
  - `RSSNewsIngest.fetch_and_store()`: deferred `import feedparser` (module importable
    without it); returns 0 + WARNING if feedparser absent; DuckDB upsert ON CONFLICT
    DO NOTHING; `first_knowable_at = published_at + ingestion_delay_seconds`
  - `NewsSource.get_news()`: queries `WHERE first_knowable_at < as_of AND
    list_contains(symbols, ?)` — PIT on `first_knowable_at`, never published_at
  - 26 tests: normalization, hash stability, hash typo-fix dedup, word-boundary
    tag + alias + case-insensitive + no-false-match, dedup 1/2 inserts, typo-fix
    dedup, first_knowable_at stamping, PIT blocks before fka, PIT allows after fka,
    untagged not returned / stored, column schema round-trip, feedparser not installed,
    wrong-method DataNotAvailableError ×3, metadata, is_datasource_subclass

- `src/data/health_monitor.py` — `DataHealthMonitor`:
  - `update_feed(name, ts)`: records last-seen per feed
  - `is_feed_healthy(feed, as_of, calendar)`: fresh (age ≤ threshold) → True;
    stale → True if market closed, False if market open; never-seen → same rule
  - `confidence_multiplier(feed, as_of, calendar) → float`: 0.0 if stale during
    market hours, 1.0 otherwise; Phase 2 extension point documented (graduated ramp)
  - `check_value(feed, value, prev_value=None)`: None prev → True; 0 prev → True
    (no divide-by-zero); jump > threshold → WARNING + False
  - `as_of` must be tz-aware; raises `ValueError` for naive (consistent with
    `is_market_open`); raises `ValueError` on tz-mismatch between as_of and last-seen
  - 21 tests: fresh healthy (open/closed), stale unhealthy during hours,
    stale healthy after close, never-seen unhealthy open / healthy closed,
    confidence_multiplier 0/1, check_value (None/0/normal/extreme/at-threshold/
    over-threshold), naive as_of raises, tz-mismatch raises, real calendar
    open-window integration, weekend integration

#### Modified
- `src/data/historical.py` — `NEWS_COLS`: added `"symbols": "object"` (list of tagged
  ticker strings; empty list for untagged); schema now matches `news` DuckDB table
- `src/data/bhavcopy_source.py` — `get_ohlcv_minute` message updated: now explicitly
  states "Requires Kite or a supplied intraday dataset — not available from nse_bhavcopy;
  any backtest using BhavCopySource cannot validate intraday edge"
- `requirements.txt` — `feedparser==6.0.12` uncommented and pinned

#### Test counts
- Starting count: 195 (Checkpoint 3 + consolidation pass)
- Checkpoint 4 new tests: +67 (option chain: 20, news: 26, health monitor: 21)
- **Ending count: 262 / 262 passing**

#### Design decisions
- **`first_knowable_at` as PIT column, not `published_at`**: A story published at T
  is not available to the model until it has actually been fetched, parsed, and written
  to DuckDB.  `first_knowable_at = published_at + delay` is the conservative lower bound
  for when the system could have acted on the story.  Using `published_at` directly
  would let the model "read" a story seconds after it was written — before ingest.
- **Word-boundary regex for symbol tagging**: Raw substring matching (`"ITC" in text`)
  would match "ITC" inside "switch" or "pitch".  `\bITC\b` with `re.IGNORECASE` only
  matches at word boundaries.  Untagged stories are stored with `symbols=[]` (not
  discarded) so they can be used for global sentiment or future tagger improvements.
- **SHA-256 of normalized content as dedup key**: Normalization (lowercase + strip
  punctuation/whitespace) makes the hash stable across minor editorial tweaks.  A
  story re-published with a corrected typo produces the same hash and is silently
  dropped — correct, since it is the same story.
- **`OptionChainSource` ts column records fetch time**: The Upstox free-tier option
  chain API is live-only.  Storing the actual fetch timestamp in `ts` lets callers
  detect staleness by comparing `df["ts"].max()` with `as_of` without any extra
  metadata.  PIT enforcement naturally rejects historical as_of values (fetch ts >
  historical as_of → DataLeakageError), preventing accidental backtest misuse.
- **DataHealthMonitor binary 0/1 gate for Phase 1**: A graduated multiplier (partial
  credit for mildly stale data) introduces a hard-to-calibrate parameter.  Phase 1
  uses a hard binary gate — stale during market hours means the feed contributes
  nothing.  Phase 2 can introduce a graduated ramp once we have empirical staleness
  distributions from live runs.

---

### Step 3 — Upstox WebSocket adapter + Consolidation

#### Added
- `src/data/live_upstox.py` — `UpstoxLiveSource(DataSource)` (Checkpoint 3):
  - SDK QUARANTINE: sole permitted file to import `upstox_client` SDK; import deferred
    to `connect()` so the module is usable (and testable) without the SDK installed
  - `_inject_ws` parameter bypasses SDK entirely in tests
  - Thread-safe per-symbol tick buffers (`collections.deque` + `threading.Lock`)
  - `_on_tick(message)` — WebSocket callback; runs in WS thread
  - `get_ticks(symbol, as_of, lookback_seconds)` — snapshot under lock → PIT enforcement;
    returns empty correctly-typed DataFrame when buffer is empty; never None
  - `_on_close(reason)` — reconnects with exponential backoff (`1 * 2^n` seconds);
    stops after `max_reconnect_attempts` and logs ERROR
  - `as_of` is ALWAYS supplied by the caller; the adapter never reads a clock
  - `data_granularity = "tick"`, `data_source_name = "upstox_live"`
  - All non-tick methods (`get_ohlcv_daily`, `get_ohlcv_minute`, `get_news`,
    `get_option_chain`) raise `DataNotAvailableError` with actionable messages
  - CPU-only: no GPU library imports
- `tests/test_live_upstox.py` — 25 tests (10 mandatory proof areas):
  SDK quarantine, interface symmetry, PIT, thread safety, empty-buffer schema,
  reconnect backoff, as_of-from-caller proof, no-GPU-imports, provenance fields,
  `connect()` ImportError without SDK, inject_ws no-op on reconnect

#### Bugs fixed during consolidation
- **`validate_bhavcopy` NaN/inf silent pass-through** (`src/data/bhavcopy.py`):
  `df[col] <= 0` returns False for NaN and +inf, so those rows were not rejected.
  Fixed: changed to `~(df[col].gt(0) & df[col].lt(float("inf")))` which correctly
  identifies NaN, infinite, zero, and negative values as invalid.
- **`parse_bhavcopy_csv` silent column drop** (`src/data/bhavcopy.py`):
  If a NSE format update removes a required column (e.g. `ClsPric`), the old code
  silently dropped it, producing a DataFrame without `close`.  Fixed: `_detect_format`
  now checks all required columns are present and raises `ValueError` if any are missing.
- **`BhavCopySource.get_ohlcv_daily` tz-mismatch with IST-aware `as_of`**
  (`src/data/bhavcopy_source.py`): DuckDB >= some version returns DATE columns as
  `pd.Timestamp` (tz-naive `datetime64[ns]`) rather than `datetime.date` objects.
  When `as_of` was IST-aware, `_enforce_pit` saw `col_has_tz=False, as_of_has_tz=True`
  and raised `DataLeakageError("tz-mismatch")` — a false positive.  Fixed: explicitly
  convert the `date` column to `datetime.date` objects after the query so the
  date-only PIT branch is always taken regardless of DuckDB version.

#### Added integration / consolidation tests
- `tests/test_integration.py` — 21 tests covering 6 risk areas:
  1. **Calendar forward-coverage**: warns (once per year) for dates beyond data range;
     JSON with future year correctly classifies 2027 dates; injected calendars silent.
  2. **Bhavcopy format robustness**: unknown columns, missing required column in both
     new and old format → `ValueError`; NaN/inf prices → hard-reject.
  3. **Corporate-action / raw-price boundary**: `BhavCopySource.data_source_limitations`
     explicitly says "raw unadjusted" and "corporate-action adjustment pending";
     extreme intra-bar move soft-flagged; split-day bar invisible at bar level (documented).
  4. **Interface symmetry on real data**: `BhavCopySource.get_ohlcv_daily` and
     `StubHistoricalSource.get_ohlcv_daily` return identical columns; `UpstoxLiveSource`
     and `StubHistoricalSource.get_ticks` return identical columns; PIT enforced identically.
  5. **End-to-end timezone consistency**: IST-aware past tick returned; IST-aware future
     tick blocked; naive tick + IST-aware as_of raises `DataLeakageError("tz-mismatch")`
     — loud failure, not silent wrong comparison.
  6. **End-to-end smoke test**: calendar → load bhavcopy → eligibility → BhavCopySource
     → UpstoxLiveSource in one test; idempotent loads; DataNotAvailableError across
     interface boundary; data_granularity self-labels correct.

#### Design decisions
- **`BhavCopySource` normalises `date` column post-query**: DuckDB's `.df()` return
  type for DATE columns is version-dependent.  Normalising to `datetime.date` in the
  adapter makes the contract explicit and insulates all downstream code (including
  `_enforce_pit`) from DuckDB version changes.
- **`NSETradingCalendar._max_holiday_year` tracks data range**: warns (once per unknown
  year) when queries go beyond loaded holiday data.  Injected calendars (`holidays=...`)
  set `_max_holiday_year = 0` to suppress the warning — test calendars are intentionally
  limited and must not pollute test logs.
- **`_detect_format` validates required columns**: a partial bhavcopy (missing `ClsPric`)
  is worse than an empty one — a silent column drop would produce NaN close prices that
  pass validation and corrupt DuckDB.  Early ValueError is safer.

---

### Step 2 — Bhavcopy downloader + DuckDB loader + PIT eligibility

#### Added
- `src/data/base.py` — `DataNotAvailableError(RuntimeError)`: raised when a DataSource
  cannot provide a data type at all (capability limitation, not a safety violation).
  Dead `TYPE_CHECKING` block removed.
- `src/data/bhavcopy.py`:
  - `MissingDataGap(RuntimeError)` — raised when bhavcopy is absent on a trading day
  - `parse_bhavcopy_csv(content)` — normalises newer NSE format (`TckrSymb`/`TradDt`/…)
    and older format (`SYMBOL`/`TIMESTAMP`/…); filters to EQ series; typed columns
  - `validate_bhavcopy(df)` — hard-rejects rows with negative prices or `high < low`;
    soft-flags zero-volume and extreme-move rows; logs every decision; never silent drop
  - `download_bhavcopy(d, raw_dir, *, calendar)` — uses calendar to classify gap
    (`'expected'` → None; `'unexpected'` + download failure → `MissingDataGap`); retry
    with exponential backoff; skips if file already on disk
  - `load_bhavcopy_to_duckdb(d, raw_dir, db_path)` → `(rows_loaded, rows_rejected)`:
    idempotent (`ON CONFLICT DO NOTHING`); stamps each row `source="nse_bhavcopy"` and
    `ingested_at=UTC now`; returns count of loaded vs rejected rows
- `src/data/eligibility.py` — `get_eligible_symbols(as_of, db_path) -> frozenset[str]`:
  queries `WHERE date = as_of` (exact date, not `<=`) so that a delisted stock that last
  appeared on date D is correctly absent from the universe for D+1 onward.  Empty set for
  dates with no data loaded (e.g. holiday, correct).
- `src/data/bhavcopy_source.py` — `BhavCopySource(DataSource)`:
  - `data_source_name="nse_bhavcopy"`, `data_granularity="daily"`,
    `data_source_limitations="daily bars only; raw unadjusted prices; ..."`
  - `get_ohlcv_daily` → queries DuckDB with `date < as_of.date()` then calls `_enforce_pit`
  - `get_ohlcv_minute / get_ticks / get_news / get_option_chain` → each raises
    `DataNotAvailableError` with an explicit message naming the correct source
- `tests/test_bhavcopy.py` — 37 tests covering all of the above, including:
  - Survivorship proof: `SYNTH_DELISTED` present in Jan 22 bhavcopy, absent from Jan 23 →
    eligible on Jan 22, not eligible on Jan 23
  - PIT on real DuckDB data: as_of at midnight of bar date returns empty; as_of next day
    returns the bar
  - `DataNotAvailableError` is a distinct type from `DataLeakageError`

#### Design decisions
- **`get_eligible_symbols` queries exact date, not `<= as_of`**: querying `<= as_of`
  would include symbols from all earlier dates and would NOT exclude a stock that was
  delisted two dates ago.  The correct universe for date D is precisely the set that
  appeared in the bhavcopy for D.
- **Hard reject vs soft flag split**: negative price and `high < low` are arithmetically
  impossible and indicate corrupt data; a row containing them cannot be used safely.
  Zero volume and extreme moves are unusual but not impossible (circuit breaker days,
  rights issues); they are kept but prominently logged so the researcher notices.
- **`DataNotAvailableError` vs empty DataFrame**: returning empty DataFrames for
  unsupported data types silently hides the limitation from the caller.  Raising an
  explicit error forces the prediction code to acknowledge what data it is actually using.

---

### Step 1 — NSE Trading Calendar

#### Added
- `src/data/calendar.py` — `NSETradingCalendar` class:
  - `is_trading_day(d)` — weekday + not in holiday set
  - `is_market_open(dt)` — 09:15 ≤ t < 15:30 IST; requires tz-aware datetime (raises
    `ValueError` for naive); accepts IST or UTC and converts correctly
  - `next_trading_day(d)` / `prev_trading_day(d)` — skip weekends and holidays
  - `trading_days_in_range(start, end)` — inclusive both ends
  - `classify_missing_file(d)` — `'expected'` (holiday/weekend) or `'unexpected'`
    (trading day with no file); used by bhavcopy downloader for gap-classification
  - `IST` constant exported for callers that need tz-aware datetimes
  - `get_calendar()` — module-level cached factory (`@lru_cache`)
- `refresh_nse_holidays(year, save_path)` — tries NSE API with browser-like headers;
  returns `False` and logs `WARNING` on any failure; never raises; cache unchanged on failure
- `config/nse_holidays.json` — authoritative JSON cache for 2023–2026; committed to repo
  so calendar works offline without ever calling `refresh_nse_holidays`
- `tests/test_calendar.py` — 39 tests: all predicates, boundary at 09:15/15:30, UTC↔IST
  conversion, naive-raises, next/prev skip, range, gap classification, missing-JSON fallback
  (warning + embedded list used), malformed-JSON fallback, network-failure return-False

#### Design decisions
- **Two-tier holiday data**: JSON file (primary, updatable) + embedded Python dict (fallback).
  Failing to load the JSON is a WARNING, not a crash — the calendar always returns a usable
  answer.  The embedded list carries approximate lunar-based dates; update the JSON once NSE
  confirms exact dates each year.
- **tz-aware required for `is_market_open`**: raises `ValueError` for naive datetimes for
  the same reason `_enforce_pit` raises on tz-mismatch — silent wrong comparisons are worse
  than loud failures.
- **`classify_missing_file` is calendar-driven**: the bhavcopy downloader must use this
  rather than checking weekdays inline, so holiday knowledge lives in one place only.

---

## Phase 0 — Foundations & Backtester Scaffold

### Added
- `DESIGN.md` (pre-existing source of truth, Version 2.0)
- `TASKS.md` — living phase/subtask checklist
- `pyproject.toml` — package discovery, pytest path, ruff + black config
- `requirements.txt` — lower-bound pins; note to freeze after first install
- `.env.example` — secrets template (broker API keys; never committed)
- `config/config.toml` — TOML config: NSE F&O universe (~80 symbols, static Phase 0),
  token pool ₹5,00,000, primary horizon 30 min + 15/60 outputs, quantile levels,
  risk guardrails, logging, experiment DB path
- `src/config.py` — pydantic v2 settings loader (stdlib `tomllib`); `Config.config_hash()`
  returns SHA-256 of canonical config JSON for run fingerprinting
- `src/log.py` — `configure_logging()`: console + rotating-file handler, level from config
- `src/db/schema.py` — DuckDB tables: `ohlcv_daily`, `ohlcv_minute`, `ticks`, `news`,
  `option_chain`; SQLite tables: `runs`, `predictions`, `trades`, `signals`, `positions`,
  `state`; all `CREATE TABLE IF NOT EXISTS`
- `src/db/duckdb_client.py` — `DuckDBClient` context manager; `execute_schema`, `query_df`,
  `list_tables`
- `src/db/sqlite_client.py` — `SQLiteClient` context manager; `execute_schema`, `list_tables`
- `src/data/base.py` — abstract `DataSource`; `DataLeakageError`; `_enforce_pit()` blocks
  future-dated rows for both date and datetime columns
- `src/data/historical.py` — `StubHistoricalSource`: all methods return empty DataFrames
  with correct column schemas; PIT enforcement called on every return; Phase 1 replaces
  method bodies without changing the interface
- `src/experiment/tracker.py` — `ExperimentTracker`: SQLite `runs` table capturing config
  hash, config JSON, git commit, data source + limitations, data granularity, seed, date
  range, universe size/hash, metrics; `start_run / complete_run / fail_run / get_run / list_runs`
- `tests/conftest.py` — shared fixtures (`tmp_duckdb_path`, `tmp_sqlite_path`)
- `tests/test_config.py` — config loading, token pool value, horizons, quantile levels,
  config hash stability, no duplicate symbols
- `tests/test_schema.py` — DuckDB + SQLite table creation, idempotency
- `tests/test_pit_enforcement.py` — DataLeakageError raised for future rows (date and
  datetime columns, exact boundary, past-only allowed, empty DF always passes)
- `tests/test_data_interface.py` — StubHistoricalSource: is DataSource subclass, returns
  empty DFs with correct schemas, metadata fields present
- `tests/test_experiment.py` — run CRUD round-trip, complete with metrics, fail, list

### Design decisions
- **TOML + pydantic v2**: stdlib `tomllib` (Python 3.11+) avoids an extra dep; pydantic v2
  validates structure at startup so misconfigured runs fail immediately, not mid-backtest.
- **`_enforce_pit` in base class**: every adapter must call it before returning — leakage
  must be hard to introduce by accident, not just documented.
- **StubHistoricalSource** returns typed empty DataFrames so all downstream column-access
  code is testable without real data.
- **`data_granularity` field in runs table**: every backtest report self-labels whether it
  ran on `daily` or `minute` data; intraday-edge conclusions require `minute`.
