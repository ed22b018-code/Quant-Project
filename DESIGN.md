# DESIGN.md — Indian Market Intraday Prediction & Paper-Trading Engine

**Version 2.0 · Source of truth for the build.**
This document is the authoritative specification. If any instruction conflicts with it, flag the conflict rather than silently choosing. Build strictly in the phase order in Section 16.

---

## 0. What this project is (and is not)

A personal, fully self-hosted, **zero-runtime-cost** intraday research and paper-trading system for NSE/BSE-listed stocks. Each trading day it scans a universe of 100+ liquid stocks, narrows them to a focused watchlist of ~10, forecasts the **probable behaviour** of those stocks over the next 15–60 minutes with **calibrated** confidence, and executes **simulated (paper) trades** on a token pool that mirrors rupees in every way except that no real money is at risk.

**This is a research instrument.** Its purpose is to *honestly determine whether any intraday edge exists, and where* — not to be a guaranteed money-maker. For liquid large-caps, public news is largely priced in within seconds. A rigorous finding of "little to no edge on liquid large-caps, weak regime-dependent edge on mid-cap OI buildup" is a **successful** outcome. A gorgeous backtest curve hiding data leakage is a **failure**.

**Hard boundaries:**
- No real money is ever placed. Paper trading only, for the entire scope of this project.
- The system forecasts a **probability distribution** (drift + range cone + scenarios + levels), never an exact price path.
- Nothing in this system constitutes financial advice.

---

## 1. Core design principles

| Principle | Meaning |
|---|---|
| **Numeric models predict; the LLM only explains** | All forecasts come from trained ML models. The LLM is a read-only narrator that turns computed numbers into plain English. It must never invent a number or override a model. |
| **Forecast distributions, not points** | The exact path is noise. Predict expected drift, a probable range (cone), the likely scenario(s), and the levels price will interact with. |
| **Honesty over optimism** | Benchmark every result against buy-and-hold, random, and always-flat, after costs. Calibrate and verify confidence. Patterns must survive multiple-testing correction. |
| **Leakage is the enemy** | Strict point-in-time data: at prediction time T, only data timestamped `< T` may be used. This single discipline separates a real result from a fictional one. |
| **Regularization, not feature-starvation** | Many features are fine if the model is regularized and validated. The danger is tuning many *free parameters on tiny samples* — so online self-adjustment is small and gated, while training uses the full feature set. |
| **Light to run, heavy only when scheduled** | The live daily system is near-idle. Heavy work (backtest/retrain) runs overnight or weekly. |

---

## 2. Non-negotiable engineering rules

1. **No data leakage, ever.** All features strictly point-in-time. Build a data-access layer that *enforces* `timestamp < T` so leakage is hard to introduce by accident. Never use split/dividend-adjusted prices for live point-in-time features; never join "latest" values without checking when they became knowable.
2. **One shared data interface for live and historical.** The exact same prediction code runs in backtest and live. The backtester is **not** a separate codebase — it is the same pipeline fed historical data through the same interface.
3. **Everything free / open-source / local.** See Section 14 for the stack. Never load full multi-year tick data into RAM — stream/query via DuckDB.
4. **The LLM only explains pre-computed numbers.** It never predicts, never overrides a model. If data is stale/missing it must say "I don't have current data," not confabulate.
5. **Reproducibility.** Fixed random seeds, versioned data snapshots, every run logged with config + result.
6. **Portability.** Pin all dependencies (`requirements.txt` + virtual environment). The project must rebuild on another machine from the repo alone.

---

## 3. The signal set (~15 features, 6 groups)

The model assigns importance during training (regularized); SHAP reveals which features actually drive predictions so dead ones are pruned. **You do not hand-pick weights; the data does, under regularization.**

| Group | Features |
|---|---|
| **Price & microstructure** | distance from VWAP, intraday momentum (rate of change), volume ratio vs 20-day avg, position in opening range / nearest support-resistance, realized volatility (ATR) |
| **F&O** | PCR, change in OI at ATM strikes, distance to max-pain, futures basis (premium/discount) |
| **News & sentiment** | FinBERT sentiment (source-weighted, time-decayed), news volume/count |
| **Market context** | Nifty regime, India VIX level, Gift Nifty premium |
| **Institutional & sector** | FII/DII (previous-day context only — it is EOD data, never a live signal), sector relative strength vs Nifty |
| **Calendar** | time-of-day bucket, event-day flag |

**Safeguard — honest news timestamps:** Deduplicate syndicated copies of one story (treat as one event). Stamp every news item with the most conservative **"first-knowable" time** (not publication time, which can precede exchange dissemination) plus a deliberate ingestion-delay buffer, so nothing trains or trades on news faster than it could realistically have been seen.

---

## 4. Two-stage scanner (100+ → ~25 → 10)

**Stage 1 — cheap screen, all 100+ stocks:** liquidity filter (volume + traded-value floors) + fast numeric features only (volume ratio, VWAP distance, gap, momentum, PCR). No FinBERT, no LLM, no model inference. Runs in milliseconds; GPU not involved. Ranks and drops obvious non-movers.

**Stage 2 — full scoring, ~20–30 survivors:** FinBERT sentiment, full F&O features, the prediction stack, regime throttle, calibrated confidence, meta-model sizing. Final 10 chosen by risk-adjusted net profit %.

**Safeguard — selection bias:** Scoring 100 and keeping the 10 highest-confidence cherry-picks the noisy tail, so top-10 confidence is overstated. Hold the top-10 to a *higher* confidence bar, and validate the **whole pipeline** (scan → pick → trade) end-to-end, measuring realized accuracy of "stocks the scanner selected" — never predictions in isolation. Periodically check the cheap screen isn't discarding stocks that would have been profitable.

---

## 5. The prediction stack — three layers

Output is a **probability cone + ranked named scenarios + key levels**. Every number traces to a model or a deterministic calculation.

**Layer 1 — Quantile cone (LightGBM quantile regression):** predict the 10/25/50/75/90th percentile of price at +15, +30, +60 min. Median = expected drift; outer bands = probable range. *Safeguard:* enforce monotonic ordering of quantiles (no crossing); the cone MUST widen with horizon — a tight 60-min band signals leakage.

**Layer 2 — Scenario classifier (multi-class):** predict the likely behaviour type with probabilities — continuation / mean-reversion to VWAP / range-bound chop / breakout / breakdown-then-bounce / distribution-fade. *Safeguards:* ground-truth labels defined mechanically in advance via the **triple-barrier method** (never eyeballed); **"no trade" is a first-class class** — most 30-min windows are flat noise, so the model is rewarded for staying out rather than forced to pick a direction.

**Layer 3 — Level engine (deterministic, no model):** compute VWAP, opening-range high/low, prior-day high/low, nearest S-R, F&O max-pain, key OI strikes. Forecasts reference these levels.

---

## 6. Empirical pattern library

A library of conditional rules mined from history: *condition → historical outcome distribution*, always with sample size N and confidence interval. Example: "rose >1.5% on >2× avg volume during 9:15–10:00 with no Tier-1/2 news → faded below the 10:00 level by close in 61% of 340 cases (95% CI 56–66%)."

**Mining discipline (mandatory):**
- Pre-register hypotheses that have a plausible mechanism; for any combinatorial search apply **Benjamini–Hochberg FDR** correction.
- Validate each pattern **out-of-sample** (discover on years 1–3, confirm on 4–5).
- Minimum sample floor (hundreds of instances); always report N + CI.
- Compare against the **unconditional base rate** — keep only patterns with real lift.
- Periodically re-test live; **retire decayed patterns**.

Surviving patterns feed the system two ways: as **features** the ML model can weigh, and as **human-readable reasons** the LLM narrator can cite.

---

## 7. Regime / market-overview layer

Classify the day **risk-on / neutral / risk-off** from VIX level + trend, Nifty trend, breadth, Gift Nifty, USD/INR, crude. Use it two ways: (1) a **0–1 multiplier on confidence and especially position size** — in risk-off, size down or go flat regardless of how good a single stock looks; (2) a **feature** the model also sees. A macro shock must *override and throttle*, not merely add a signal. **Safeguard:** report every metric per regime; a result good in only one regime is flagged untrustworthy.

---

## 8. Confidence, calibration & sizing

- **Confidence = the model's predicted probability, then calibrated** with isotonic regression (or Platt scaling) on a validation set, verified with a reliability diagram. A "78%" must be right ~78% of the time. Do **not** define confidence by counting agreeing signals (correlated signals would triple-count one idea).
- **Sizing via meta-labeling:** a primary model decides direction; a second model decides whether/how much to bet given the primary signal + context. Pair with **fractional Kelly or volatility-targeted sizing**.
- **Optimize risk-adjusted return (Sharpe/Sortino), not raw return** — raw-return optimization concentrates, over-leverages, and takes fat-tailed bets.

---

## 9. Self-learning, done safely

Instead of ad-hoc daily weight nudges, do **periodic walk-forward retraining (weekly)**. New weights are accepted **only if they beat the current model on data neither version trained on** — the validation gate is the primary defense.

| Safeguard | Role |
|---|---|
| Validation gate (must beat prior out-of-sample) | A noisy update simply won't pass |
| Shrinkage to research-backed seed weights | Solves cold-start & small-sample; lean on prior early |
| Minimum ~30 observations before updating a signal | No learning from noise |
| Max change cap per update (~15%) | Prevents lurching |
| Time-decay sample weights; monitor gradual live decay | Handles non-stationarity (rule/lot-size/structure changes) |

---

## 10. Two-tier monitoring

- **All watchlist stocks — continuous, lighter:** three tracks — cheap tick-level checks on all, a signal-state sweep every few minutes, and the full 30-min refresh. This is how a *new* opportunity is found or a stock is promoted into a trade.
- **Open positions — continuous, intense, low-latency:** runs exit logic in seconds (target hit, stop approached, confidence decay, new negative signal, time stops). **An open position is monitored even if it drops out of the watchlist** — the exit must still be managed.

**Safeguards:** (1) **Event-priority + idempotency** — when news, a volume spike, and a refresh fire at once for one stock: one prediction per stock per cycle, latest-event-wins, capital locked the moment a signal is committed. (2) **Data Health Monitor** — if any critical feed is >60s stale, flag "data incomplete," drop confidence below the trade threshold, and suppress new entries until it recovers.

---

## 11. LLM narrator (explain-only)

On demand ("what about Infosys right now?"), code gathers the computed state — quantile cone, scenario probabilities, calibrated confidence, SHAP drivers, matched patterns, news, open-position P&L — and the LLM narrates it in detailed plain English (probable path, range, ranked scenarios, key levels, driving indicators). **It cannot invent a number or override a model.** **Anti-confabulation guard:** if the stock is outside the scanned universe or its data is stale, respond "I don't have current data on this" (tied to the same Data Health Monitor). Runs locally via Ollama (Qwen 2.5 7B on GPU; 3B if VRAM-limited; template fallback if needed).

---

## 12. Paper-trading engine

A user-defined **token pool** behaves exactly like rupees for all cost/P&L math; all sizing and risk rules scale to the input pool. Every order passes through an **impact-cost / slippage estimator** — no perfect fills.

**Cost simulation (Zerodha/Upstox-style):** brokerage ₹20/order; STT 0.025% (MIS sell) / 0.1% (CNC sell); exchange txn 0.00345%; GST 18% on brokerage; stamp 0.003% (MIS buy) / 0.015% (CNC buy); slippage modelled per order size — **deliberately pessimistic**. Treat paper P&L as an **upper bound**.

**Hard guardrails (cannot be overridden):** max 10% pool per trade; max 40% total intraday / 50% delivery exposure; min 10% cash reserve; all MIS auto-squared-off by 3:15 PM; pause if pool drops 20% from peak; no single sector >25% exposure; illiquid stocks never traded.

**Safeguards:** (1) **Correlation-aware risk** — cap total correlated (cross-sector beta) exposure, not just per-sector; compute performance stats respecting trade dependence. (2) **Clean failure & recovery** — persist all state (positions, pending signals, pool value) to disk continuously; on restart, reconcile open positions against the live feed before any action.

### 12.1 Capital allocation layer (Phase 4)

**Selection (which stocks) and sizing (how much capital each) are separate problems** solved by different logic. The allocation layer receives a ranked candidate list from the scanner and produces per-position sizes. It does NOT spread capital equally or by raw model confidence.

It sizes each position by **expected after-cost profit, adjusted for risk**, weighing four inputs:

1. **Calibrated confidence, not raw score** — use the post-isotonic-calibration probability (§8). A "78%" must be right ~78% of the time; uncalibrated softmax scores are not suitable for sizing.
2. **Expected net move minus full costs** — estimated directional move × calibrated probability, minus brokerage + STT + slippage. A small high-confidence move that does not clear costs gets zero allocation.
3. **Volatility scaling** — size is inversely proportional to recent realized volatility (ATR). Jumpy stocks receive smaller positions; their execution uncertainty and stop-placement error are higher.
4. **Correlation penalty** — reduce aggregate allocation to stocks that move together (same sector, high cross-beta). Concentrating into five highly-correlated stocks is effectively one position with five times the per-position cost.

**Sizing method:** fractional Kelly (not full Kelly — estimation error on Kelly input makes full Kelly dangerous) combined with volatility targeting — the position size that would contribute a target-sized volatility unit to the portfolio, scaled by the Kelly fraction.

**Validation gate:** test this layer independently of the prediction models. A new allocation method is accepted only if it improves risk-adjusted P&L (Sharpe / Sortino after costs) on unseen data vs. equal-weight and confidence-weight baselines. Prefer a simple, interpretable version over a clever one — complexity here is a direct lever on overfitting.

The hard guardrails in §12 remain non-overridable caps applied AFTER the allocation engine produces its unconstrained sizes.

### 12.2 ExecutionEngine interface — paper now, real money only through a hard gate

All order placement goes through a single **ExecutionEngine** abstract interface (same design pattern as DataSource). Nothing upstream — scanner, signal engine, prediction stack — knows or cares which implementation is active. The paper simulator implements it now (Phase 4); a real-broker adapter can be added later without touching any other part of the system.

```
ExecutionEngine (abstract)
  ├── PaperExecutionEngine     ← implement in Phase 4
  └── BrokerExecutionEngine    ← can be added later; NOT in project scope now
```

**Non-negotiable go-live gate — real money only after all of the following:**

1. At least **60 days of forward paper trading** (not backtest) shows edge consistent with the model's calibrated confidence.
2. A **reliability diagram on live predictions** (not backtest) confirms calibration holds out-of-sample.
3. **Net P&L after full costs beats a "buy Nifty on open, sell at close" passive strategy** during the same 60-day period.
4. **Per-regime analysis shows no systematic failure** — risk-off regime performance must be flat or short (not large drawdown).

If all gates pass, **start at ≈10 % of the paper-trading pool size**. Scale up only as real-money results continue to match paper results within a reasonable tolerance. Any unexplained divergence between real and paper P&L (after accounting for live slippage) triggers a freeze and investigation before scaling further.

> **This gate cannot be skipped, overridden, or shortened.** It is recorded here so it is visible to anyone extending the system, not buried in a comment.

---

## 13. Backtesting & validation (built FIRST — it is the edge-detector)

| Technique | Purpose |
|---|---|
| Strict point-in-time data | #1 defense against fictional results |
| Walk-forward validation | Mirrors how the live system updates |
| Purged k-fold + embargo | Prevents time-series CV leakage from overlapping samples |
| Triple-barrier labeling | Label by which barrier (target/stop/time) hits first |
| Interval coverage / pinball loss | Score the cone honestly — does price land in the 80% band ~80% of the time? |
| Deflated Sharpe ratio | Discounts the best result by number of configs tried |
| Survivorship-free universe | Reconstruct which stocks *were* tradeable each date, incl. later-delisted |
| Corporate-action handling | Adjusted history for backtest; raw for live point-in-time features |
| Ingestion-delay buffer | Never "react" faster than possible live |
| Block-bootstrap by day | Honest CIs respecting autocorrelation (trades aren't independent) |
| Significance over story | Treat "no edge" as the null; require data to reject it |
| Frozen test set (anti human-leakage) | Freeze design on a dev slice; touch the final test set once. Your own eyes are a leakage vector. |
| Always-on benchmarks | Strategy vs buy-and-hold Nifty vs random vs always-flat, after costs |
| Per-regime reporting | A number good in only one regime can't be trusted |
| Reproducibility | Fixed seeds, versioned snapshots, logged runs |

**Reference:** Marcos López de Prado, *Advances in Financial Machine Learning* (triple-barrier, meta-labeling, purged k-fold).

### 13.1 Known loopholes & where they're handled

These are subtle biases that survive even rigorous backtesting if not explicitly wired in. Each is listed with the phase that closes it.

| Loophole | Mechanism | Target phase |
|---|---|---|
| **Feedback contamination ("grading its own homework")** | The scanner selects top-10 from what it finds interesting. Prediction accuracy measured only on selected stocks is circular — accuracy looks good partly because the model chose stocks it was already confident about. **Fix:** at evaluation time, log scored predictions for a small *random* sample of NON-selected eligible symbols each day. Compare to top-10 accuracy; a shrinking gap signals the selection is providing less real information. | Phase 4 evaluation |
| **Multiple-comparisons creep** | Every config tried, threshold tuned, or feature added is an implicit test. Enough tries and something will look significant by chance. **Fix:** maintain a running count of every config evaluated (logged in the runs table). Hold a final test set that is touched exactly once — after all tuning decisions are frozen. Report Deflated Sharpe (already in §13) discounting by number of configs tried. | Phases 2–6, ongoing |
| **Regime-transition blindness** | A model good in steady risk-on may be useless — or actively dangerous — during the 2–3 days around a regime flip. Lumping those transitions with steady-state hides the danger. **Fix:** label each trading day as steady-state or transition-window (the N days before and after a detected regime change); report performance on each subset separately. | Phase 2 + all reporting |
| **Survivorship in our own watchlist** | If evaluation uses "the stocks the scanner historically watched" as the universe, it backdates knowledge of which stocks were interesting — that list implicitly selected on hindsight. **Fix:** every evaluation path must reconstruct the full eligible universe for each date using the bhavcopy-derived eligibility set (Phase 1), never the live watchlist. | Phase 4 |
| **Label-window leakage** | Triple-barrier labels span a forward window. Any CV split that lets future windows overlap with a training sample, or whose embargo is shorter than the label horizon, trains partially on its own target. **Fix:** apply López de Prado purging (remove training samples whose forward window overlaps the test period) and embargo (additional gap equal to at least one label horizon) specifically to triple-barrier labeled samples. | Phase 2 |
| **Cost-model drift** | Brokerage rates, STT, and typical slippage change over time. A cost model calibrated in 2022 underestimates current costs, inflating simulated P&L. **Fix:** all cost parameters live in dated config (not hard-coded), tagged with the date they were last verified. Cost-regime changes are logged and reported; runs compare net P&L under both the contemporaneous and current cost params. | Phase 4 |
| **Subtly-wrong (not just stale) feed values** | A stale feed is obvious (timestamp check). A feed that silently delivers a wrong value — a DDE glitch, an off-by-a-decimal-place error, a zero-volume bar on an active stock — is harder to detect and can corrupt features without triggering any staleness alert. **Fix:** value sanity checks at ingestion time (negative prices, high < low, zero volume on an active trading day, single-bar move beyond a threshold); log and flag every rejected row; cross-check key high-frequency symbols against a second source periodically. | Phase 1 (health monitor) + Phase 4 |

> **Rule:** do not implement future-phase loophole mitigations in an earlier phase. Record them here so they are not forgotten, and close each one in its designated phase.

---

## 14. Free tech stack (target: RTX 3060 / Ryzen 9 / 16GB RAM)

- **Language/tooling:** Python 3.11+, pytest, ruff + black, MLflow (or a SQLite table) for experiment tracking.
- **Sentiment:** FinBERT (`ProsusAI/finbert`) via HuggingFace transformers + torch (CUDA/GPU).
- **Models:** LightGBM (quantile + scenario + meta-label), scikit-learn (isotonic calibration), SHAP.
- **Pattern library:** pure Python + DuckDB.
- **LLM narrator:** Ollama + Qwen 2.5 7B (3B fallback / template fallback).
- **Storage:** DuckDB (larger-than-RAM analytical queries off disk) + SQLite (state/logs).
- **Dashboard:** Streamlit (local).
- **Live data:** broker free tier (Upstox/Dhan) for ticks + option chain; RSS for news; free-tier macro APIs. Cache aggressively (e.g. option chain every 60s) to respect rate limits.

**Data hygiene at the edges:** normalize all timestamps to a single timezone (handles India-vs-global DST mismatch); drive sessions by a real **NSE trading calendar** (holidays, special timings, muhurat), not weekday assumptions; exclude the pre-open auction (9:00–9:15) from continuous-market tick features; exponential backoff + cached last-good chain + broker-API fallback for NSE option-chain calls.

**Two honest caveats no hardware fixes:** (1) clean, survivorship-free, corporate-action-adjusted intraday **history is the hardest thing to get free** — and it caps conclusion quality; (2) hardware can test for edge rigorously but cannot manufacture one. Verify current free-tier limits before relying on them — they change.

---

## 15. Load profile

- **Live system — near-idle:** Stage-1 screen on 100 stocks is arithmetic (ms); FinBERT + models touch only ~25 survivors; LLM only on-demand / final 10.
- **Heavy work — scheduled:** backtest & retrain run overnight/weekends; retrain weekly, not daily; full multi-year sweep occasionally, rolling recent window for iteration.
- **Real load centers:** live data ingestion within free rate limits (I/O — cache); backtesting across years × 100+ stocks (stream via DuckDB, never one big DataFrame).
- **The one quality-vs-lightness lever:** backtest history depth — schedule it overnight rather than permanently shrinking it.

---

## 16. Build phases (BUILD IN THIS ORDER — do not jump ahead)

Each phase ends with a review checkpoint. Post a plan and wait for "go" before coding each phase; run code/tests and show real output before moving on. Maintain `TASKS.md` and `CHANGELOG.md`.

**Phase 0 — Foundations & backtester scaffold (FIRST).**
Repo structure; config system; DuckDB + SQLite schema (ticks, news, predictions, trades, signals, runs); the **abstract data-source interface** with a stub historical implementation that *enforces point-in-time access*; logging; experiment-tracking skeleton; continuous state-persistence hooks; a green `pytest` suite. *Done = schema + interface + tests pass, leakage-enforcement unit-tested.*

**Phase 1 — Data pipeline (live + historical behind the shared interface).**
Broker free-tier tick feed + option chain (cached, backoff, fallback); RSS news ingest + dedup + first-knowable timestamps; BSE/NSE filings + SEBI feed; free-tier macro feeds; NSE trading-calendar; Data Health Monitor (stale-feed suppression); assemble historical OHLCV (3–5 yrs), corporate-action-clean, survivorship gaps documented.

**Phase 2 — Intelligence (GPU).**
FinBERT sentiment (source-weighted, time-decayed); the ~15-feature engineering set; LightGBM quantile cone + scenario classifier (triple-barrier labels, no-trade class) + level engine; isotonic calibration + reliability diagram; SHAP drivers; research-backed seed weights.

**Phase 3 — Pattern library.**
Hypothesis register; conditional mining; FDR correction + out-of-sample confirmation + base-rate comparison; wire surviving patterns in as features AND narrator reasons.

**Phase 4 — Scanner + paper-trading engine.**
Two-stage scanner (100 → ~25 → 10); regime layer throttle; token paper engine (MIS/CNC, full cost + pessimistic slippage, hard guardrails, auto square-off); meta-label sizing (fractional Kelly / vol-target) on a risk-adjusted objective; two-tier monitor + exit optimizer; event-priority/idempotency; correlation-aware risk.

**Phase 5 — Narrator, dashboard & self-learning.**
Ollama narrator (explain-only) + anti-confabulation guard; Streamlit dashboard (cone, scenarios, positions, P&L, always-on benchmarks, per-regime stats); gated weekly retrain (validation gate, shrinkage, min-N, max-change cap).

**Phase 6 — Paper-trading evaluation (60–90 days).**
Run live on real data, no real money; track accuracy, calibration, interval coverage, net P&L vs benchmarks; identify systematic biases & failure modes; calibrate confidence thresholds on real data; reach an honest verdict: is there edge, and where?

---

## 17. What success honestly looks like

**Success:** calibrated confidence verified by a reliability diagram; cone intervals with correct empirical coverage; a handful of patterns surviving FDR + out-of-sample testing, reported with CIs; a clear, truthful answer to "is there edge, and where?" (even if "very little on liquid large-caps"); zero detectable leakage; results that hold out-of-sample and per-regime.

**Failure to avoid:** a beautiful backtest hiding look-ahead or survivorship bias; "50 reliable patterns" (noise in a costume); over-trading a flat market and bleeding costs; mistaking one lucky regime for durable edge.
