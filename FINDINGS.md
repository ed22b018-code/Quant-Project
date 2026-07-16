# FINDINGS.md — Honest Research Report

**Project:** Intraday scenario-classification trading system for NSE large-cap equities
**Universe:** 109 F&O-eligible NSE large-caps (survivorship-corrected)
**Period studied:** 2023-07-05 → 2026-07-03 (3 years)
**Report date:** 2026-07-17
**Status:** Research concluded — no demonstrated edge. This document is the project's real deliverable.

---

## 1. The core question

> Can a machine-learning model trade intraday on large-cap NSE equities and produce a
> **real, statistically significant, after-cost edge** over naive benchmarks (buy-and-hold,
> always-flat) — first from price/volume alone, and then with F&O open-interest signals added?

## 2. The honest answer

**No.** After rigorous, leakage-controlled, after-cost evaluation:

- **Price/volume model (v1_full): no demonstrated edge.** Net +5.37% over the sealed test
  window — but it *trails* buy-and-hold (+6.12%) and the return is not statistically
  significant (block t-stat 1.11–1.19, need |t| ≥ 2.0).
- **F&O-augmented model (v2_fo): worse.** Adding real daily ATM open-interest features
  *reduced* net return to +2.75% (−2.61pp vs the price/volume model). The F&O features were
  actively used by the model (14.4% in-sample gain importance) yet degraded out-of-sample
  performance — the signature of **fitting noise, not signal**.
- **Neither model beats buy-and-hold after costs**, and **~87% of gross P&L is consumed by
  transaction costs**.

**Bottom line: there is no demonstrated intraday edge on this universe with these signals,
after costs.** This is a valid and valuable result — it says *stop*, not *tune harder*.

---

## 3. The full evidence chain

### 3.1 Price/volume: from a +37% mirage down to the truth

The first "headline" result was **+37.66%** net over the test window. It was a **mirage**,
caught and dismantled by the evaluation harness — this is the single most important episode
in the project because it demonstrates the checks work:

| Stage | Result | What happened |
|---|---|---|
| Initial backtest | +37.66% | Looked spectacular |
| **Exit-price bug found** | — | SHORT+BEARISH trades booked exit at `entry×(1−PROFIT_PCT)` = −1%, but the BEARISH barrier actually fires at `entry×(1−STOP_PCT)` = −0.5%. Every short profit was overstated ~2× (663 trades, ₹161,404 phantom profit). |
| Corrected backtest | **+5.37%** | The real number after fixing the exit price |
| Benchmark comparison | trails B&H | Buy-and-hold returned +6.12% over the same window — the model *underperforms* the do-nothing benchmark |
| Statistical test | insignificant | Block t-stat ≈ 1.11 (5 blocks of 15 days); cannot reject "zero edge" |
| Regime check | not regime-riding | Test window was bullish (+19% cumulative); model made money in both bull and bear weeks, so it wasn't purely riding the tape — but it still had no edge over B&H |

**Verdict (price/volume): no demonstrated edge.** Expected — price/volume signals on liquid
large-caps are rapidly arbitraged.

### 3.2 F&O open interest: adding it made things worse

Checkpoint 7.2 retrained the identical pipeline (same date range, same walk-forward folds,
same purge/embargo, same sealed test set) with real F&O features added:
`pcr_oi`, `pcr_volume`, `net_oi_change` (3-year depth), plus `spot_futures_basis` (~3-month
depth). Head-to-head on **identical** sealed-test timestamps:

| Model | Trades | Net P&L | % of ₹5L pool | vs price/vol |
|---|---|---|---|---|
| v1_full (price/volume only) | 1,361 | +₹26,836 | **+5.37%** | — |
| **v2_fo (F&O-augmented)** | 1,300 | +₹13,768 | **+2.75%** | **−2.61 pp** |
| Buy-and-hold (equal-weight) | — | +₹30,606 | +6.12% | — |

The live v1_full re-run reproduced its Checkpoint 6 number **exactly** (₹26,836 / +5.37% /
1,361 trades), confirming the evaluation harness is deterministic and trustworthy.

**Why this is overfitting, not signal:**
- F&O features drew **14.4% of in-sample gain importance** (`spot_futures_basis` 4.5%,
  `pcr_volume` 3.3%, `pcr_oi` 3.3%, `net_oi_change` 3.2%) — they were *used*, not ignored —
  yet OOS performance got *worse*. High in-sample importance + worse OOS = textbook
  noise-fitting.
- Tell within the tell: the top F&O feature (`spot_futures_basis`) has only ~3 months of
  real data (the rest zero-filled). The model leaned hardest on its thinnest, least
  representative feature.
- Block t-stat over 10 OOS months: **+1.19** (mean monthly +0.264%, std 0.700%) — not
  significant. Even v2_fo's own +2.75% cannot be distinguished from zero.

**This is not a data leak.** A daily-OI-in-intraday leak would make v2 look artificially
*better*, not worse — we observed the opposite. Corroborated by: PIT test PASS (features at
t0 see only `date < t0`, i.e. D−1 data; the same-day OI row is correctly blocked), and a
clean feature-dominance check (top feature `atr_20d` at 32.6% is only 2.6× the second,
below the 5× leak-flag threshold).

### 3.3 The cost wall

Across the OOS window, **gross P&L +21.27% collapsed to net +2.75% — costs ate ~87% of
gross**. On large-cap NSE intraday with a triple-barrier ±0.5–1.0% target, the edge required
to overcome the real cost stack (STT, exchange/SEBI/stamp charges, brokerage, GST, plus
slippage) simply isn't present in these signals. Slippage sensitivity is thin: the
price/volume model goes net-negative at roughly +10 bps of additional slippage.

---

## 4. What was rigorously tested vs. what remains untested

### Rigorously tested (trustworthy conclusions)
- **Price/volume features** (8): 3-year depth, full intraday OHLCV. Conclusion solid.
- **Daily ATM open-interest** (`pcr_oi` 98%, `pcr_volume` 99.4%, `net_oi_change` 100%
  non-null; 104 symbols; 2023-07-17 → 2026-07-15). Conclusion solid.

### Tested but data-limited (weaker conclusions)
- **`spot_futures_basis`**: only ~3 months of real depth (Dhan retains minute/near-dated
  futures OI only for active contracts). Contributed to overfitting; longer history *might*
  behave differently but there is no reason to expect edge.
- **`max_pain_distance`**: effectively NULL/zero — only ATM OI was downloaded, no per-strike
  chain, so max-pain cannot be computed historically.

### Never tested (data unavailable)
- **Sentiment / news** (`news_count_1d`, `news_count_5d`, `news_sentiment`): the `news`
  table is **empty (0 rows)**. There is no historical news feed available from the current
  sources; the FinBERT scorer and news pipeline are built and unit-tested but have never run
  on real historical data. These features were all-zero in *both* models.
- **Per-strike option-chain history** and **full minute-level futures OI history**: not
  retained by Dhan for expired contracts (see [DATA-INVENTORY.md](DATA-INVENTORY.md)).

---

## 5. Why these findings are trustworthy — the methodology (the real achievement)

The engineering that makes a *negative* result credible is the project's durable value:

1. **Point-in-time (PIT) enforcement.** Every feature query uses `WHERE date < t0`
   (D−1 rule). A dedicated leakage exception (`DataLeakageError`) and a 28-test
   `test_pit_enforcement.py` suite guard against look-ahead. Verified live: at t0=2026-07-01
   the max visible OI date is 2026-06-30; the same-day row is blocked.
2. **Survivorship-corrected universe.** The 109-symbol universe is built from F&O-eligibility
   logic, not from today's index membership — avoiding the classic "only winners survive" bias.
3. **Walk-forward validation with purge + embargo.** 5 expanding folds; `purge_and_embargo()`
   removes training windows whose label horizon overlaps the validation start (proven by test
   to leave zero `t1 ≥ val_start`).
4. **Sealed test set, touched once.** The final 10% of the date range is evaluated exactly
   once, after all training/calibration. A code-level guard raises on a second access.
5. **After-cost benchmarking.** A rupee-accurate NSE cost engine plus slippage is applied to
   every trade, and results are always shown against buy-and-hold and always-flat — not in
   isolation.
6. **Multi-period + statistical significance.** P&L is reported per-month over ~9 OOS months
   with a block t-stat, not as a single cherry-picked window.
7. **Mirage-catching stress tests.** The harness caught the +37% exit-price bug and reduced it
   to the honest +5.37%. A pipeline that will *disprove its own headline* is the thing worth
   keeping.

---

## 6. Candidate future directions — HYPOTHESES, not promises

Stated honestly, with odds:

- **Cut trade frequency / attack the 87% cost drag (best lever, still a long shot).** The
  biggest single destroyer of returns is cost, not signal quality. Fewer, higher-conviction
  trades (higher confidence threshold, wider barriers, position-level netting) would reduce
  drag — but if the underlying signal has no edge, lower costs only move a negative number
  closer to zero, not into profit. *Worth testing; unlikely to create edge on its own.*
- **Genuinely orthogonal signals via forward-collected news/sentiment (untested, uncertain).**
  The one input never actually tested is news/sentiment, because no historical feed exists.
  The only honest way to test it is to **collect it forward** from now and evaluate on future
  OOS data — no backtest is possible. This is a real experiment, not a promise; large-cap
  news is also widely disseminated and may already be priced in fast.
- **Accept market efficiency at this granularity (most likely truth).** The evidence points
  to daily/intraday large-cap NSE price, volume, and ATM-OI being efficiently arbitraged. A
  rational conclusion is to stop adding features to this formulation and either change the
  problem (different horizon, less-liquid universe, event-driven setups) or repurpose the
  engineering (see [REUSABLE-ASSETS.md](REUSABLE-ASSETS.md)).

**None of these is expected to produce a deployable edge.** The value delivered is a
trustworthy *no*, backed by machinery that can be trusted to say *yes* if a yes ever exists.

---

## 7. Model artifacts (neither recommended for deployment)

| Artifact | Type | Test-window net | Recommendation |
|---|---|---|---|
| `models/v1_full/` | PRICE_VOLUME_ONLY | +5.37% | **Do not deploy** — trails buy-and-hold, not significant |
| `models/v2_fo/` | FO_AUGMENTED | +2.75% | **Do not deploy** — worse than v1_full; F&O overfits noise |

Both are retained for reproducibility and reference only. Model binaries live on the training
machine (git-ignored); provenance JSON documents each. See [ARCHITECTURE.md](ARCHITECTURE.md)
for how they were produced.
