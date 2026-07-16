# DATA-INVENTORY.md — What's Actually on Disk

Snapshot of `data/market.duckdb` as of **2026-07-17**. This lets you know exactly what data
exists without re-querying. The DB is **git-ignored** (~2.8 GB) and lives on the training
machine only — it must be copied between machines out-of-band.

**DB file:** `data/market.duckdb` — **2,828 MB**, **12 tables**.

---

## 1. Table row counts

| Table | Rows | Status |
|---|---:|---|
| `ohlcv_intraday` | 40,281,399 | ✅ Primary intraday store (5 timeframes) |
| `fo_futures_oi_minute` | 1,201,891 | ⚠️ Only ~2.5 months (active contracts) |
| `fo_history` | 212,539 | ✅ Daily F&O history |
| `dhan_instruments` | 177,422 | ✅ Instrument master |
| `ohlcv_daily` | 81,477 | ✅ Daily OHLCV |
| `fo_daily_metrics` | 69,196 | ✅ Daily ATM OI metrics (feature source) |
| `download_jobs` | 16,172 | ℹ️ Download job queue/ledger |
| `corporate_actions` | 0 | ⬜ Empty (applied on the fly, not stored) |
| `news` | 0 | ❌ **Empty — no historical news feed** |
| `ohlcv_minute` | 0 | ⬜ Empty (unused; intraday tf='1' used instead) |
| `option_chain` | 0 | ❌ Empty (no per-strike chain history) |
| `ticks` | 0 | ⬜ Empty (unused) |

---

## 2. `ohlcv_intraday` by timeframe (the core dataset)

All timeframes span **2023-07-03 → 2026-07-03** (3 years).

| Timeframe | Rows | Symbols | Completeness |
|---|---:|---:|---|
| `1` (1-min) | 30,397,732 | 111 | ✅ Primary — used for all labeling/features |
| `5` (5-min) | 6,046,575 | 110 | ✅ |
| `15` (15-min) | 2,008,253 | 109 | ✅ |
| `25` (25-min) | 1,262,214 | 109 | ✅ |
| `60` (60-min) | 566,625 | 109 | ✅ |

**Note:** all timeframes contain a 09:30 IST bar — the `training_pipeline` query filters
`AND timeframe='1'` to avoid multi-timeframe row duplication.

## 3. `ohlcv_daily`
- **81,477 rows**, **111 symbols**, **2023-07-03 → 2026-07-03**. ✅ Complete.

---

## 4. F&O data

### `fo_daily_metrics` (feature source for `pcr_oi` / `pcr_volume` / `net_oi_change`)
- **69,196 rows**, **104 symbols**, **2023-07-17 → 2026-07-15**.
- Columns: `symbol, date, pcr_oi, pcr_volume, total_ce_oi, total_pe_oi, net_oi_change,
  max_pain_strike, futures_settle, source, ingested_at`.
- Feature completeness (non-null):
  | Feature | Non-null | % |
  |---|---:|---:|
  | `pcr_oi` | 67,791 | 98.0% |
  | `pcr_volume` | 68,756 | 99.4% |
  | `net_oi_change` | 69,196 | 100.0% |
- **104 symbols** (not 109): 5 symbols (CHOLAFIN, LTIM, MCDOWELL-N, TATAMOTORS, ZOMATO) lack
  F&O metrics due to instrument-master naming issues — deferred. In training these were
  zero-filled.

### `fo_history` (daily F&O history)
- **212,539 rows**, **104 symbols**, **2023-07-17 → 2026-07-15**. ✅

### `fo_futures_oi_minute` (minute futures OI)
- **1,201,891 rows**, **101 symbols**, **2026-04-29 → 2026-07-15** only (~2.5 months).
- ⚠️ **Structural limit:** Dhan retains minute-level futures OI only for *currently active*
  contracts. Historical/expired contract months return HTTP 400. This will **not** backfill.

### Computed-but-empty F&O features
- `spot_futures_basis` — ~3-month real depth only (built from active-contract futures),
  otherwise zero. Contributed to overfitting in v2_fo.
- `max_pain_distance` — effectively NULL/zero: only **ATM** OI was downloaded, so there is no
  per-strike chain to compute max-pain historically (`option_chain` table is empty).

---

## 5. `download_jobs` — download ledger status

| Timeframe | Done | Pending | Partial | Failed | Total | % Done |
|---|---:|---:|---:|---:|---:|---:|
| `1` (intraday 1-min) | 1,559 | 1,528 | 11 | 57 | 3,155 | 49% |
| `5` | 224 | 6 | 0 | 0 | 230 | 97% |
| `15` | 193 | 0 | 0 | 0 | 193 | 100% |
| `25` | 193 | 0 | 0 | 0 | 193 | 100% |
| `60` | 193 | 0 | 0 | 0 | 193 | 100% |
| `FO_CE_ATM` | 2,496 | 0 | 229 | 0 | 2,725 | 92% |
| `FO_PE_ATM` | 2,521 | 0 | 204 | 0 | 2,725 | 93% |
| `FO_FUT_HIST` | 2,477 | 0 | 0 | 248 | 2,725 | 91% |
| `FO_FUT_1M` | 300 | 0 | 3,437 | 296 | 4,033 | 7% |

**Reading this:**
- Intraday tf=1 shows 49% "done" in the *job ledger* but the table itself holds 30.4M rows
  across 111 symbols for the full 3 years — the pending jobs are redundant/backfill chunks
  already covered, not missing core data.
- `FO_FUT_1M` partials (3,437) = expired contract months returning HTTP 400 — a Dhan
  limitation, not a fixable gap.
- `FO_*_ATM` partials = chunks with sparse/short-dated coverage; the daily metrics that feed
  features are 98–100% complete regardless.

---

## 6. What's complete vs missing — quick reference

| Data | State | Usable for research? |
|---|---|---|
| Intraday OHLCV (1/5/15/25/60-min, 3 yr) | ✅ Complete | Yes — fully |
| Daily OHLCV (3 yr) | ✅ Complete | Yes |
| Daily ATM OI (`pcr_oi`/`pcr_volume`/`net_oi_change`) | ✅ 98–100% | Yes — tested, no edge |
| `spot_futures_basis` | ⚠️ ~3 months | Weak — overfits |
| `max_pain_distance` / per-strike chain | ❌ Not downloaded | No |
| Minute futures OI | ⚠️ ~2.5 months | No (structural Dhan limit) |
| News / sentiment | ❌ Empty (0 rows) | No — never tested; needs forward collection |
| Corporate actions | Applied on the fly | N/A (adjustment works; not stored as rows) |

---

## 7. Portability note

`data/market.duckdb` is **not in git** (`.gitignore`: `data/*.duckdb`). To move the complete
state to another machine: pull the repo (code + docs + provenance), then copy
`market.duckdb` separately (USB/network). Model binaries under `models/` are likewise
git-ignored; only `provenance.json` files travel with git if placed outside `/models/`.
