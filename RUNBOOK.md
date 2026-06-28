# Dhan Deployment Runbook

**One-time path from "I have an API key" to "models trained and saved."**

Follow these steps in order. Each step has success criteria and a failure action.
The key **never travels through git** — it must be entered by hand on each machine.

---

## Step 1 — Dev Laptop: Add key, run test-fetch, fix any surprises

**Goal:** Verify the Dhan API responds correctly and the parser handles the real format.
This is the first real API call. Expect small surprises here — that is fine and expected.

### 1a. Add credentials to `.env`

Create (or edit) the `.env` file in the project root. It is git-ignored — never committed.

```
DHAN_CLIENT_ID=your_client_id_here
DHAN_ACCESS_TOKEN=your_access_token_here
DHAN_ADJUSTMENT_STATUS=adjusted
```

> `DHAN_ADJUSTMENT_STATUS` must be exactly `adjusted` or `unadjusted`.

### 1b. Run the small test-fetch

```bash
python -m scripts.test_fetch
```

Default: RELIANCE, 3 days, 1-minute bars. To use a different symbol or window:

```bash
python -m scripts.test_fetch --symbol INFY --days 5 --timeframe 5
```

**Success criteria:**
- `[PASS] Response format OK — all expected columns present.`
- Row count is non-zero.

**Failure actions:**

| Output | Fix |
|---|---|
| `[FAIL] API returned empty or None data` | Check `.env` — wrong key or expired token |
| `[PARTIAL] Column mismatch — missing: ['ts', ...]` | Fix `src/data/dhan_downloader.py` → `DhanBulkDownloader._parse_ohlcv_response()`. The raw API uses different key names than expected. Check `raw.get("timestamp", [])` — verify all 6 array key names |
| `401 Unauthorized` in logs | Token expired — re-generate on Dhan portal |
| Any other exception | Run with `--verbose` for debug trace |

---

## Step 2 — Dev Laptop: Commit fixes, push to GitHub

After the test-fetch passes (and after fixing any parser issues):

```bash
git add -A
git commit -m "fix: Dhan response parsing from first real API call"
git push
```

**Do NOT commit `.env` or any data files.** They are git-ignored.
Check `git status` — it should show no `.env` or `data/` in staged files.

---

## Step 3 — RTX 3060: Clone repo, set up environment

On the RTX 3060 machine:

```bash
git clone <your-repo-url>
cd <repo-dir>
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Then **add the Dhan key again by hand** — it does not travel through git:

```
DHAN_CLIENT_ID=your_client_id_here
DHAN_ACCESS_TOKEN=your_access_token_here
DHAN_ADJUSTMENT_STATUS=adjusted
```

> The code is already fixed (from Step 1). But the machine's network connection
> and environment variable loading are unverified until Step 4.

---

## Step 4 — RTX 3060: Run test-fetch to confirm THIS machine connects

```bash
python -m scripts.test_fetch
```

Same command as Step 1 — but now you are verifying that **this machine** can
reach the Dhan API. Network config, proxy settings, and env var loading can
differ between machines.

**Success criteria:** Same as Step 1 — `[PASS]`, non-zero row count.

**Failure actions:** Same as Step 1.
If the test passes on the dev laptop but fails here, the issue is machine-specific:
- Check `.env` was created on THIS machine (not copied via git)
- Check network/firewall/proxy
- Confirm Python version: `python --version` (requires 3.10+)

---

## Step 5 — RTX 3060: Run full orchestration

The pipeline is **resumable** — if it fails at any stage, fix the issue and
re-run. Completed stages are skipped automatically.

```bash
python -m scripts.run_orchestration --skip-fo
```

> **Use `--skip-fo` for the first run.** See [F&O Known Limitation](#fo-known-limitation) below.

This runs all 8 stages in order:

| # | Stage | What it does |
|---|---|---|
| 1 | `preflight` | Checks env vars, disk space (≥ 25 GB), DB path writable |
| 2 | `test_fetch` | Tiny API probe (1 symbol, 3 days) — format validation |
| 3 | `bulk_download` | Full 5-year intraday OHLCV (F&O skipped with --skip-fo) |
| 4 | `coverage_check` | Prints coverage matrix — how many dates/symbols have data |
| 5 | `build_dataset` | Assembles PIT feature matrix with all gate flags |
| 6 | `train_models` | LightGBM quantile cone + scenario classifier (walk-forward) |
| 7 | `calibrate` | Isotonic calibration on last validation fold |
| 8 | `save_summary` | Writes `models/v1/provenance.json`, prints final metrics |

Progress is checkpointed after every stage in `models/v1/.orchestration_state.json`.

### Resuming after a failure

```bash
# Re-run from a specific stage (skips already-completed stages before it):
python -m scripts.run_orchestration --start-from bulk_download

# Skip the format probe (already verified in Step 4):
python -m scripts.run_orchestration --skip-test-fetch

# Dry run (validates wiring without real API or training):
python -m scripts.run_orchestration --dry-run
```

### Custom universe or date range

```bash
python -m scripts.run_orchestration \
    --symbols RELIANCE TCS INFY HDFCBANK \
    --start 2020-01-01 \
    --end 2024-12-31 \
    --output-dir models/v2
```

### Success criteria

- All 8 stages report `complete` in the log.
- `models/v1/provenance.json` exists and contains `"dry_run": false`.
- No `RuntimeError: Stage X failed` in output.

---

## Quick reference — common commands

```bash
# Preflight only (check env / disk before anything else):
python -c "from src.orchestration.preflight import run_preflight; run_preflight('data/market.duckdb')"

# Test-fetch only:
python -m scripts.test_fetch

# Bulk download only (without training):
python -m scripts.run_bulk_download

# Full orchestration:
python -m scripts.run_orchestration

# Full orchestration, resuming from training:
python -m scripts.run_orchestration --start-from train_models

# Dry run (checks wiring, no real calls):
python -m scripts.run_orchestration --dry-run
```

---

## Credential security

- `.env` is listed in `.gitignore` — it will never appear in `git status` as an untracked file to stage.
- Credentials must be entered by hand on **each machine** — they do not travel through git.
- Large downloaded data (`data/*.duckdb`) is also git-ignored — it lives on the training machine only.
- Trained models (`models/`) are git-ignored — save them via other means (cloud storage, direct copy) if you need them on another machine.
- If a key is accidentally committed: rotate it immediately on the Dhan portal, then remove it from git history with `git filter-repo`.

---

## F&O Known Limitation {#fo-known-limitation}

**Confirmed on first real API contact (2026-06-29).**

### What the real Dhan API returns for F&O

`POST /charts/historical` with `instrument: "FUTSTK"` returns:
```json
{"open": [...], "high": [...], "low": [...], "close": [...], "volume": [...], "timestamp": [...]}
```

This is the **same OHLCV structure as equity** — **no open interest (OI), no strike, no expiry in the response**.

### What features need

| Feature | Needs | Available from API? |
|---|---|---|
| `pcr_oi` | CE OI / PE OI | ❌ OI not in response |
| `pcr_volume` | CE volume / PE volume | ✅ volume present |
| `net_oi_change` | daily Δ OI | ❌ OI not in response |
| `max_pain_distance` | strike where ITM payout is minimised | ❌ OI not in response |
| `spot_futures_basis` | futures settle - spot | ⚠️ close ≈ settle (imprecise) |

### Impact

- Run bulk download with `--skip-fo` flag.
- F&O-gated features (`pcr_oi`, `net_oi_change`, `max_pain_distance`) will be NaN.
- Model trains on the remaining features (price/volume + sentiment when available).
- `spot_futures_basis` may be added later once a reliable futures settle price source is confirmed.

### Additional F&O API finding

- The equity security_id (e.g., 2885 for RELIANCE) is **not valid** for F&O API calls.
- Each futures contract has its own security_id in the instrument master (e.g., RELIANCE near-month future = 62802).
- The instrument master must be refreshed (`DhanInstrumentMaster.refresh()`) before any F&O download.
- Historical F&O would require downloading each contract separately by its security_id — one contract per request.

### Recommendation

Use `--skip-fo` for the bulk download. The price/volume features alone are sufficient for the first model run. Revisit F&O OI when a data source that provides historical open interest is confirmed.
