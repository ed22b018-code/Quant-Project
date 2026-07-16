"""
Resumable F&O OI download — Dhan API, ATM-only.

Phase 0: Quick probes (strike offsets, 3-key burst)
Phase 1: Schema + instrument master
Phase 2: Download CE/PE ATM + FUTSTK daily + FUTSTK minute OI
Phase 3: Compute fo_daily_metrics
Phase 4: Coverage report

Runs fully unattended. Decides key parallelism from burst test.
Resumable: completed jobs are never retried (ON CONFLICT DO NOTHING).
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import os
import queue
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Ensure project root is on sys.path so 'src.*' imports resolve when the script
# is launched directly (e.g. via Start-Process python scripts\run_fo_oi_download.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Load .env manually (dotenv optional) ─────────────────────────────────────
_env_path = Path(__file__).parents[1] / ".env"
if _env_path.exists():
    for _line in _env_path.read_text().splitlines():
        if "=" in _line and not _line.strip().startswith("#"):
            _k, _, _v = _line.partition("=")
            _k, _v = _k.strip(), _v.strip()
            if _k and _v and _k not in os.environ:
                os.environ[_k] = _v

# ── Constants ──────────────────────────────────────────────────────────────────
DB_PATH            = "data/market.duckdb"
BASE_URL           = "https://api.dhan.co/v2"
START_DATE         = date(2023, 7, 17)   # 3 yr back
END_DATE           = date.today()
CHUNK_DAYS         = 45                  # 46-day calendar limit discovered empirically
INTRADAY_CHUNK_DAYS= 30                  # smaller for large minute payloads
MAX_RETRIES        = 5

TF_CE_ATM   = "FO_CE_ATM"
TF_PE_ATM   = "FO_PE_ATM"
TF_FUT_HIST = "FO_FUT_HIST"
TF_FUT_1M   = "FO_FUT_1M"
FO_TFS      = [TF_CE_ATM, TF_PE_ATM, TF_FUT_HIST, TF_FUT_1M]

SENTINEL_EXPIRY = date(2099, 1, 1)
SENTINEL_STRIKE = 0.0

_log = logging.getLogger("fo_oi_dl")
_DB_WRITE_LOCK = threading.Lock()   # DuckDB: serialize writes from worker threads

# ── Extra schema (new table only) ─────────────────────────────────────────────
_EXTRA_SCHEMA = """
CREATE TABLE IF NOT EXISTS fo_futures_oi_minute (
    symbol      VARCHAR     NOT NULL,
    ts          TIMESTAMPTZ NOT NULL,
    oi          BIGINT,
    close       DOUBLE,
    volume      BIGINT,
    source      VARCHAR     NOT NULL DEFAULT 'dhan',
    ingested_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (symbol, ts)
);
"""


# ═════════════════════════════════════════════════════════════════════════════
# 1. HTTP layer
# ═════════════════════════════════════════════════════════════════════════════

class _KeyLimiter:
    def __init__(self, gap: float):
        self._lock = threading.Lock()
        self._last = 0.0
        self._gap  = gap

    def wait(self):
        with self._lock:
            now = time.monotonic()
            wait = self._gap - (now - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


class ApiClient:
    """Thread-safe Dhan client with per-key or shared rate limiting."""

    def __init__(self, tokens: list[tuple[str, str]], client_id: str,
                 per_key: bool, gap: float = 1.5):
        self._cid     = client_id
        self._tokens  = tokens
        self._per_key = per_key
        if per_key:
            self._limiters = [_KeyLimiter(gap) for _ in tokens]
        else:
            self._shared = _KeyLimiter(gap)
        self._idx_lock = threading.Lock()
        self._idx = 0

    def _pick(self):
        with self._idx_lock:
            i = self._idx
            self._idx = (self._idx + 1) % len(self._tokens)
        return i

    def post(self, path: str, body: dict) -> dict:
        url  = BASE_URL + path
        data = json.dumps(body).encode()
        for attempt in range(MAX_RETRIES):
            idx = self._pick()
            token = self._tokens[idx][1]
            if self._per_key:
                self._limiters[idx].wait()
            else:
                self._shared.wait()
            hdrs = {"access-token": token, "client-id": self._cid,
                    "Content-Type": "application/json", "Accept": "application/json"}
            req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode()
                if exc.code == 429:
                    w = 15.0 * (2 ** attempt)
                    _log.warning("429 rate-limit attempt %d; sleeping %.0fs", attempt+1, w)
                    time.sleep(w)
                elif exc.code >= 500:
                    w = 5.0 * (2 ** attempt)
                    _log.warning("HTTP %d server error attempt %d; sleeping %.0fs",
                                 exc.code, attempt+1, w)
                    time.sleep(w)
                else:
                    try:
                        msg = json.loads(raw).get("errorMessage", raw[:200])
                    except Exception:
                        msg = raw[:200]
                    raise ValueError(f"HTTP {exc.code}: {msg}") from exc
            except Exception as exc:
                w = 3.0 * (2 ** attempt)
                _log.warning("Network error attempt %d: %s; sleeping %.0fs",
                             attempt+1, exc, w)
                time.sleep(w)
        raise RuntimeError(f"All {MAX_RETRIES} attempts failed for POST {path}")


# ═════════════════════════════════════════════════════════════════════════════
# 2. Phase 0: Probes
# ═════════════════════════════════════════════════════════════════════════════

def probe_strike_offsets(client: ApiClient) -> None:
    _log.info("PROBE: Strike offset keywords...")
    base = dict(exchangeSegment="NSE_FNO", securityId="2885", instrument="OPTSTK",
                expiryFlag="MONTH", expiryCode=1, drvOptionType="CALL", interval="1",
                requiredData=["oi"], fromDate="2026-07-07", toDate="2026-07-09")
    for offset in ["ATM+1", "ATM-1", "ATM+2"]:
        try:
            resp = client.post("/charts/rollingoption", {**base, "strike": offset})
            data = resp.get("data", resp)
            ce   = data.get("ce", {}) if isinstance(data, dict) else {}
            oi   = ce.get("oi", []) if isinstance(ce, dict) else []
            ok   = len(oi) > 0 and any(x and x != 0 for x in oi)
            _log.info("  strike=%s: %s", offset, "OK" if ok else "empty/unsupported")
            time.sleep(4)
        except Exception as exc:
            _log.info("  strike=%s: error — %s", offset, str(exc)[:80])
            time.sleep(2)


def probe_multikey(tokens: list[tuple[str, str]], client_id: str) -> bool:
    """Burst all keys simultaneously. Returns True → per-key, False → per-account."""
    if len(tokens) < 2:
        _log.info("PROBE: Only 1 key — treating as per-key.")
        return True
    _log.info("PROBE: %d-key burst test...", len(tokens))
    body = dict(exchangeSegment="NSE_FNO", securityId="2885", instrument="OPTSTK",
                expiryFlag="MONTH", expiryCode=1, strike="ATM",
                drvOptionType="CALL", interval="1", requiredData=["oi"],
                fromDate="2026-07-07", toDate="2026-07-09")

    def _req(name, token):
        hdrs = {"access-token": token, "client-id": client_id,
                "Content-Type": "application/json", "Accept": "application/json"}
        data = json.dumps(body).encode()
        req  = urllib.request.Request(BASE_URL + "/charts/rollingoption",
                                      data=data, headers=hdrs, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return name, 200, None
        except urllib.error.HTTPError as e:
            return name, e.code, e.read().decode()[:80]
        except Exception as exc:
            return name, 0, str(exc)[:80]

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tokens)) as pool:
        results = list(pool.map(lambda t: _req(*t), tokens))

    for name, status, err in results:
        _log.info("  %s: HTTP %d %s", name, status, f"({err})" if err else "OK")

    if all(s == 200 for _, s, _ in results):
        _log.info("-> ALL 200 with zero delay => PER-KEY limits. Using all %d keys.", len(tokens))
        return True
    elif any(s == 429 for _, s, _ in results):
        _log.info("-> Some 429 => PER-ACCOUNT limits. Using 1 key.")
        return False
    else:
        _log.info("-> Inconclusive (no 429). Defaulting to per-key.")
        return True


# ═════════════════════════════════════════════════════════════════════════════
# 3. Schema + instrument master
# ═════════════════════════════════════════════════════════════════════════════

def setup_schema(db_path: str) -> None:
    import duckdb
    from src.db.schema import DUCKDB_SCHEMA_SQL
    with duckdb.connect(db_path) as con:
        con.execute(DUCKDB_SCHEMA_SQL)
        con.execute(_EXTRA_SCHEMA)
    _log.info("Schema: verified/created.")


def refresh_instruments(db_path: str) -> None:
    from src.data.dhan_downloader import DhanBulkDownloader
    dl = DhanBulkDownloader(db_path=db_path)
    n = dl._instrument_master.refresh()
    _log.info("Instrument master: %d rows loaded.", n)


def _get_secid(db_path: str, symbol_pat: str, itype: str,
               order_by: str = "expiry_date ASC NULLS LAST") -> str | None:
    """symbol_pat may contain SQL LIKE wildcards (e.g. 'RELIANCE-%' for futures)."""
    import duckdb
    try:
        with duckdb.connect(db_path, read_only=True) as con:
            q = f"""
                SELECT security_id FROM dhan_instruments
                WHERE symbol LIKE ? AND instrument_type = ?
                  AND exchange LIKE 'NSE%'
                ORDER BY {order_by} LIMIT 1
            """
            rows = con.execute(q, (symbol_pat, itype)).fetchall()
        return rows[0][0] if rows else None
    except Exception:
        return None


def load_secids(db_path: str, symbols: list[str]):
    futstk, equity = {}, {}
    for s in symbols:
        # FUTSTK symbols in instrument master are like 'RELIANCE-Jul2026-FUT'
        futstk[s] = _get_secid(db_path, s + "-%", "FUTSTK",
                                order_by="security_id ASC")
        equity[s]  = _get_secid(db_path, s, "EQUITY")
    missing_f = [s for s in symbols if not futstk[s]]
    missing_e = [s for s in symbols if not equity[s]]
    if missing_f:
        _log.warning("No FUTSTK secId for %d symbols: %s…", len(missing_f), missing_f[:5])
    if missing_e:
        _log.warning("No EQUITY secId for %d symbols: %s…",  len(missing_e), missing_e[:5])
    return futstk, equity


# ═════════════════════════════════════════════════════════════════════════════
# 4. Job management
# ═════════════════════════════════════════════════════════════════════════════

def _chunks(start: date, end: date, days: int) -> list[tuple[date, date]]:
    out, cur, d = [], start, timedelta(days=days - 1)
    while cur <= end:
        out.append((cur, min(cur + d, end)))
        cur = out[-1][1] + timedelta(days=1)
    return out


def ensure_jobs(db_path: str, symbols: list[str]) -> None:
    import duckdb
    rows = []
    for s in symbols:
        for tf in (TF_CE_ATM, TF_PE_ATM, TF_FUT_HIST):
            for cs, ce in _chunks(START_DATE, END_DATE, CHUNK_DAYS):
                rows.append((s, tf, cs, ce, "pending", 0))
        for cs, ce in _chunks(START_DATE, END_DATE, INTRADAY_CHUNK_DAYS):
            rows.append((s, TF_FUT_1M, cs, ce, "pending", 0))

    _log.info("Ensuring %d job rows…", len(rows))
    df = pd.DataFrame(rows, columns=["symbol","timeframe","chunk_start","chunk_end","status","rows_downloaded"])
    df["chunk_start"] = pd.to_datetime(df["chunk_start"]).dt.date
    df["chunk_end"]   = pd.to_datetime(df["chunk_end"]).dt.date

    with duckdb.connect(db_path) as con:
        con.register("_j", df)
        con.execute("""
            INSERT INTO download_jobs
              (symbol, timeframe, chunk_start, chunk_end, status, rows_downloaded)
            SELECT symbol, timeframe,
                   CAST(chunk_start AS DATE), CAST(chunk_end AS DATE),
                   status, rows_downloaded
            FROM _j
            ON CONFLICT (symbol, timeframe, chunk_start) DO NOTHING
        """)
        con.unregister("_j")
    _log.info("Job rows inserted (conflicts skipped).")


def pending_jobs(db_path: str) -> list[dict]:
    import duckdb
    with duckdb.connect(db_path, read_only=True) as con:
        rows = con.execute("""
            SELECT symbol, timeframe, chunk_start, chunk_end
            FROM download_jobs
            WHERE timeframe IN ('FO_CE_ATM','FO_PE_ATM','FO_FUT_HIST','FO_FUT_1M')
              AND status IN ('pending','failed')
            ORDER BY
              CASE timeframe
                WHEN 'FO_CE_ATM'   THEN 1
                WHEN 'FO_PE_ATM'   THEN 2
                WHEN 'FO_FUT_HIST' THEN 3
                WHEN 'FO_FUT_1M'   THEN 4
              END, chunk_start, symbol
        """).fetchall()
    return [{"sym": r[0], "tf": r[1], "start": r[2], "end": r[3]} for r in rows]


def mark_job(db_path: str, sym: str, tf: str, start: date,
             status: str, rows: int = 0, err: str | None = None) -> None:
    import duckdb
    now = datetime.now(timezone.utc)
    with _DB_WRITE_LOCK, duckdb.connect(db_path) as con:
        con.execute("""
            UPDATE download_jobs
            SET status=?, rows_downloaded=?, last_attempted=?, error=?
            WHERE symbol=? AND timeframe=? AND chunk_start=CAST(? AS DATE)
        """, (status, rows, now, err, sym, tf, start))


# ═════════════════════════════════════════════════════════════════════════════
# 5. Parse helpers
# ═════════════════════════════════════════════════════════════════════════════

def _ts_to_ist_date(ts: float) -> date:
    return (datetime.fromtimestamp(ts, tz=timezone.utc)
            + timedelta(hours=5, minutes=30)).date()


def parse_rollingoption(resp: dict, side_key: str) -> pd.DataFrame:
    """Last bar of each IST trading date → daily EOD row."""
    data = resp.get("data", resp)
    side = (data.get(side_key, {}) if isinstance(data, dict) else {}) or {}
    ts_arr    = side.get("timestamp", []) or []
    oi_arr    = side.get("oi",        []) or []
    vol_arr   = side.get("volume",    []) or []
    close_arr = side.get("close",     []) or []
    iv_arr    = side.get("iv",        []) or []
    if not ts_arr:
        return pd.DataFrame()

    n = len(ts_arr)
    rows = [{"date":  _ts_to_ist_date(ts_arr[i]),
             "ts":    ts_arr[i],
             "oi":    int(oi_arr[i])    if i < len(oi_arr)   and oi_arr[i]   else None,
             "vol":   int(vol_arr[i])   if i < len(vol_arr)  and vol_arr[i]  else None,
             "close": float(close_arr[i]) if i < len(close_arr) and close_arr[i] else None,
             "iv":    float(iv_arr[i])  if i < len(iv_arr)   and iv_arr[i]   else None,
             } for i in range(n)]
    df = pd.DataFrame(rows)
    # take last bar per trading date
    eod = df.sort_values("ts").groupby("date").last().reset_index()
    return eod[["date", "oi", "vol", "close", "iv"]]


def parse_hist_futstk(resp: dict) -> pd.DataFrame:
    ts_arr  = resp.get("timestamp",     []) or []
    close_a = resp.get("close",         []) or []
    vol_a   = resp.get("volume",        []) or []
    oi_a    = resp.get("open_interest", []) or []
    if not ts_arr:
        return pd.DataFrame()
    n = len(ts_arr)
    return pd.DataFrame([{
        "date":  _ts_to_ist_date(ts_arr[i]),
        "close": float(close_a[i]) if i < len(close_a) and close_a[i] else None,
        "vol":   int(vol_a[i])   if i < len(vol_a)   and vol_a[i]   else 0,
        "oi":    int(oi_a[i])    if i < len(oi_a)    and oi_a[i]    else None,
    } for i in range(n)])


def parse_intraday_futstk(resp: dict) -> pd.DataFrame:
    ts_arr  = resp.get("timestamp",     []) or []
    close_a = resp.get("close",         []) or []
    vol_a   = resp.get("volume",        []) or []
    oi_a    = resp.get("open_interest", []) or []
    if not ts_arr:
        return pd.DataFrame()
    n = len(ts_arr)
    return pd.DataFrame([{
        "ts":    datetime.fromtimestamp(ts_arr[i], tz=timezone.utc),
        "close": float(close_a[i]) if i < len(close_a) and close_a[i] else None,
        "vol":   int(vol_a[i])   if i < len(vol_a)   and vol_a[i]   else 0,
        "oi":    int(oi_a[i])    if i < len(oi_a)    and oi_a[i]    else None,
    } for i in range(n)])


# ═════════════════════════════════════════════════════════════════════════════
# 6. Storage
# ═════════════════════════════════════════════════════════════════════════════

def upsert_fo_history(db_path: str, df: pd.DataFrame,
                      symbol: str, option_type: str) -> int:
    if df.empty:
        return 0
    import duckdb
    now = datetime.now(timezone.utc)
    ins = df.copy()
    ins["symbol"]       = symbol
    ins["expiry_date"]  = SENTINEL_EXPIRY
    ins["strike_price"] = SENTINEL_STRIKE
    ins["option_type"]  = option_type
    ins["source"]       = ("dhan_rollingoption" if option_type in ("CE","PE")
                           else "dhan_historical")
    ins["ingested_at"]  = now
    if "vol" in ins.columns:
        ins = ins.rename(columns={"vol": "contracts"})
    else:
        ins["contracts"] = 0
    ins["settle_price"] = ins.get("close")
    for col in ["open","high","low","change_in_oi"]:
        if col not in ins.columns:
            ins[col] = None
    if "close" not in ins.columns:
        ins["close"] = None

    with _DB_WRITE_LOCK, duckdb.connect(db_path) as con:
        con.register("_fh", ins)
        con.execute("""
            INSERT INTO fo_history
              (symbol, date, expiry_date, strike_price, option_type,
               open, high, low, close, settle_price, contracts, oi,
               change_in_oi, source, ingested_at)
            SELECT symbol,
                   CAST(date AS DATE), CAST(expiry_date AS DATE),
                   CAST(strike_price AS DOUBLE), CAST(option_type AS VARCHAR),
                   CAST(open AS DOUBLE), CAST(high AS DOUBLE),
                   CAST(low AS DOUBLE),  CAST(close AS DOUBLE),
                   CAST(settle_price AS DOUBLE), CAST(contracts AS INTEGER),
                   CAST(oi AS BIGINT),   CAST(change_in_oi AS INTEGER),
                   source, CAST(ingested_at AS TIMESTAMP)
            FROM _fh
            ON CONFLICT (symbol, date, expiry_date, strike_price, option_type) DO NOTHING
        """)
        con.unregister("_fh")
    return len(ins)


def upsert_fut_minute(db_path: str, df: pd.DataFrame, symbol: str) -> int:
    if df.empty:
        return 0
    import duckdb
    now = datetime.now(timezone.utc)
    ins = df.copy().rename(columns={"vol": "volume"})
    ins["symbol"]      = symbol
    ins["source"]      = "dhan"
    ins["ingested_at"] = now
    if "volume" not in ins.columns:
        ins["volume"] = 0

    with _DB_WRITE_LOCK, duckdb.connect(db_path) as con:
        con.register("_fm", ins)
        con.execute("""
            INSERT INTO fo_futures_oi_minute
              (symbol, ts, oi, close, volume, source, ingested_at)
            SELECT symbol, CAST(ts AS TIMESTAMP), CAST(oi AS BIGINT),
                   CAST(close AS DOUBLE), CAST(volume AS BIGINT),
                   source, CAST(ingested_at AS TIMESTAMP)
            FROM _fm
            ON CONFLICT (symbol, ts) DO NOTHING
        """)
        con.unregister("_fm")
    return len(ins)


# ═════════════════════════════════════════════════════════════════════════════
# 7. Job processor
# ═════════════════════════════════════════════════════════════════════════════

def process_job(job: dict, client: ApiClient, db_path: str,
                futstk: dict, equity: dict) -> tuple[str, int]:
    sym, tf, start, end = job["sym"], job["tf"], job["start"], job["end"]
    frm = start.strftime("%Y-%m-%d")
    to  = end.strftime("%Y-%m-%d")

    try:
        if tf in (TF_CE_ATM, TF_PE_ATM):
            eq_id = equity.get(sym)
            if not eq_id:
                return "failed", 0
            opt_type = "CALL" if tf == TF_CE_ATM else "PUT"
            side_key = "ce"   if tf == TF_CE_ATM else "pe"
            db_type  = "CE"   if tf == TF_CE_ATM else "PE"
            resp = client.post("/charts/rollingoption", {
                "exchangeSegment": "NSE_FNO", "securityId": eq_id,
                "instrument": "OPTSTK", "expiryFlag": "MONTH", "expiryCode": 1,
                "strike": "ATM", "drvOptionType": opt_type, "interval": "1",
                "requiredData": ["close","volume","oi","iv"],
                "fromDate": frm, "toDate": to,
            })
            df = parse_rollingoption(resp, side_key)
            if df.empty:
                return "partial", 0
            rows = upsert_fo_history(db_path, df, sym, db_type)
            return "done", rows

        elif tf == TF_FUT_HIST:
            fut_id = futstk.get(sym)
            if not fut_id:
                return "failed", 0
            resp = client.post("/charts/historical", {
                "securityId": fut_id, "exchangeSegment": "NSE_FNO",
                "instrument": "FUTSTK", "oi": True,
                "fromDate": frm, "toDate": to,
            })
            df = parse_hist_futstk(resp)
            if df.empty:
                return "partial", 0
            rows = upsert_fo_history(db_path, df, sym, "FUT")
            return "done", rows

        elif tf == TF_FUT_1M:
            fut_id = futstk.get(sym)
            if not fut_id:
                return "failed", 0
            resp = client.post("/charts/intraday", {
                "securityId": fut_id, "exchangeSegment": "NSE_FNO",
                "instrument": "FUTSTK", "interval": "1", "oi": True,
                "fromDate": frm, "toDate": to,
            })
            df = parse_intraday_futstk(resp)
            if df.empty:
                return "partial", 0
            rows = upsert_fut_minute(db_path, df, sym)
            return "done", rows

    except Exception as exc:
        _log.error("Job %s %s %s→%s: %s", sym, tf, frm, to, exc)
        return "failed", 0

    return "failed", 0


# ═════════════════════════════════════════════════════════════════════════════
# 8. Download loop (single or multi-key)
# ═════════════════════════════════════════════════════════════════════════════

def run_download(jobs: list[dict], client: ApiClient, db_path: str,
                 futstk: dict, equity: dict, n_workers: int) -> tuple[int, int]:
    done_c, fail_c = 0, 0
    n_total  = len(jobs)
    cnt_lock = threading.Lock()

    def _worker(q: "queue.Queue"):
        nonlocal done_c, fail_c
        while True:
            try:
                job = q.get(timeout=5)
            except queue.Empty:
                break
            status, rows = process_job(job, client, db_path, futstk, equity)
            mark_job(db_path, job["sym"], job["tf"], job["start"], status, rows,
                     None if status != "failed" else "process_job returned failed")
            with cnt_lock:
                if status == "done":
                    done_c += 1
                else:
                    fail_c += 1
                total_done = done_c + fail_c
                if total_done % 25 == 0 or total_done == n_total:
                    pct = 100.0 * total_done / n_total
                    _log.info("Progress: %d/%d (%.0f%%) — %d done, %d failed",
                              total_done, n_total, pct, done_c, fail_c)
            q.task_done()

    jq: "queue.Queue" = queue.Queue()
    for j in jobs:
        jq.put(j)

    threads = [threading.Thread(target=_worker, args=(jq,), daemon=True)
               for _ in range(n_workers)]
    for t in threads:
        t.start()
    jq.join()
    return done_c, fail_c


# ═════════════════════════════════════════════════════════════════════════════
# 9. Phase 3: Compute fo_daily_metrics
# ═════════════════════════════════════════════════════════════════════════════

def compute_metrics(db_path: str, symbols: list[str]) -> None:
    import duckdb
    _log.info("Back-filling change_in_oi via LAG…")
    with duckdb.connect(db_path) as con:
        con.execute("""
            WITH lagged AS (
                SELECT symbol, date, expiry_date, strike_price, option_type,
                       LAG(oi) OVER (
                           PARTITION BY symbol, expiry_date, strike_price, option_type
                           ORDER BY date
                       ) AS prev_oi
                FROM fo_history
                WHERE expiry_date = CAST('2099-01-01' AS DATE)
            )
            UPDATE fo_history
            SET change_in_oi = CAST(fo_history.oi AS INTEGER) - CAST(lagged.prev_oi AS INTEGER)
            FROM lagged
            WHERE fo_history.symbol       = lagged.symbol
              AND fo_history.date         = lagged.date
              AND fo_history.expiry_date  = lagged.expiry_date
              AND fo_history.strike_price = lagged.strike_price
              AND fo_history.option_type  = lagged.option_type
              AND fo_history.change_in_oi IS NULL
              AND lagged.prev_oi IS NOT NULL
        """)
    _log.info("change_in_oi back-filled.")

    _log.info("Aggregating fo_history → fo_daily_metrics…")
    from src.data.dhan_downloader import DhanBulkDownloader
    dl = DhanBulkDownloader(db_path=db_path)
    total = dl.compute_fo_metrics(symbols=symbols)
    _log.info("fo_daily_metrics: %d rows upserted.", total)

    # NULL out max_pain_strike — ATM-only data makes it meaningless (would be 0.0)
    import duckdb
    with duckdb.connect(db_path) as con:
        con.execute("UPDATE fo_daily_metrics SET max_pain_strike = NULL "
                    "WHERE max_pain_strike = 0.0")
    _log.info("max_pain_strike NULLed out (ATM-only build — no per-strike data).")


# ═════════════════════════════════════════════════════════════════════════════
# 10. Coverage report
# ═════════════════════════════════════════════════════════════════════════════

def coverage_report(db_path: str) -> None:
    import duckdb
    with duckdb.connect(db_path, read_only=True) as con:
        df = con.execute("""
            SELECT timeframe,
                   SUM(CASE WHEN status='done'    THEN 1 ELSE 0 END) AS done,
                   SUM(CASE WHEN status='partial' THEN 1 ELSE 0 END) AS partial,
                   SUM(CASE WHEN status='failed'  THEN 1 ELSE 0 END) AS failed,
                   SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending,
                   COUNT(*) AS total
            FROM download_jobs
            WHERE timeframe IN ('FO_CE_ATM','FO_PE_ATM','FO_FUT_HIST','FO_FUT_1M')
            GROUP BY timeframe ORDER BY timeframe
        """).df()
        dm = con.execute("SELECT COUNT(*) FROM fo_daily_metrics").fetchone()[0]
        try:
            fm = con.execute("SELECT COUNT(*) FROM fo_futures_oi_minute").fetchone()[0]
        except Exception:
            fm = 0
        try:
            fh = con.execute(
                "SELECT COUNT(*) FROM fo_history WHERE expiry_date = '2099-01-01'"
            ).fetchone()[0]
        except Exception:
            fh = 0

    sep = "=" * 74
    print(sep)
    print("F&O OI DOWNLOAD — COVERAGE REPORT")
    print(sep)
    print(f"{'TF':<16} {'Done':>8} {'Partial':>8} {'Failed':>8} {'Pending':>8} {'Total':>8}")
    print("-" * 74)
    for _, r in df.iterrows():
        pct = 100.0 * r["done"] / max(r["total"], 1)
        print(f"{r['timeframe']:<16} {r['done']:>8} {r['partial']:>8} "
              f"{r['failed']:>8} {r['pending']:>8} {r['total']:>8}  ({pct:.0f}%)")
    print("-" * 74)
    print(f"fo_history rows (OI data):       {fh:>12,}")
    print(f"fo_daily_metrics rows:           {dm:>12,}")
    print(f"fo_futures_oi_minute rows:       {fm:>12,}")
    print(sep)


# ═════════════════════════════════════════════════════════════════════════════
# 11. Main
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    log_file = Path(__file__).parents[1] / "fo_oi_download.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), mode="a", encoding="utf-8"),
        ],
    )
    _log.info("=" * 60)
    _log.info("F&O OI download start — %s", datetime.now().isoformat())
    _log.info("Date range: %s to %s", START_DATE, END_DATE)

    # Credentials
    client_id = os.environ.get("DHAN_CLIENT_ID", "")
    tokens: list[tuple[str, str]] = []
    for k in sorted(os.environ):
        if k.startswith("DHAN_ACCESS_TOKEN") and os.environ[k].strip():
            tokens.append((k, os.environ[k].strip()))
    if not tokens:
        _log.error("No DHAN_ACCESS_TOKEN in environment. Aborting.")
        sys.exit(1)
    _log.info("Loaded %d API token(s): %s", len(tokens), [t[0] for t in tokens])

    # ── Phase 0: Probes ───────────────────────────────────────────────────────
    probe_client = ApiClient(tokens[:1], client_id, per_key=False, gap=4.0)
    probe_strike_offsets(probe_client)
    time.sleep(4)
    per_key = probe_multikey(tokens, client_id)
    time.sleep(4)

    effective_tokens = tokens if per_key else tokens[:1]
    n_workers = len(effective_tokens)
    client = ApiClient(effective_tokens, client_id, per_key=per_key, gap=0.5)
    _log.info("Using %d key(s), %d worker thread(s), 0.5s/key gap.", n_workers, n_workers)

    # ── Phase 1: Schema + instruments ─────────────────────────────────────────
    setup_schema(DB_PATH)
    refresh_instruments(DB_PATH)

    from src.config import load_config
    symbols = load_config().universe.symbols
    _log.info("Universe: %d symbols", len(symbols))

    futstk_ids, equity_ids = load_secids(DB_PATH, symbols)

    # ── Phase 2: Jobs → Download ──────────────────────────────────────────────
    ensure_jobs(DB_PATH, symbols)
    jobs = pending_jobs(DB_PATH)
    _log.info("Pending/failed jobs: %d", len(jobs))

    if jobs:
        t0 = time.monotonic()
        done_c, fail_c = run_download(jobs, client, DB_PATH,
                                      futstk_ids, equity_ids, n_workers)
        elapsed = (time.monotonic() - t0) / 60
        _log.info("Download finished in %.1f min: %d done, %d failed.",
                  elapsed, done_c, fail_c)
    else:
        _log.info("No pending jobs — all already complete.")

    # ── Phase 3: Compute metrics ──────────────────────────────────────────────
    compute_metrics(DB_PATH, symbols)

    # ── Phase 4: Coverage ─────────────────────────────────────────────────────
    coverage_report(DB_PATH)
    _log.info("All done. Log: %s", log_file)


if __name__ == "__main__":
    main()
