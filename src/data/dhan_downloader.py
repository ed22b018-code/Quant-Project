"""Dhan Bulk Historical Downloader.

Paced, resumable batch job that pulls ~5yr intraday OHLCV + F&O/OI history
for the full F&O universe from Dhan's REST API, storing to DuckDB.

No SDK import here — uses plain urllib.request for all REST calls (same pattern
as bhavcopy.py and option_chain.py).  dhanhq SDK is quarantined to dhan_adapter.py.

Key design
----------
* Resumable: every (symbol, timeframe, chunk_start) is a row in `download_jobs`.
  Status: 'pending' | 'done' | 'failed' | 'partial'.  On restart, only
  'pending' and 'failed' rows are retried.  DONE rows are skipped forever.
* Paced: configurable `request_delay_seconds` between API calls + exponential
  backoff on 429/5xx.  Stays within Dhan's per-request and daily caps.
* Idempotent storage: ON CONFLICT DO NOTHING — re-running never duplicates rows.
* Coverage report: per-symbol × timeframe matrix shows DONE/PARTIAL/FAILED/PENDING
  so completeness can be verified before the 30-day window closes.

Usage
-----
    dl = DhanBulkDownloader(db_path="data/market.duckdb")
    dl.run(symbols=config.universe.symbols,
           timeframes=["1", "5", "15", "25", "60"],
           start_date=date(2019, 1, 1),
           end_date=date(2024, 12, 31))
    dl.run_fo(symbols=config.universe.symbols,
              start_date=date(2019, 1, 1),
              end_date=date(2024, 12, 31))
    dl.compute_fo_metrics(symbols=config.universe.symbols)
    print(dl.coverage_report())

CLI entry point: scripts/run_bulk_download.py

Instrument master
-----------------
Dhan's REST API uses numeric security IDs (not NSE ticker strings).
DhanInstrumentMaster downloads a mapping CSV from Dhan and caches it in the
dhan_instruments DuckDB table.  Call refresh() once before the first run.

TODO (on key arrival): verify exact Dhan API endpoint paths and request/response
JSON structure in dhan_adapter.py and this file against the live API docs.
The mocked structure used in tests mirrors the expected Dhan format — adjust
_parse_ohlcv_response() and _build_request_body() if the real API differs.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from src.db.duckdb_client import DuckDBClient
from src.db.schema import DUCKDB_SCHEMA_SQL

_log = logging.getLogger(__name__)

# ── Dhan API endpoints (base URL in config; these are path suffixes) ──────────
_DHAN_INTRADAY_PATH = "/charts/intraday"
_DHAN_HISTORICAL_PATH = "/charts/historical"
_DHAN_INSTRUMENT_MASTER_URL = "https://images.dhan.co/api-data/api-scrip-master.csv"

_BACKOFF_BASE: float = 1.0

# Timeframe label used for F&O jobs in download_jobs table
_FO_TIMEFRAME = "FO"


# ─────────────────────────────────────────────────────────────────────────────
# DhanInstrumentMaster
# ─────────────────────────────────────────────────────────────────────────────

class DhanInstrumentMaster:
    """Dhan instrument master — maps NSE ticker → Dhan numeric security_id.

    Dhan's REST API requires a numeric security_id for every request.  This class
    downloads the master CSV from Dhan's public endpoint and caches it in the
    dhan_instruments DuckDB table.

    _mock_master: dict[tuple[str, str], str] injection for tests.
      Key: (symbol, instrument_type)  e.g. ("RELIANCE", "EQUITY")
      Value: security_id string        e.g. "1333"
    """

    def __init__(self, db_path: str | Any, *, _mock_master: dict | None = None) -> None:
        self._db_path = str(db_path)
        self._mock_master = _mock_master
        self._cache: dict[tuple[str, str], str] = {}
        if _mock_master is not None:
            self._cache.update(_mock_master)

    def refresh(self, url: str = _DHAN_INSTRUMENT_MASTER_URL) -> int:
        """Download instrument master CSV and upsert into dhan_instruments.

        Returns the number of rows loaded.
        Credentials not required — this is a public endpoint.
        """
        if self._mock_master is not None:
            _log.debug("DhanInstrumentMaster: mock master in use, skipping refresh")
            return len(self._mock_master)

        _log.info("DhanInstrumentMaster: downloading from %s", url)
        try:
            df = pd.read_csv(url, sep=",", low_memory=False)
            # Filter valid NSE instruments
            df = df[df["SEM_SMST_SECURITY_ID"].notna()].copy()
            df = df[df["SEM_TRADING_SYMBOL"].notna()].copy()
            df = df[df["SEM_EXM_EXCH_ID"].notna() & df["SEM_EXM_EXCH_ID"].astype(str).str.startswith("NSE")].copy()
            
            # Map columns
            df_insert = pd.DataFrame()
            df_insert["security_id"] = df["SEM_SMST_SECURITY_ID"].astype(str).str.strip()
            df_insert["symbol"] = df["SEM_TRADING_SYMBOL"].astype(str).str.strip()
            df_insert["exchange"] = df["SEM_EXM_EXCH_ID"].astype(str).str.strip()
            df_insert["instrument_type"] = df["SEM_INSTRUMENT_NAME"].astype(str).str.strip()
            df_insert["lot_size"] = pd.to_numeric(df["SEM_LOT_UNITS"], errors="coerce").fillna(0).astype(int)
            df_insert["tick_size"] = pd.to_numeric(df["SEM_TICK_SIZE"], errors="coerce").fillna(0.0)
            df_insert["expiry_date"] = pd.to_datetime(df.get("SEM_EXPIRY_DATE"), errors="coerce").dt.date
            df_insert["strike_price"] = pd.to_numeric(df["SEM_STRIKE_PRICE"], errors="coerce").fillna(0.0)
            df_insert["ingested_at"] = datetime.now(timezone.utc)
            
            # Populate cache
            for row in df_insert.itertuples():
                self._cache[(row.symbol, row.instrument_type)] = row.security_id
                
            rows_loaded = len(df_insert)
            with DuckDBClient(self._db_path) as db:
                db.conn.execute("""
                    INSERT INTO dhan_instruments
                    SELECT security_id, symbol, exchange, instrument_type, lot_size, tick_size, expiry_date, strike_price, ingested_at
                    FROM df_insert
                    ON CONFLICT (security_id, exchange, instrument_type) DO NOTHING
                """)
            return rows_loaded
        except Exception as exc:
            _log.error("DhanInstrumentMaster: download failed: %s", exc)
            return 0

    def get_security_id(self, symbol: str, instrument_type: str = "EQUITY") -> str:
        """Return Dhan security_id for (symbol, instrument_type).

        Checks in-memory cache first, then queries dhan_instruments DuckDB table.
        Returns empty string if not found (logged at WARNING).
        """
        key = (symbol, instrument_type)
        if key in self._cache:
            return self._cache[key]

        if self._mock_master is not None:
            _log.warning(
                "DhanInstrumentMaster: %r not in mock master (instrument_type=%r)",
                symbol, instrument_type,
            )
            return ""

        try:
            with DuckDBClient(self._db_path) as db:
                result = db.query_df(
                    "SELECT security_id FROM dhan_instruments "
                    "WHERE symbol = ? AND instrument_type = ? "
                    "AND exchange LIKE 'NSE%'",
                    (symbol, instrument_type),
                )
                if not result.empty:
                    sec_id = str(result.iloc[0]["security_id"])
                    self._cache[key] = sec_id
                    return sec_id
        except Exception as exc:  # noqa: BLE001
            _log.error("DhanInstrumentMaster: DB lookup failed for %r: %s", symbol, exc)

        _log.warning(
            "DhanInstrumentMaster: security_id not found for symbol=%r instrument_type=%r.  "
            "Run refresh() first.",
            symbol, instrument_type,
        )
        return ""

    def get_expiry_date(self, symbol: str, instrument_type: str) -> "date | None":
        """Return the nearest future expiry_date for (symbol, instrument_type).

        Used to populate fo_history.expiry_date when the API response does not
        include it.  Returns None if not found.
        """
        if self._mock_master is not None:
            return None
        try:
            with DuckDBClient(self._db_path) as db:
                result = db.query_df(
                    """SELECT expiry_date FROM dhan_instruments
                       WHERE symbol = ? AND instrument_type = ?
                         AND expiry_date >= CURRENT_DATE
                       ORDER BY expiry_date ASC LIMIT 1""",
                    (symbol, instrument_type),
                )
            if not result.empty and result.iloc[0]["expiry_date"] is not None:
                exp = result.iloc[0]["expiry_date"]
                return exp.date() if hasattr(exp, "date") else exp
        except Exception as exc:  # noqa: BLE001
            _log.debug("DhanInstrumentMaster: expiry_date lookup failed for %r: %s", symbol, exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DhanBulkDownloader
# ─────────────────────────────────────────────────────────────────────────────

class DhanBulkDownloader:
    """Paced, resumable bulk historical downloader for Dhan API.

    No SDK import — all HTTP calls use plain urllib.request.
    Credentials read from env vars: DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN.

    Resumability guarantee
    ----------------------
    If the process is killed mid-run (power failure, CTRL-C, OOM), the job
    resumes from the first non-DONE chunk on next invocation.  DONE chunks are
    never re-fetched or re-inserted.  Idempotent upsert prevents duplicate rows.

    Parameters
    ----------
    db_path:
        Path to the DuckDB database file.
    client_id / access_token:
        Dhan credentials.  Fall back to DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN env vars.
    request_delay_seconds:
        Sleep between successive API calls.  Set to ≥1.0 to respect Dhan rate limits.
    max_retries:
        Per-chunk retry attempts on HTTP error before marking chunk as FAILED.
    days_per_chunk:
        Calendar days per API request.  Dhan caps at ~90 days per call.
    _mock_fetch:
        Test hook.  callable(symbol, timeframe, start_date, end_date) → pd.DataFrame | None.
        When set, replaces all real HTTP calls.  Return None to simulate a failure.
    _mock_instrument_master:
        Test hook.  dict passed to DhanInstrumentMaster for security_id lookup.
    """

    def __init__(
        self,
        db_path: str | Any,
        *,
        client_id: str | None = None,
        access_token: str | None = None,
        base_url: str = "https://api.dhan.co/v2",
        request_delay_seconds: float = 1.0,
        max_retries: int = 3,
        days_per_chunk: int = 90,
        _mock_fetch: Callable | None = None,
        _mock_instrument_master: dict | None = None,
    ) -> None:
        self._db_path = str(db_path)
        self._client_id = client_id or os.environ.get("DHAN_CLIENT_ID", "")
        
        self._access_tokens = []
        if access_token:
            self._access_tokens.append(access_token)
        for k, v in os.environ.items():
            if k.startswith("DHAN_ACCESS_TOKEN") and v and v not in self._access_tokens:
                self._access_tokens.append(v)
        if not self._access_tokens:
            self._access_tokens.append("")

        self._base_url = base_url.rstrip("/")
        self._request_delay = request_delay_seconds
        self._max_retries = max_retries
        self._days_per_chunk = days_per_chunk
        self._mock_fetch = _mock_fetch

        self._instrument_master = DhanInstrumentMaster(
            db_path,
            _mock_master=_mock_instrument_master,
        )
        # Global rate-limiter: serialises real API calls across all threads so
        # request_delay_seconds is enforced between consecutive requests, not
        # just per-worker (which allows all 15 workers to burst simultaneously).
        self._api_throttle_lock = threading.Lock()
        self._api_last_call_at: float = 0.0
        self._ensure_schema()

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        symbols: list[str],
        timeframes: list[str],
        start_date: date,
        end_date: date,
    ) -> None:
        """Download equity intraday OHLCV for all (symbol, timeframe) combos.

        Resumable: pre-existing DONE chunks are skipped; pending/failed are retried.
        """
        import concurrent.futures
        
        chunks = self._build_chunks(start_date, end_date)
        total_jobs = len(symbols) * len(timeframes) * len(chunks)
        _log.info(
            "DhanBulkDownloader.run: %d symbols × %d timeframes × %d chunks = %d jobs",
            len(symbols), len(timeframes), len(chunks), total_jobs,
        )

        all_pending_jobs = []
        for symbol in symbols:
            for tf in timeframes:
                self._ensure_jobs(symbol, tf, chunks)
                pending = self._load_pending_jobs(symbol=symbol, timeframe=tf)
                for job in pending:
                    all_pending_jobs.append((symbol, tf, job["chunk_start"], job["chunk_end"]))

        def _worker(args):
            sym, tf, start, end = args
            self._process_job(sym, tf, start, end)

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            list(executor.map(_worker, all_pending_jobs))

    def run_fo(
        self,
        symbols: list[str],
        start_date: date,
        end_date: date,
    ) -> None:
        """Download F&O historical data for all symbols.

        Stores raw strike-level data in fo_history table.
        Call compute_fo_metrics() afterward to populate fo_daily_metrics.
        """
        import concurrent.futures
        
        chunks = self._build_chunks(start_date, end_date)
        _log.info(
            "DhanBulkDownloader.run_fo: %d symbols × %d chunks = %d jobs",
            len(symbols), len(chunks), len(symbols) * len(chunks),
        )
        
        all_pending_jobs = []
        for symbol in symbols:
            self._ensure_jobs(symbol, _FO_TIMEFRAME, chunks)
            pending = self._load_pending_jobs(symbol=symbol, timeframe=_FO_TIMEFRAME)
            for job in pending:
                all_pending_jobs.append((symbol, job["chunk_start"], job["chunk_end"]))

        def _worker(args):
            sym, start, end = args
            self._process_fo_job(sym, start, end)

        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            list(executor.map(_worker, all_pending_jobs))

    def compute_fo_metrics(self, symbols: list[str]) -> int:
        """Aggregate fo_history → fo_daily_metrics for given symbols.

        Returns total rows upserted.
        """
        now = datetime.now(timezone.utc)
        total = 0
        with DuckDBClient(self._db_path) as db:
            for symbol in symbols:
                try:
                    # Compute PCR, OI totals, net OI change, max-pain, futures settle
                    agg = db.query_df(
                        """
                        WITH ce AS (
                            SELECT date, SUM(oi) AS ce_oi, SUM(contracts) AS ce_vol,
                                   SUM(change_in_oi) AS ce_oi_chg
                            FROM fo_history
                            WHERE symbol = ? AND option_type = 'CE'
                            GROUP BY date
                        ),
                        pe AS (
                            SELECT date, SUM(oi) AS pe_oi, SUM(contracts) AS pe_vol,
                                   SUM(change_in_oi) AS pe_oi_chg
                            FROM fo_history
                            WHERE symbol = ? AND option_type = 'PE'
                            GROUP BY date
                        ),
                        fut AS (
                            SELECT date, settle_price AS futures_settle
                            FROM fo_history
                            WHERE symbol = ? AND option_type = 'FUT'
                              AND expiry_date = (
                                  SELECT MIN(expiry_date) FROM fo_history
                                  WHERE symbol = ? AND option_type = 'FUT'
                                    AND expiry_date >= date
                              )
                        )
                        SELECT ce.date,
                               COALESCE(ce.ce_oi, 0) AS total_ce_oi,
                               COALESCE(pe.pe_oi, 0) AS total_pe_oi,
                               CASE WHEN pe.pe_oi > 0
                                    THEN ce.ce_oi::DOUBLE / pe.pe_oi
                                    ELSE NULL END AS pcr_oi,
                               CASE WHEN pe.pe_vol > 0
                                    THEN ce.ce_vol::DOUBLE / pe.pe_vol
                                    ELSE NULL END AS pcr_volume,
                               COALESCE(ce.ce_oi_chg, 0) + COALESCE(pe.pe_oi_chg, 0)
                                    AS net_oi_change,
                               fut.futures_settle
                        FROM ce
                        LEFT JOIN pe  ON ce.date = pe.date
                        LEFT JOIN fut ON ce.date = fut.date
                        ORDER BY ce.date
                        """,
                        (symbol, symbol, symbol, symbol),
                    )
                except Exception as exc:  # noqa: BLE001
                    _log.error("compute_fo_metrics: aggregation failed for %r: %s", symbol, exc)
                    continue

                # Compute max-pain per date (strike that minimises total option pain)
                max_pain_rows = self._compute_max_pain(db, symbol)

                for _, row in agg.iterrows():
                    d = row["date"]
                    if hasattr(d, "date"):
                        d = d.date()
                    mp = max_pain_rows.get(d, None)
                    try:
                        db.execute(
                            """
                            INSERT INTO fo_daily_metrics
                              (symbol, date, pcr_oi, pcr_volume, total_ce_oi, total_pe_oi,
                               net_oi_change, max_pain_strike, futures_settle, source, ingested_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'dhan_fo_metrics', ?)
                            ON CONFLICT (symbol, date) DO NOTHING
                            """,
                            (
                                symbol, d,
                                _nan_to_none(row.get("pcr_oi")),
                                _nan_to_none(row.get("pcr_volume")),
                                int(row.get("total_ce_oi", 0) or 0),
                                int(row.get("total_pe_oi", 0) or 0),
                                int(row.get("net_oi_change", 0) or 0),
                                mp,
                                _nan_to_none(row.get("futures_settle")),
                                now,
                            ),
                        )
                        total += 1
                    except Exception as exc:  # noqa: BLE001
                        _log.debug("compute_fo_metrics: upsert failed for %r %s: %s", symbol, d, exc)

        _log.info("compute_fo_metrics: upserted %d rows total", total)
        return total

    def coverage_report(self) -> str:
        """Return a string coverage matrix (symbol × timeframe → DONE/PARTIAL/FAILED/PENDING).

        Also logs the report at INFO level.
        """
        try:
            with DuckDBClient(self._db_path) as db:
                df = db.query_df(
                    """
                    SELECT symbol, timeframe,
                           SUM(CASE WHEN status = 'done'    THEN 1 ELSE 0 END) AS done,
                           SUM(CASE WHEN status = 'partial' THEN 1 ELSE 0 END) AS partial,
                           SUM(CASE WHEN status = 'failed'  THEN 1 ELSE 0 END) AS failed,
                           SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending,
                           COUNT(*) AS total
                    FROM download_jobs
                    GROUP BY symbol, timeframe
                    ORDER BY symbol, timeframe
                    """
                )
        except Exception as exc:  # noqa: BLE001
            return f"Coverage report unavailable: {exc}"

        if df.empty:
            return "Coverage report: no download_jobs found."

        lines = [
            "=" * 72,
            "DHAN BULK DOWNLOAD COVERAGE REPORT",
            f"{'Symbol':<16} {'TF':<6} {'Done':>6} {'Partial':>8} {'Failed':>7} {'Pending':>8} {'Total':>7} Status",
            "-" * 72,
        ]
        for _, row in df.iterrows():
            done    = int(row["done"])
            partial = int(row["partial"])
            failed  = int(row["failed"])
            pending = int(row["pending"])
            total   = int(row["total"])
            if done == total:
                status = "COMPLETE"
            elif done == 0 and partial == 0 and failed == 0:
                status = "NOT_STARTED"
            elif failed > 0:
                status = "HAS_ERRORS"
            else:
                status = "IN_PROGRESS"
            lines.append(
                f"{str(row['symbol']):<16} {str(row['timeframe']):<6} "
                f"{done:>6} {partial:>8} {failed:>7} {pending:>8} {total:>7}  {status}"
            )
        lines.append("=" * 72)
        report = "\n".join(lines)
        _log.info("\n%s", report)
        return report

    # ── Internal: schema ─────────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        with DuckDBClient(self._db_path) as db:
            db.execute_schema(DUCKDB_SCHEMA_SQL)

    # ── Internal: chunk building ──────────────────────────────────────────────

    def _build_chunks(self, start_date: date, end_date: date) -> list[tuple[date, date]]:
        """Split [start_date, end_date] into consecutive chunks of days_per_chunk days.

        Each chunk: (chunk_start, chunk_end) both inclusive.
        Chunks never overlap — chunk N+1 starts on chunk N's end + 1 day.
        """
        chunks: list[tuple[date, date]] = []
        current = start_date
        delta = timedelta(days=self._days_per_chunk - 1)
        while current <= end_date:
            chunk_end = min(current + delta, end_date)
            chunks.append((current, chunk_end))
            current = chunk_end + timedelta(days=1)
        return chunks

    # ── Internal: job management ──────────────────────────────────────────────

    def _ensure_jobs(
        self,
        symbol: str,
        timeframe: str,
        chunks: list[tuple[date, date]],
    ) -> None:
        """Insert PENDING rows for chunks not already present in download_jobs."""
        with DuckDBClient(self._db_path) as db:
            for chunk_start, chunk_end in chunks:
                db.execute(
                    """
                    INSERT INTO download_jobs
                      (symbol, timeframe, chunk_start, chunk_end, status)
                    VALUES (?, ?, ?, ?, 'pending')
                    ON CONFLICT (symbol, timeframe, chunk_start) DO NOTHING
                    """,
                    (symbol, timeframe, chunk_start, chunk_end),
                )

    def _load_pending_jobs(
        self,
        symbol: str | None = None,
        timeframe: str | None = None,
    ) -> list[dict]:
        """Return pending/failed jobs sorted by chunk_start ASC."""
        clauses = ["status IN ('pending', 'failed')"]
        params: list[Any] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol)
        if timeframe is not None:
            clauses.append("timeframe = ?")
            params.append(timeframe)
        where = " AND ".join(clauses)
        with DuckDBClient(self._db_path) as db:
            df = db.query_df(
                f"SELECT * FROM download_jobs WHERE {where} ORDER BY chunk_start ASC",
                tuple(params) if params else None,
            )
        if df.empty:
            return []
        return df.to_dict(orient="records")

    def _mark_job(
        self,
        symbol: str,
        timeframe: str,
        chunk_start: date,
        status: str,
        rows: int = 0,
        error: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        with DuckDBClient(self._db_path) as db:
            db.execute(
                """
                UPDATE download_jobs
                SET status = ?, rows_downloaded = ?, last_attempted = ?, error = ?
                WHERE symbol = ? AND timeframe = ? AND chunk_start = ?
                """,
                (status, rows, now, error, symbol, timeframe, chunk_start),
            )

    # ── Internal: equity OHLCV download ──────────────────────────────────────

    def _process_job(
        self,
        symbol: str,
        timeframe: str,
        chunk_start: date,
        chunk_end: date,
    ) -> None:
        _log.debug(
            "DhanBulkDownloader: %s/%s %s→%s",
            symbol, timeframe, chunk_start, chunk_end,
        )
        df = self._download_chunk(symbol, timeframe, chunk_start, chunk_end)

        if df is None:
            self._mark_job(symbol, timeframe, chunk_start, "failed",
                           error="all retries exhausted")
            return

        if df.empty:
            self._mark_job(symbol, timeframe, chunk_start, "partial",
                           rows=0, error="API returned empty data")
            return

        rows = self._upsert_intraday(df, symbol, timeframe)
        
        if timeframe == "1" and not df.empty:
            self._derive_and_upsert_timeframes(df, symbol)
            
        self._mark_job(symbol, timeframe, chunk_start, "done", rows=rows)

    def _download_chunk(
        self,
        symbol: str,
        timeframe: str,
        chunk_start: date,
        chunk_end: date,
    ) -> pd.DataFrame | None:
        """Fetch one chunk via mock or real HTTP, with exponential backoff."""
        if self._mock_fetch is not None:
            return self._mock_fetch(symbol, timeframe, chunk_start, chunk_end)

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return self._do_http_fetch(symbol, timeframe, chunk_start, chunk_end)
            except urllib.error.HTTPError as exc:
                last_exc = exc
                wait = _BACKOFF_BASE * (2 ** attempt)
                _log.warning(
                    "HTTP %d for %s/%s chunk %s→%s (attempt %d/%d); wait %.0fs",
                    exc.code, symbol, timeframe, chunk_start, chunk_end,
                    attempt + 1, self._max_retries, wait,
                )
                time.sleep(wait)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = _BACKOFF_BASE * (2 ** attempt)
                _log.warning(
                    "Fetch error %s/%s chunk %s→%s (attempt %d/%d): %s",
                    symbol, timeframe, chunk_start, chunk_end,
                    attempt + 1, self._max_retries, exc,
                )
                if attempt < self._max_retries - 1:
                    time.sleep(wait)
        _log.error(
            "All %d attempts failed for %s/%s chunk %s→%s: %s",
            self._max_retries, symbol, timeframe, chunk_start, chunk_end, last_exc,
        )
        return None

    def _do_http_fetch(
        self,
        symbol: str,
        timeframe: str,
        chunk_start: date,
        chunk_end: date,
    ) -> pd.DataFrame:
        """Call Dhan REST API for OHLCV data; return DataFrame.

        TODO (on key arrival): verify exact endpoint, request body fields, and
        response JSON structure against live Dhan API docs.
        Expected response: {"open": [...], "high": [...], "low": [...],
                            "close": [...], "volume": [...], "timestamp": [...]}
        """
        sec_id = self._instrument_master.get_security_id(symbol, "EQUITY")
        if not sec_id:
            raise ValueError(f"No security_id for {symbol!r} — run refresh() first")

        if timeframe == "D":
            url = f"{self._base_url}{_DHAN_HISTORICAL_PATH}"
            body = {
                "securityId": sec_id,
                "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY",
                "fromDate": chunk_start.strftime("%Y-%m-%d"),
                "toDate":   chunk_end.strftime("%Y-%m-%d"),
            }
        else:
            url = f"{self._base_url}{_DHAN_INTRADAY_PATH}"
            body = {
                "securityId": sec_id,
                "exchangeSegment": "NSE_EQ",
                "instrument": "EQUITY",
                "interval": timeframe,
                "fromDate": chunk_start.strftime("%Y-%m-%d"),
                "toDate":   chunk_end.strftime("%Y-%m-%d"),
            }

        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=self._get_headers(),
            method="POST",
        )
        self._throttle()
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        return self._parse_ohlcv_response(raw, symbol, timeframe)

    def _parse_ohlcv_response(
        self, raw: dict, symbol: str, timeframe: str
    ) -> pd.DataFrame:
        """Convert Dhan OHLCV JSON to a DataFrame ready for upsert.

        Expected format:
          {"open": [...], "high": [...], "low": [...],
           "close": [...], "volume": [...], "timestamp": [...]}
        """
        opens     = raw.get("open",      [])
        highs     = raw.get("high",      [])
        lows      = raw.get("low",       [])
        closes    = raw.get("close",     [])
        volumes   = raw.get("volume",    [])
        timestamps = raw.get("timestamp", [])

        if not timestamps:
            return pd.DataFrame()

        n = len(timestamps)
        df = pd.DataFrame({
            "ts":     pd.to_datetime(timestamps, unit="s", utc=True),
            "open":   [float(v) for v in (opens  if len(opens)  == n else [None] * n)],
            "high":   [float(v) for v in (highs  if len(highs)  == n else [None] * n)],
            "low":    [float(v) for v in (lows   if len(lows)   == n else [None] * n)],
            "close":  [float(v) for v in (closes if len(closes) == n else [None] * n)],
            "volume": [int(v)   for v in (volumes if len(volumes) == n else [0] * n)],
        })
        return df

    def _upsert_intraday(
        self, df: pd.DataFrame, symbol: str, timeframe: str
    ) -> int:
        """Upsert rows into ohlcv_intraday.  Returns number of rows inserted."""
        if df.empty: return 0
        now = datetime.now(timezone.utc)
        insert_df = df.copy()
        if "ts" not in insert_df.columns:
            insert_df = insert_df.reset_index()
        insert_df["symbol"] = symbol
        insert_df["timeframe"] = timeframe
        insert_df["source"] = "dhan_historical"
        insert_df["ingested_at"] = now
        
        rows_inserted = 0
        with DuckDBClient(self._db_path) as db:
            try:
                db.conn.register("temp_intraday_df", insert_df)
                db.conn.execute(
                    """
                    INSERT INTO ohlcv_intraday
                      (symbol, timeframe, ts, open, high, low, close, volume, source, ingested_at)
                    SELECT symbol, timeframe, CAST(ts AS TIMESTAMP), CAST(open AS DOUBLE), CAST(high AS DOUBLE), CAST(low AS DOUBLE), CAST(close AS DOUBLE), CAST(volume AS INTEGER), source, CAST(ingested_at AS TIMESTAMP)
                    FROM temp_intraday_df
                    ON CONFLICT (symbol, timeframe, ts) DO NOTHING
                    """
                )
                db.conn.unregister("temp_intraday_df")
                rows_inserted = len(insert_df)
            except Exception as exc:
                _log.error("upsert_intraday: dataframe insert failed: %s", exc)
        return rows_inserted

    def _derive_and_upsert_timeframes(self, df_1m: pd.DataFrame, symbol: str) -> None:
        """Derive 5, 15, 25, 60 minute timeframes from 1-minute data and upsert."""
        if df_1m.empty:
            return
            
        # Ensure index is datetime for resampling
        df = df_1m.set_index('ts')
        
        agg_dict = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }
        
        derived_timeframes = {
            "5": "5min",
            "15": "15min",
            "25": "25min",
            "60": "60min"
        }
        
        for tf_str, freq in derived_timeframes.items():
            try:
                resampled = df.resample(freq).agg(agg_dict).dropna()
                if not resampled.empty:
                    # Reset index to bring 'ts' back as a column
                    resampled_df = resampled.reset_index()
                    self._upsert_intraday(resampled_df, symbol, tf_str)
                    
            except Exception as exc:
                _log.error("Failed to derive %s timeframe for %s: %s", tf_str, symbol, exc)

    # ── Internal: F&O download ────────────────────────────────────────────────

    def _process_fo_job(
        self,
        symbol: str,
        chunk_start: date,
        chunk_end: date,
    ) -> None:
        _log.debug("DhanBulkDownloader: FO %s %s→%s", symbol, chunk_start, chunk_end)
        df = self._download_fo_chunk(symbol, chunk_start, chunk_end)

        if df is None:
            self._mark_job(symbol, _FO_TIMEFRAME, chunk_start, "failed",
                           error="all retries exhausted")
            return
        if df.empty:
            self._mark_job(symbol, _FO_TIMEFRAME, chunk_start, "partial",
                           rows=0, error="API returned empty data")
            return

        rows = self._upsert_fo_history(df, symbol)
        self._mark_job(symbol, _FO_TIMEFRAME, chunk_start, "done", rows=rows)

    def _download_fo_chunk(
        self,
        symbol: str,
        chunk_start: date,
        chunk_end: date,
    ) -> pd.DataFrame | None:
        if self._mock_fetch is not None:
            return self._mock_fetch(symbol, _FO_TIMEFRAME, chunk_start, chunk_end)
        # Real HTTP call — TODO: verify Dhan F&O endpoint on key arrival
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return self._do_fo_http_fetch(symbol, chunk_start, chunk_end)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                wait = _BACKOFF_BASE * (2 ** attempt)
                if attempt < self._max_retries - 1:
                    time.sleep(wait)
        _log.error("All %d FO attempts failed for %s: %s", self._max_retries, symbol, last_exc)
        return None

    def _do_fo_http_fetch(
        self, symbol: str, chunk_start: date, chunk_end: date
    ) -> pd.DataFrame:
        """Fetch F&O historical data for one symbol/chunk.

        KNOWN LIMITATION (confirmed on first real API contact):
          The Dhan /charts/historical endpoint for FUTSTK returns per-contract
          OHLCV only: {"open":[], "high":[], "low":[], "close":[], "volume":[],
          "timestamp":[]}.  It does NOT return open interest (OI), change_in_oi,
          strike_price, or expiry_date.  PCR, max-pain, and net-OI-change features
          therefore cannot be computed from this endpoint.  The F&O gate should
          remain closed until an OI data source is identified.

        The FUTSTK security_id (not the equity security_id) must be used here.
        Each futures contract has its own security_id in the instrument master.
        """
        # FUTSTK has its own security_id — equity ID (e.g. 2885 for RELIANCE)
        # returns HTTP 400 "Missing required fields" from NSE_FNO endpoint.
        sec_id = self._instrument_master.get_security_id(symbol, "FUTSTK")
        if not sec_id:
            _log.warning(
                "DhanBulkDownloader: no FUTSTK security_id for %r — "
                "run instrument_master.refresh() first. Skipping F&O chunk.",
                symbol,
            )
            return pd.DataFrame()

        url = f"{self._base_url}{_DHAN_HISTORICAL_PATH}"
        body = {
            "securityId":      sec_id,
            "exchangeSegment": "NSE_FNO",
            "instrument":      "FUTSTK",
            "fromDate":        chunk_start.strftime("%Y-%m-%d"),
            "toDate":          chunk_end.strftime("%Y-%m-%d"),
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=self._get_headers(),
            method="POST",
        )
        self._throttle()
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))

        expiry_date = self._instrument_master.get_expiry_date(symbol, "FUTSTK")
        return self._parse_fo_response(raw, symbol, expiry_date=expiry_date)

    def _parse_fo_response(
        self, raw: dict, symbol: str, *, expiry_date: "date | None" = None
    ) -> pd.DataFrame:
        """Parse Dhan F&O JSON response.

        Real API format (confirmed on first contact):
          {"open": [...], "high": [...], "low": [...], "close": [...],
           "volume": [...], "timestamp": [...]}
          — same structure as equity OHLCV; OI is NOT included.

        Legacy mock format ({"data": [...]}) is also accepted for backward compat.
        """
        # Real API format: same OHLCV structure as equity
        if "open" in raw or "timestamp" in raw:
            opens      = raw.get("open",      [])
            highs      = raw.get("high",      [])
            lows       = raw.get("low",       [])
            closes     = raw.get("close",     [])
            volumes    = raw.get("volume",    [])
            timestamps = raw.get("timestamp", [])
            if not timestamps:
                return pd.DataFrame()
            n = len(timestamps)
            rows = []
            for i in range(n):
                ts = pd.to_datetime(timestamps[i], unit="s", utc=True).date()
                rows.append({
                    "date":         ts,
                    "expiry_date":  expiry_date,
                    "strike_price": 0.0,
                    "option_type":  "FUT",
                    "open":         float(opens[i])  if i < len(opens)  else None,
                    "high":         float(highs[i])  if i < len(highs)  else None,
                    "low":          float(lows[i])   if i < len(lows)   else None,
                    "close":        float(closes[i]) if i < len(closes) else None,
                    "settle_price": float(closes[i]) if i < len(closes) else None,
                    "contracts":    int(volumes[i])  if i < len(volumes) else 0,
                    "oi":           None,
                    "change_in_oi": None,
                })
            return pd.DataFrame(rows)

        # Legacy mock format: {"data": [{...per-row dicts...}]}
        data = raw.get("data", [])
        if not data:
            return pd.DataFrame()
        rows = []
        for item in data:
            rows.append({
                "date":         _safe_date(item.get("date")),
                "expiry_date":  _safe_date(item.get("expiry")),
                "strike_price": float(item.get("strike", 0) or 0),
                "option_type":  str(item.get("option_type", "FUT")),
                "open":         _safe_float(item.get("open")),
                "high":         _safe_float(item.get("high")),
                "low":          _safe_float(item.get("low")),
                "close":        _safe_float(item.get("close")),
                "settle_price": _safe_float(item.get("settle_price")),
                "contracts":    _safe_int(item.get("contracts")),
                "oi":           _safe_int(item.get("oi")),
                "change_in_oi": _safe_int(item.get("change_in_oi")),
            })
        return pd.DataFrame(rows)

    def _upsert_fo_history(self, df: pd.DataFrame, symbol: str) -> int:
        if df.empty: return 0
        now = datetime.now(timezone.utc)
        insert_df = df.copy()
        insert_df["symbol"] = symbol
        insert_df["source"] = "dhan_historical"
        insert_df["ingested_at"] = now
        
        rows_inserted = 0
        with DuckDBClient(self._db_path) as db:
            try:
                db.conn.register("temp_fo_df", insert_df)
                db.conn.execute(
                    """
                    INSERT INTO fo_history
                      (symbol, date, expiry_date, strike_price, option_type,
                       open, high, low, close, settle_price, contracts, oi,
                       change_in_oi, source, ingested_at)
                    SELECT symbol, CAST(date AS DATE), CAST(expiry_date AS DATE), CAST(strike_price AS DOUBLE), CAST(option_type AS VARCHAR),
                           CAST(open AS DOUBLE), CAST(high AS DOUBLE), CAST(low AS DOUBLE), CAST(close AS DOUBLE), CAST(settle_price AS DOUBLE), CAST(contracts AS INTEGER), CAST(oi AS INTEGER),
                           CAST(change_in_oi AS INTEGER), source, CAST(ingested_at AS TIMESTAMP)
                    FROM temp_fo_df
                    ON CONFLICT (symbol, date, expiry_date, strike_price, option_type) DO NOTHING
                    """
                )
                db.conn.unregister("temp_fo_df")
                rows_inserted = len(insert_df)
            except Exception as exc:
                _log.error("upsert_fo_history: dataframe insert failed: %s", exc)
        return rows_inserted

    # ── Internal: max-pain calculation ───────────────────────────────────────

    def _compute_max_pain(self, db: DuckDBClient, symbol: str) -> dict[date, float]:
        """Compute max-pain strike per date from fo_history.

        Max-pain = strike where total option pain (ITM payout) is minimised.
        """
        try:
            df = db.query_df(
                """
                SELECT date, strike_price, option_type, oi
                FROM fo_history
                WHERE symbol = ? AND option_type IN ('CE', 'PE')
                ORDER BY date, strike_price
                """,
                (symbol,),
            )
        except Exception as exc:  # noqa: BLE001
            _log.debug("_compute_max_pain: query failed for %r: %s", symbol, exc)
            return {}

        if df.empty:
            return {}

        result: dict[date, float] = {}
        for grp_date, grp in df.groupby("date"):
            strikes = sorted(grp["strike_price"].unique())
            if not strikes:
                continue
            min_pain = float("inf")
            best_strike = strikes[0]
            for candidate in strikes:
                pain = 0.0
                for _, row in grp.iterrows():
                    oi_val = float(row["oi"] or 0)
                    s = float(row["strike_price"])
                    if row["option_type"] == "CE":
                        pain += max(0.0, candidate - s) * oi_val
                    else:
                        pain += max(0.0, s - candidate) * oi_val
                if pain < min_pain:
                    min_pain = pain
                    best_strike = candidate
            d = grp_date.date() if hasattr(grp_date, "date") else grp_date
            result[d] = float(best_strike)
        return result

    # ── Internal: HTTP helpers ────────────────────────────────────────────────

    def _throttle(self) -> None:
        """Enforce request_delay_seconds globally across all worker threads.

        Acquires a lock so only one thread issues an API call at a time, and
        sleeps for the remainder of request_delay_seconds since the last call.
        This prevents the 15-worker ThreadPoolExecutor from bursting all 15
        requests simultaneously before any per-worker sleep fires.
        """
        with self._api_throttle_lock:
            now = time.monotonic()
            gap = self._request_delay - (now - self._api_last_call_at)
            if gap > 0:
                time.sleep(gap)
            self._api_last_call_at = time.monotonic()

    def _get_headers(self) -> dict:
        import random
        token = random.choice(self._access_tokens)
        return {
            "access-token":  token,
            "client-id":     self._client_id,
            "Content-Type":  "application/json",
            "Accept":        "application/json",
        }


# ── Module-level helpers ──────────────────────────────────────────────────────

def _safe_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _safe_date(v: Any) -> date | None:
    if v is None:
        return None
    if isinstance(v, date):
        return v
    try:
        return date.fromisoformat(str(v)[:10])
    except (ValueError, TypeError):
        return None


def _nan_to_none(v: Any) -> Any:
    """Convert NaN/inf to None for SQL insertion."""
    try:
        import math
        if v is None:
            return None
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None
