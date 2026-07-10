"""DuckDB and SQLite schema definitions.

All statements use CREATE TABLE IF NOT EXISTS so they are safe to re-run
(idempotent).  Run once at startup via the respective client's execute_schema().
"""

# ── DuckDB — analytical / larger-than-RAM data ────────────────────────────────
DUCKDB_SCHEMA_SQL = """
-- Daily OHLCV from NSE bhavcopy (Phase 1+).
-- 'date' is the trading date; bar is knowable from midnight of date+1.
-- 'series' distinguishes EQ / BE / BL etc. (filter to EQ for equity).
CREATE TABLE IF NOT EXISTS ohlcv_daily (
    symbol        VARCHAR      NOT NULL,
    date          DATE         NOT NULL,
    open          DOUBLE       NOT NULL,
    high          DOUBLE       NOT NULL,
    low           DOUBLE       NOT NULL,
    close         DOUBLE       NOT NULL,
    volume        BIGINT       NOT NULL,
    traded_value  DOUBLE,
    series        VARCHAR,
    delivery_qty  BIGINT,
    source        VARCHAR      NOT NULL DEFAULT 'nse_bhavcopy',
    ingested_at   TIMESTAMPTZ  NOT NULL,
    PRIMARY KEY (symbol, date)
);

-- Minute OHLCV — Kite historical API or supplied dataset (Phase 1+).
-- 'ts' is the bar-open timestamp stored in UTC; convert to IST (UTC+05:30) for display.
CREATE TABLE IF NOT EXISTS ohlcv_minute (
    symbol       VARCHAR      NOT NULL,
    ts           TIMESTAMPTZ  NOT NULL,
    open         DOUBLE       NOT NULL,
    high         DOUBLE       NOT NULL,
    low          DOUBLE       NOT NULL,
    close        DOUBLE       NOT NULL,
    volume       BIGINT       NOT NULL,
    source       VARCHAR      NOT NULL,
    ingested_at  TIMESTAMPTZ  NOT NULL,
    PRIMARY KEY (symbol, ts)
);

-- Raw ticks — live WebSocket ingest and historical replay.
-- No PRIMARY KEY: multiple fills can share the same millisecond timestamp.
CREATE TABLE IF NOT EXISTS ticks (
    symbol       VARCHAR      NOT NULL,
    ts           TIMESTAMPTZ  NOT NULL,
    ltp          DOUBLE       NOT NULL,
    volume       BIGINT,
    bid          DOUBLE,
    ask          DOUBLE,
    oi           BIGINT,
    source       VARCHAR      NOT NULL,
    ingested_at  TIMESTAMPTZ  NOT NULL
);

-- News with first-knowable timestamp.
-- 'first_knowable_at' = max(published_at, ingestion_time) + deliberate delay buffer.
-- Deduplication is performed at ingest: one story, one news_id.
CREATE TABLE IF NOT EXISTS news (
    news_id           VARCHAR      PRIMARY KEY,
    headline          VARCHAR      NOT NULL,
    body              TEXT,
    source            VARCHAR      NOT NULL,
    published_at      TIMESTAMPTZ  NOT NULL,
    first_knowable_at TIMESTAMPTZ  NOT NULL,
    symbols           VARCHAR[],
    ingested_at       TIMESTAMPTZ  NOT NULL
);

-- Option chain snapshots cached every ~60 s.
-- Strike stored as DOUBLE to handle half-strikes (e.g. 23750.0).
CREATE TABLE IF NOT EXISTS option_chain (
    symbol       VARCHAR      NOT NULL,
    expiry       DATE         NOT NULL,
    strike       DOUBLE       NOT NULL,
    option_type  VARCHAR      NOT NULL,
    ts           TIMESTAMPTZ  NOT NULL,
    ltp          DOUBLE,
    oi           BIGINT,
    oi_change    BIGINT,
    volume       BIGINT,
    iv           DOUBLE,
    source       VARCHAR      NOT NULL,
    PRIMARY KEY (symbol, expiry, strike, option_type, ts)
);

-- Corporate actions (splits, bonuses) for backward price adjustment.
-- Used exclusively by AdjustedBhavCopySource — BACKTEST USE ONLY.
-- adjustment_factor: multiply pre-ex-date OHLC prices by this value.
--   SPLIT    factor = new_face_value / old_face_value  (< 1.0; e.g. 10→2 FV → 0.20)
--   BONUS    factor = existing / (existing + bonus)    (< 1.0; e.g. 1:1 bonus → 0.50)
--   DIVIDEND factor = 1.0  (intentionally not price-adjusted; decision is logged)
--   RIGHTS   factor = 1.0  (not adjusted in Phase 1.5)
--   OTHER    factor = 1.0  (unrecognized purpose string)
CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol            VARCHAR      NOT NULL,
    ex_date           DATE         NOT NULL,
    action_type       VARCHAR      NOT NULL,
    adjustment_factor DOUBLE       NOT NULL,
    description       VARCHAR,
    source            VARCHAR      NOT NULL DEFAULT 'nse_corporate_actions',
    ingested_at       TIMESTAMPTZ  NOT NULL,
    PRIMARY KEY (symbol, ex_date, action_type)
);

-- Intraday OHLCV bars from Dhan (timeframe: '1','5','15','25','60' min, or 'D' for daily).
-- Kept separate from ohlcv_daily (NSE bhavcopy) so sources never conflict.
-- ts is the bar-OPEN timestamp in UTC.
CREATE TABLE IF NOT EXISTS ohlcv_intraday (
    symbol      VARCHAR      NOT NULL,
    timeframe   VARCHAR      NOT NULL,
    ts          TIMESTAMPTZ  NOT NULL,
    open        DOUBLE       NOT NULL,
    high        DOUBLE       NOT NULL,
    low         DOUBLE       NOT NULL,
    close       DOUBLE       NOT NULL,
    volume      BIGINT       NOT NULL,
    source      VARCHAR      NOT NULL,
    ingested_at TIMESTAMPTZ  NOT NULL,
    PRIMARY KEY (symbol, timeframe, ts)
);

-- F&O historical data from Dhan (end-of-day OI/price per strike).
-- option_type: 'CE' | 'PE' | 'FUT'
CREATE TABLE IF NOT EXISTS fo_history (
    symbol        VARCHAR      NOT NULL,
    date          DATE         NOT NULL,
    expiry_date   DATE         NOT NULL,
    strike_price  DOUBLE       NOT NULL,
    option_type   VARCHAR      NOT NULL,
    open          DOUBLE,
    high          DOUBLE,
    low           DOUBLE,
    close         DOUBLE,
    settle_price  DOUBLE,
    contracts     BIGINT,
    oi            BIGINT,
    change_in_oi  BIGINT,
    source        VARCHAR      NOT NULL,
    ingested_at   TIMESTAMPTZ  NOT NULL,
    PRIMARY KEY (symbol, date, expiry_date, strike_price, option_type)
);

-- F&O daily aggregates (PCR, OI totals, max-pain, futures basis).
-- Populated by DhanBulkDownloader.compute_fo_metrics() from fo_history.
-- Used by DhanFOMetricsSource — BACKTEST USE ONLY.
CREATE TABLE IF NOT EXISTS fo_daily_metrics (
    symbol          VARCHAR      NOT NULL,
    date            DATE         NOT NULL,
    pcr_oi          DOUBLE,
    pcr_volume      DOUBLE,
    total_ce_oi     BIGINT,
    total_pe_oi     BIGINT,
    net_oi_change   BIGINT,
    max_pain_strike DOUBLE,
    futures_settle  DOUBLE,
    source          VARCHAR      NOT NULL,
    ingested_at     TIMESTAMPTZ  NOT NULL,
    PRIMARY KEY (symbol, date)
);

-- Bulk download progress tracker (one row per symbol × timeframe × chunk).
-- status: 'pending' | 'done' | 'failed' | 'partial'
-- Resumability: on restart, only 'pending' and 'failed' rows are retried.
CREATE TABLE IF NOT EXISTS download_jobs (
    symbol          VARCHAR      NOT NULL,
    timeframe       VARCHAR      NOT NULL,
    chunk_start     DATE         NOT NULL,
    chunk_end       DATE         NOT NULL,
    status          VARCHAR      NOT NULL DEFAULT 'pending',
    rows_downloaded INTEGER               DEFAULT 0,
    last_attempted  TIMESTAMPTZ,
    error           VARCHAR,
    PRIMARY KEY (symbol, timeframe, chunk_start)
);

-- Dhan instrument master (ticker → numeric security_id).
-- Populated by DhanInstrumentMaster.refresh() from Dhan's CSV endpoint.
-- Required because Dhan REST API uses numeric IDs, not NSE ticker strings.
CREATE TABLE IF NOT EXISTS dhan_instruments (
    security_id     VARCHAR      NOT NULL,
    symbol          VARCHAR      NOT NULL,
    exchange        VARCHAR      NOT NULL,
    instrument_type VARCHAR      NOT NULL,
    lot_size        INTEGER,
    tick_size       DOUBLE,
    expiry_date     DATE,
    strike_price    DOUBLE,
    ingested_at     TIMESTAMPTZ  NOT NULL,
    PRIMARY KEY (security_id, exchange, instrument_type)
);
"""

# ── SQLite — state, logs, small reference tables ──────────────────────────────
SQLITE_SCHEMA_SQL = """
-- Experiment run log.  Every run captures enough to fully reproduce it.
CREATE TABLE IF NOT EXISTS runs (
    run_id                   TEXT     PRIMARY KEY,
    created_at               TEXT     NOT NULL,
    completed_at             TEXT,
    status                   TEXT     NOT NULL DEFAULT 'running',
    config_hash              TEXT     NOT NULL,
    config_json              TEXT     NOT NULL,
    git_commit               TEXT,
    data_source              TEXT     NOT NULL,
    data_source_limitations  TEXT     NOT NULL,
    data_granularity         TEXT     NOT NULL,
    random_seed              INTEGER  NOT NULL,
    date_range_start         TEXT,
    date_range_end           TEXT,
    universe_size            INTEGER,
    universe_hash            TEXT,
    metrics_json             TEXT,
    error                    TEXT,
    notes                    TEXT
);

-- Model predictions (one row per symbol × horizon × cycle).
CREATE TABLE IF NOT EXISTS predictions (
    pred_id        TEXT     PRIMARY KEY,
    run_id         TEXT     NOT NULL,
    symbol         TEXT     NOT NULL,
    predicted_at   TEXT     NOT NULL,
    horizon_min    INTEGER  NOT NULL,
    q10            REAL,
    q25            REAL,
    q50            REAL,
    q75            REAL,
    q90            REAL,
    scenario       TEXT,
    scenario_probs TEXT,
    confidence     REAL,
    features_json  TEXT,
    shap_json      TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Paper trades.  net_pnl = gross_pnl - all costs (see costs_json for breakdown).
CREATE TABLE IF NOT EXISTS trades (
    trade_id        TEXT     PRIMARY KEY,
    run_id          TEXT     NOT NULL,
    symbol          TEXT     NOT NULL,
    direction       TEXT     NOT NULL,
    order_type      TEXT     NOT NULL,
    quantity        INTEGER  NOT NULL,
    entry_time      TEXT     NOT NULL,
    entry_price     REAL     NOT NULL,
    entry_slippage  REAL     NOT NULL,
    exit_time       TEXT,
    exit_price      REAL,
    exit_slippage   REAL,
    exit_reason     TEXT,
    gross_pnl       REAL,
    net_pnl         REAL,
    costs_json      TEXT,
    pred_id         TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Signal log — every generated signal, whether acted on or not.
CREATE TABLE IF NOT EXISTS signals (
    signal_id     TEXT     PRIMARY KEY,
    run_id        TEXT     NOT NULL,
    symbol        TEXT     NOT NULL,
    generated_at  TEXT     NOT NULL,
    signal_type   TEXT     NOT NULL,
    strength      REAL,
    pred_id       TEXT,
    acted_on      INTEGER  NOT NULL DEFAULT 0,
    notes         TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Open positions — persisted continuously so the engine survives a restart.
CREATE TABLE IF NOT EXISTS positions (
    position_id   TEXT     PRIMARY KEY,
    run_id        TEXT     NOT NULL,
    trade_id      TEXT     NOT NULL,
    symbol        TEXT     NOT NULL,
    direction     TEXT     NOT NULL,
    quantity      INTEGER  NOT NULL,
    entry_price   REAL     NOT NULL,
    entry_time    TEXT     NOT NULL,
    current_price REAL,
    last_updated  TEXT,
    status        TEXT     NOT NULL DEFAULT 'open',
    FOREIGN KEY (run_id)   REFERENCES runs(run_id),
    FOREIGN KEY (trade_id) REFERENCES trades(trade_id)
);

-- Key-value state store for miscellaneous runtime state.
CREATE TABLE IF NOT EXISTS state (
    key         TEXT  PRIMARY KEY,
    value       TEXT  NOT NULL,
    updated_at  TEXT  NOT NULL
);
"""
