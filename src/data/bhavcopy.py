"""NSE bhavcopy downloader, parser, validator, and DuckDB loader.

Public surface
--------------
MissingDataGap          — raised when a bhavcopy file is absent on a trading day
parse_bhavcopy_csv      — normalise raw NSE CSV content to our schema DataFrame
validate_bhavcopy       — row-level sanity checks; hard-rejects impossible rows,
                          soft-flags suspicious rows; logs every decision
download_bhavcopy       — fetch ZIP from NSE archives, extract CSV, save to disk
load_bhavcopy_to_duckdb — parse + validate + upsert into ohlcv_daily; idempotent

NSE bhavcopy URL (newer format, 2022+):
  https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_YYYYMMDD_F_0000.csv.zip

Column mapping (new → schema, old → schema)
  New: TckrSymb→symbol, TradDt→date, SctySrs→series, OpnPric→open,
       HghPric→high, LwPric→low, ClsPric→close,
       TtlTradgVol→volume, TtlTrfVal→traded_value
  Old: SYMBOL→symbol, TIMESTAMP→date, SERIES→series, OPEN→open,
       HIGH→high, LOW→low, CLOSE→close, TOTTRDQTY→volume, TOTTRDVAL→traded_value
"""
from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from src.db.duckdb_client import DuckDBClient
from src.db.schema import DUCKDB_SCHEMA_SQL

_log = logging.getLogger(__name__)

_BHAVCOPY_URL = (
    "https://nsearchives.nseindia.com/content/cm/"
    "BhavCopy_NSE_CM_0_0_0_{date}_F_0000.csv.zip"
)
_MAX_RETRIES = 3
_EXTREME_MOVE_THRESHOLD = 0.25  # 25 % single-day high-low range / close

# ── Column name mappings ─────────────────────────────────────────────────────

_NEW_TO_SCHEMA = {
    "TckrSymb":    "symbol",
    "TradDt":      "date",
    "SctySrs":     "series",
    "OpnPric":     "open",
    "HghPric":     "high",
    "LwPric":      "low",
    "ClsPric":     "close",
    "TtlTradgVol": "volume",
    "TtlTrfVal":   "traded_value",
}

_OLD_TO_SCHEMA = {
    "SYMBOL":    "symbol",
    "TIMESTAMP": "date",
    "SERIES":    "series",
    "OPEN":      "open",
    "HIGH":      "high",
    "LOW":       "low",
    "CLOSE":     "close",
    "TOTTRDQTY": "volume",
    "TOTTRDVAL": "traded_value",
}


class MissingDataGap(RuntimeError):
    """Raised when a bhavcopy file is absent on a trading day.

    Use NSETradingCalendar.classify_missing_file() before calling
    download_bhavcopy() to distinguish expected holiday gaps from real gaps.
    """


# ── Parsing ──────────────────────────────────────────────────────────────────

_REQUIRED_NEW = frozenset({
    "TckrSymb", "TradDt", "SctySrs", "OpnPric", "HghPric", "LwPric", "ClsPric", "TtlTradgVol",
})
_REQUIRED_OLD = frozenset({
    "SYMBOL", "SERIES", "TIMESTAMP", "OPEN", "HIGH", "LOW", "CLOSE", "TOTTRDQTY",
})


def _detect_format(columns: list[str]) -> dict[str, str]:
    col_set = set(columns)
    if "TckrSymb" in col_set:
        missing = _REQUIRED_NEW - col_set
        if missing:
            raise ValueError(
                f"Bhavcopy CSV is newer NSE format (TckrSymb present) but is missing "
                f"required columns: {sorted(missing)}. Got: {columns[:15]}"
            )
        return _NEW_TO_SCHEMA
    if "SYMBOL" in col_set:
        missing = _REQUIRED_OLD - col_set
        if missing:
            raise ValueError(
                f"Bhavcopy CSV is older NSE format (SYMBOL present) but is missing "
                f"required columns: {sorted(missing)}. Got: {columns[:15]}"
            )
        return _OLD_TO_SCHEMA
    raise ValueError(
        f"Unrecognised bhavcopy column layout. Got: {columns[:10]}. "
        "Expected 'TckrSymb' (newer format) or 'SYMBOL' (older format)."
    )


def _parse_date_col(series: pd.Series) -> pd.Series:
    """Try multiple NSE date formats; raise on unrecognised input."""
    sample = series.dropna().iloc[0] if not series.dropna().empty else ""
    sample = str(sample).strip()
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y"):
        try:
            datetime.strptime(sample, fmt)
            return pd.to_datetime(series, format=fmt).dt.date
        except (ValueError, TypeError):
            continue
    # Last resort — let pandas infer (handles mixed formats)
    _log.warning("Could not detect bhavcopy date format from %r; letting pandas infer", sample)
    return pd.to_datetime(series, infer_datetime_format=True).dt.date


def parse_bhavcopy_csv(content: str | bytes) -> pd.DataFrame:
    """Parse raw NSE bhavcopy CSV content into a normalised DataFrame.

    Filters to EQ series only.  Returned columns match ohlcv_daily schema:
    symbol (str), date (datetime.date), series (str),
    open/high/low/close/traded_value (float64), volume (int64).
    """
    if isinstance(content, bytes):
        content = content.decode("utf-8", errors="replace")

    raw = pd.read_csv(io.StringIO(content))
    raw.columns = [c.strip() for c in raw.columns]

    mapping = _detect_format(list(raw.columns))
    present = {k: v for k, v in mapping.items() if k in raw.columns}
    df = raw[list(present.keys())].rename(columns=present).copy()

    # Filter to equity series
    if "series" in df.columns:
        df = df[df["series"].str.strip() == "EQ"].copy()
    df = df.reset_index(drop=True)

    if df.empty:
        _log.warning("parse_bhavcopy_csv: no EQ rows found in input")
        return df

    # Date column → Python datetime.date objects
    df["date"] = _parse_date_col(df["date"].astype(str))

    # Numeric casts
    for col in ("open", "high", "low", "close", "traded_value"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("int64")

    df["symbol"] = df["symbol"].str.strip()
    if "series" in df.columns:
        df["series"] = df["series"].str.strip()

    return df


# ── Validation ───────────────────────────────────────────────────────────────

def validate_bhavcopy(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate bhavcopy rows.  Returns (valid_df, rejection_log).

    Hard rejects (row removed):
      • Any price column (open/high/low/close) <= 0
      • high < low

    Soft flags (row kept, warning logged):
      • volume == 0 on a trading day
      • (high - low) / close > EXTREME_MOVE_THRESHOLD
    """
    if df.empty:
        return df, []

    rejections: list[str] = []
    mask_ok = pd.Series(True, index=df.index)

    # Hard reject: non-positive, NaN, or non-finite prices
    for col in ("open", "high", "low", "close"):
        if col not in df.columns:
            continue
        # ~(gt(0) & lt(inf)) catches: zero, negative, NaN, +inf, -inf
        bad = ~(df[col].gt(0) & df[col].lt(float("inf")))
        if bad.any():
            for sym in df.loc[bad, "symbol"]:
                msg = f"{sym}: {col} invalid (zero, negative, NaN, or non-finite) — rejected"
                rejections.append(msg)
                _log.warning("Bhavcopy row rejected: %s", msg)
            mask_ok &= ~bad

    # Hard reject: high < low (impossible bar)
    if "high" in df.columns and "low" in df.columns:
        inv = mask_ok & (df["high"] < df["low"])
        if inv.any():
            for sym in df.loc[inv, "symbol"]:
                msg = f"{sym}: high < low — rejected"
                rejections.append(msg)
                _log.warning("Bhavcopy row rejected: %s", msg)
            mask_ok &= ~inv

    valid = df[mask_ok].copy()

    # Soft flag: zero volume
    if "volume" in valid.columns:
        zero_vol = valid["volume"] == 0
        if zero_vol.any():
            for sym in valid.loc[zero_vol, "symbol"]:
                _log.warning("Bhavcopy row flagged (zero volume): %s — kept", sym)

    # Soft flag: extreme single-day move
    if {"high", "low", "close"}.issubset(valid.columns):
        safe_close = valid["close"].where(valid["close"] > 0, other=1.0)
        move = (valid["high"] - valid["low"]) / safe_close
        extreme = move > _EXTREME_MOVE_THRESHOLD
        if extreme.any():
            for sym in valid.loc[extreme, "symbol"]:
                _log.warning(
                    "Bhavcopy row flagged (extreme move >%.0f%%): %s — kept",
                    _EXTREME_MOVE_THRESHOLD * 100,
                    sym,
                )

    return valid, rejections


# ── Download ─────────────────────────────────────────────────────────────────

def download_bhavcopy(
    d: date,
    raw_dir: Path,
    *,
    calendar=None,
) -> Optional[Path]:
    """Download NSE bhavcopy CSV for trading date `d` to `raw_dir/YYYY-MM-DD.csv`.

    Returns the saved Path, or None if `d` is a holiday / weekend (expected gap).
    Raises MissingDataGap if `d` is a trading day and the download fails after
    all retries.

    Parameters
    ----------
    d        : The trading date to fetch.
    raw_dir  : Directory for raw CSV files (created if absent).
    calendar : NSETradingCalendar instance (loaded from default if None).
    """
    import urllib.request  # noqa: PLC0415

    if calendar is None:
        from src.data.calendar import get_calendar  # noqa: PLC0415
        calendar = get_calendar()

    classification = calendar.classify_missing_file(d)
    if classification == "expected":
        _log.info("Bhavcopy for %s skipped — not a trading day (holiday/weekend)", d)
        return None

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / f"{d.isoformat()}.csv"

    if out_path.exists():
        _log.debug("Bhavcopy for %s already on disk at %s", d, out_path)
        return out_path

    url = _BHAVCOPY_URL.format(date=d.strftime("%Y%m%d"))
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Referer": "https://www.nseindia.com/",
    }

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=15) as resp:
                zip_bytes = resp.read()

            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    raise ValueError(f"No CSV inside ZIP for {d}")
                csv_content = zf.read(csv_names[0]).decode("utf-8", errors="replace")

            out_path.write_text(csv_content, encoding="utf-8")
            _log.info("Bhavcopy downloaded for %s → %s", d, out_path)
            return out_path

        except Exception as exc:
            last_exc = exc
            if attempt < _MAX_RETRIES - 1:
                wait = 2 ** attempt
                _log.warning(
                    "Bhavcopy download attempt %d/%d failed for %s: %s. Retrying in %ds.",
                    attempt + 1, _MAX_RETRIES, d, exc, wait,
                )
                time.sleep(wait)

    raise MissingDataGap(
        f"Bhavcopy unavailable for trading day {d} after {_MAX_RETRIES} attempts. "
        f"Last error: {last_exc}"
    ) from last_exc


# ── DuckDB loader ─────────────────────────────────────────────────────────────

def load_bhavcopy_to_duckdb(
    d: date,
    raw_dir: Path,
    db_path: Path,
) -> tuple[int, int]:
    """Parse, validate, and upsert bhavcopy for `d` into DuckDB ohlcv_daily.

    Idempotent: ON CONFLICT (symbol, date) DO NOTHING — loading the same date
    twice is safe.

    Returns
    -------
    (rows_loaded, rows_rejected)
    """
    csv_path = raw_dir / f"{d.isoformat()}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No bhavcopy CSV at {csv_path}. Call download_bhavcopy() first."
        )

    content = csv_path.read_text(encoding="utf-8")
    raw_df = parse_bhavcopy_csv(content)
    valid_df, rejected = validate_bhavcopy(raw_df)

    if valid_df.empty:
        _log.warning("load_bhavcopy_to_duckdb: nothing to load for %s", d)
        return 0, len(rejected)

    now_utc = datetime.now(timezone.utc)

    # Build insert DataFrame with exactly the columns we're inserting
    insert_df = pd.DataFrame(
        {
            "symbol":       valid_df["symbol"].values,
            "date":         valid_df["date"].values,        # datetime.date objects
            "open":         valid_df["open"].values,
            "high":         valid_df["high"].values,
            "low":          valid_df["low"].values,
            "close":        valid_df["close"].values,
            "volume":       valid_df["volume"].values,
            "traded_value": valid_df.get("traded_value", pd.Series([None] * len(valid_df))).values,
            "source":       "nse_bhavcopy",
            "ingested_at":  now_utc,
        }
    )

    with DuckDBClient(db_path) as db:
        db.execute_schema(DUCKDB_SCHEMA_SQL)
        db.conn.register("_bhavcopy_insert", insert_df)
        try:
            db.conn.execute("""
                INSERT INTO ohlcv_daily
                    (symbol, date, open, high, low, close,
                     volume, traded_value, source, ingested_at)
                SELECT
                    symbol,
                    CAST(date AS DATE),
                    CAST(open AS DOUBLE),
                    CAST(high AS DOUBLE),
                    CAST(low AS DOUBLE),
                    CAST(close AS DOUBLE),
                    CAST(volume AS BIGINT),
                    CAST(traded_value AS DOUBLE),
                    source,
                    CAST(ingested_at AS TIMESTAMPTZ)
                FROM _bhavcopy_insert
                ON CONFLICT (symbol, date) DO NOTHING
            """)
        finally:
            db.conn.unregister("_bhavcopy_insert")

    loaded = len(valid_df)
    _log.info(
        "load_bhavcopy_to_duckdb: %d rows loaded, %d rejected for %s",
        loaded, len(rejected), d,
    )
    return loaded, len(rejected)
