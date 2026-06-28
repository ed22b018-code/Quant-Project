"""Corporate-action data fetcher and backward-adjusted price adapter.

The raw-vs-adjusted boundary (explicit definition):
  BhavCopySource        → raw unadjusted prices; safe for live point-in-time features
  AdjustedBhavCopySource → backward-adjusted prices; BACKTEST USE ONLY

BACKTEST_ONLY is a structural discriminator on AdjustedBhavCopySource.  Any code
path that checks this flag will correctly reject the adjusted source for live use.
data_source_name is deliberately distinct so the experiment run log always records
which source was used.

Adjustment method — backward, anchored on the most-recent bar:
  For price bar on date D the cumulative factor = product of all action factors where
  D < ex_date AND ex_date <= as_of.date()  (PIT-correct: a split effective after
  as_of is unknown to this backtest context and must not be applied).

  Consequence: the most-recent bar's OHLC values equal the true unadjusted market
  price; all earlier bars are scaled into the same unit.  Volume is not adjusted.

Action types and factors:
  SPLIT    factor = new_face_value / old_face_value   (< 1.0)
           "SPLIT FROM Rs 10 TO Rs 2"  →  2/10 = 0.20
  BONUS    factor = existing / (existing + bonus)     (< 1.0)
           "BONUS 1:1"                 →  1/(1+1) = 0.50
  DIVIDEND factor = 1.0  (intentionally NOT price-adjusted; logged as known decision —
                    ordinary dividends do not create a price discontinuity)
  RIGHTS   factor = 1.0  (complex; excluded in Phase 1.5; logged)
  OTHER    factor = 1.0  (unrecognized purpose string; WARNING logged)

Conservative guarantee: unrecognized or partially-parseable input always returns
('OTHER', 1.0).  A missed adjustment is recoverable via load_corporate_actions_csv;
a wrong factor silently corrupts all historical prices.

CPU-only — no GPU library imports.
"""
from __future__ import annotations

import json
import logging
import re
import urllib.request
from datetime import date, datetime, timezone
from typing import Union

import pandas as pd

from src.data.base import DataNotAvailableError, DataSource
from src.data.historical import OHLCV_DAILY_COLS, _empty
from src.db.duckdb_client import DuckDBClient
from src.db.schema import DUCKDB_SCHEMA_SQL

_log = logging.getLogger(__name__)

# ── NSE purpose-string regex patterns ────────────────────────────────────────

# Matches: "STOCK SPLIT FROM Rs 10/- TO Rs 2/-"
#          "FACE VALUE SPLIT FROM RS.10/- TO RS.1/-"
#          "FACE VALUE CHANGE FROM RS.10/- TO RS.5/-"
#          "SPLIT FROM 10 TO 5"
_SPLIT_RE = re.compile(
    r"(?:stock\s+split|face\s+value\s+(?:split|change)|split)\s+from\s+"
    r"(?:rs\.?\s*)?(\d+(?:\.\d+)?)(?:\s*/-)?.*?\bto\s+"
    r"(?:rs\.?\s*)?(\d+(?:\.\d+)?)",
    re.IGNORECASE | re.DOTALL,
)

# Matches: "BONUS 1:1", "BONUS 2:5", "BONUS (1:2)"
_BONUS_RE = re.compile(
    r"\bbonus\b\s*\(?(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\)?",
    re.IGNORECASE,
)

# Matches: "INTERIM DIVIDEND", "FINAL DIVIDEND", "SPECIAL DIVIDEND", "DIVIDEND"
_DIVIDEND_RE = re.compile(r"\bdividend\b", re.IGNORECASE)

# Matches: "RIGHTS 1:10", "RIGHT 2:5"
_RIGHTS_RE = re.compile(r"\brights?\b.*?\d+\s*:\s*\d+", re.IGNORECASE)


# ── Free-text parser ─────────────────────────────────────────────────────────

def parse_action_description(purpose: str) -> tuple[str, float]:
    """Parse NSE's free-text corporate-action purpose string.

    Returns (action_type, adjustment_factor):
      ('SPLIT',    factor)  — factor = new_FV / old_FV  (< 1.0)
      ('BONUS',    factor)  — factor = existing / (existing + bonus)  (< 1.0)
      ('DIVIDEND', 1.0)     — explicitly not price-adjusted; logged as intentional
      ('RIGHTS',   1.0)     — not adjusted in Phase 1.5; logged
      ('OTHER',    1.0)     — unrecognized or partially-parseable; WARNING logged

    Conservative guarantee: an ambiguous or unrecognized string NEVER produces a
    wrong factor.  Returning ('OTHER', 1.0) leaves history unchanged; returning a
    wrong factor silently corrupts every historical price for that symbol.
    """
    if not purpose or not purpose.strip():
        _log.warning("parse_action_description: empty purpose string → ('OTHER', 1.0)")
        return ("OTHER", 1.0)

    text = purpose.strip()

    # Dividends — recognise but explicitly do not price-adjust
    if _DIVIDEND_RE.search(text):
        _log.info(
            "parse_action_description: dividend %r — intentionally not price-adjusted "
            "(ordinary dividends do not create a price-continuity break requiring "
            "backward correction; decision recorded in corporate_actions table)",
            text[:80],
        )
        return ("DIVIDEND", 1.0)

    # Rights issues — not adjusted in Phase 1.5
    if _RIGHTS_RE.search(text):
        _log.info(
            "parse_action_description: rights issue %r — not price-adjusted in "
            "Phase 1.5; supply manual correction via load_corporate_actions_csv "
            "if required",
            text[:80],
        )
        return ("RIGHTS", 1.0)

    # Split — extract old_FV and new_FV; compute factor = new_FV / old_FV
    split_match = _SPLIT_RE.search(text)
    if split_match:
        try:
            old_fv = float(split_match.group(1))
            new_fv = float(split_match.group(2))
            if old_fv <= 0 or new_fv <= 0:
                raise ValueError(
                    f"Non-positive face value: old_FV={old_fv}, new_FV={new_fv}"
                )
            if new_fv >= old_fv:
                raise ValueError(
                    f"new_FV ({new_fv}) >= old_FV ({old_fv}): looks like a reverse "
                    f"split or unrecognized format"
                )
            factor = round(new_fv / old_fv, 10)
            _log.info(
                "parse_action_description: split %r → factor=%.6f (FV %g → %g)",
                text[:80], factor, old_fv, new_fv,
            )
            return ("SPLIT", factor)
        except Exception as exc:
            _log.warning(
                "parse_action_description: split pattern matched but ratio parse "
                "failed for %r: %s — falling back to ('OTHER', 1.0)",
                text[:80], exc,
            )
            return ("OTHER", 1.0)

    # Bonus — extract N:M ratio (N bonus shares for every M existing shares)
    #   factor = M / (M + N)
    bonus_match = _BONUS_RE.search(text)
    if bonus_match:
        try:
            bonus_n = float(bonus_match.group(1))    # N (bonus shares)
            existing_m = float(bonus_match.group(2)) # M (existing shares)
            if bonus_n <= 0 or existing_m <= 0:
                raise ValueError(
                    f"Non-positive bonus ratio: {bonus_n}:{existing_m}"
                )
            factor = round(existing_m / (existing_m + bonus_n), 10)
            _log.info(
                "parse_action_description: bonus %r → factor=%.6f (%g:%g)",
                text[:80], factor, bonus_n, existing_m,
            )
            return ("BONUS", factor)
        except Exception as exc:
            _log.warning(
                "parse_action_description: bonus pattern matched but ratio parse "
                "failed for %r: %s — falling back to ('OTHER', 1.0)",
                text[:80], exc,
            )
            return ("OTHER", 1.0)

    _log.warning(
        "parse_action_description: unrecognized purpose %r → ('OTHER', 1.0). "
        "No price adjustment applied.  Supply a manual correction via "
        "load_corporate_actions_csv if this event changes the price series.",
        text[:80],
    )
    return ("OTHER", 1.0)


# ── NSE API fetcher ───────────────────────────────────────────────────────────

_NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/companies-listing/corporate-filings-actions",
}


class CorporateActionFetcher:
    """Fetch NSE corporate actions and upsert into DuckDB.

    The NSE API requires session cookies and changes layout periodically; treat it
    as best-effort.  For reliable production ingestion use load_corporate_actions_csv
    with a manually prepared or externally sourced CSV.

    Parameters
    ----------
    _mock_response:
        Test hook — list of action-record dicts matching the NSE API response
        shape.  When provided, bypasses the real HTTP call.
    """

    _NSE_CA_URL = (
        "https://www.nseindia.com/api/corporates-corporateActions"
        "?index=equities&from_date={from_date}&to_date={to_date}"
    )

    def __init__(self, _mock_response=None) -> None:
        self._mock_response = _mock_response

    def fetch_and_store(
        self,
        symbol: str,
        from_date: date,
        to_date: date,
        db_path,
    ) -> int:
        """Fetch corporate actions for *symbol* and upsert into DuckDB.

        Returns the number of new rows inserted (0 on any API failure).
        """
        if self._mock_response is not None:
            records = list(self._mock_response)
        else:
            try:
                records = self._call_nse_api(from_date, to_date)
            except Exception as exc:
                _log.warning(
                    "CorporateActionFetcher: NSE API call failed: %s. "
                    "Use load_corporate_actions_csv as the reliable fallback.",
                    exc,
                )
                return 0

        sym_upper = symbol.strip().upper()
        sym_records = [
            r for r in records
            if r.get("symbol", "").strip().upper() == sym_upper
        ]
        if not sym_records:
            return 0

        now_utc = datetime.now(timezone.utc)
        rows = []
        for r in sym_records:
            raw_date = (
                r.get("exDate") or r.get("ex_date") or r.get("date") or ""
            )
            raw_purpose = (
                r.get("subject") or r.get("purpose") or r.get("description") or ""
            )
            try:
                ex_date_val = pd.to_datetime(raw_date, dayfirst=True).date()
            except Exception:
                _log.warning(
                    "CorporateActionFetcher: cannot parse ex_date %r for %r — skipping",
                    raw_date, symbol,
                )
                continue

            action_type, factor = parse_action_description(str(raw_purpose))
            rows.append({
                "symbol":            sym_upper,
                "ex_date":           ex_date_val,
                "action_type":       action_type,
                "adjustment_factor": factor,
                "description":       str(raw_purpose)[:500] if raw_purpose else None,
                "source":            "nse_corporate_actions",
                "ingested_at":       now_utc,
            })

        if not rows:
            return 0

        return _upsert_actions(pd.DataFrame(rows), db_path)

    def _call_nse_api(self, from_date: date, to_date: date) -> list[dict]:
        """Make the NSE corporate-actions API call (best-effort; may raise)."""
        import http.cookiejar

        url = self._NSE_CA_URL.format(
            from_date=from_date.strftime("%d-%m-%Y"),
            to_date=to_date.strftime("%d-%m-%Y"),
        )

        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(cj)
        )
        for key, val in _NSE_HEADERS.items():
            opener.addheaders.append((key, val))

        opener.open("https://www.nseindia.com/", timeout=10)
        resp = opener.open(url, timeout=15)
        data = json.loads(resp.read())

        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("data", [])
        return []


# ── CSV fallback ─────────────────────────────────────────────────────────────

def load_corporate_actions_csv(path: str, db_path) -> int:
    """Load corporate actions from a manually prepared CSV file.

    Expected columns (case-sensitive):
      symbol             NSE ticker, e.g. "HDFC"
      ex_date            ISO date YYYY-MM-DD
      action_type        SPLIT | BONUS | DIVIDEND | RIGHTS | OTHER
      adjustment_factor  float (1.0 for DIVIDEND / RIGHTS / OTHER)
      description        optional raw purpose string (column may be absent)

    Returns the number of new rows inserted.  Duplicate rows (same primary key)
    are silently skipped via ON CONFLICT DO NOTHING.
    """
    required = {"symbol", "ex_date", "action_type", "adjustment_factor"}
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"load_corporate_actions_csv: CSV is missing required columns: {missing!r}"
        )

    df = df.copy()
    df["ex_date"] = pd.to_datetime(df["ex_date"], format="%Y-%m-%d").dt.date
    df["adjustment_factor"] = df["adjustment_factor"].astype(float)
    df["symbol"] = df["symbol"].str.strip().str.upper()
    df["action_type"] = df["action_type"].str.strip().str.upper()
    if "description" not in df.columns:
        df["description"] = None
    df["source"] = "csv_manual"
    df["ingested_at"] = datetime.now(timezone.utc)

    return _upsert_actions(df, db_path)


def _upsert_actions(df: pd.DataFrame, db_path) -> int:
    """Upsert corporate-action rows into DuckDB; return count of new rows."""
    with DuckDBClient(db_path) as db:
        db.execute_schema(DUCKDB_SCHEMA_SQL)
        db.conn.register("_ca_staging", df)
        before = db.conn.execute(
            "SELECT COUNT(*) FROM corporate_actions"
        ).fetchone()[0]
        db.conn.execute("""
            INSERT INTO corporate_actions
                (symbol, ex_date, action_type, adjustment_factor,
                 description, source, ingested_at)
            SELECT symbol, ex_date, action_type, adjustment_factor,
                   description, source, ingested_at
            FROM _ca_staging
            ON CONFLICT DO NOTHING
        """)
        after = db.conn.execute(
            "SELECT COUNT(*) FROM corporate_actions"
        ).fetchone()[0]
        db.conn.unregister("_ca_staging")
    return after - before


# ── Adjustment-factor computation ─────────────────────────────────────────────

def compute_adjustment_factors(
    actions: pd.DataFrame,
    price_dates: pd.Series,
    as_of: Union[datetime, date],
) -> pd.Series:
    """Compute the per-bar backward adjustment factor for each price date.

    For a price bar on date D the factor is the cumulative product of all action
    factors where  D < ex_date  AND  ex_date <= as_of.date()  (PIT-correct).

    A split/bonus effective AFTER as_of is excluded — it is unknown to this
    backtest context and must not retroactively change historical prices.

    Parameters
    ----------
    actions:
        DataFrame with columns 'ex_date' (date) and 'adjustment_factor' (float).
        Typically queried from the corporate_actions DuckDB table.
    price_dates:
        Series of datetime.date values (e.g., df["date"] from ohlcv_daily after
        normalisation).  The returned Series shares the same index.
    as_of:
        Backtest cutoff.  Actions with ex_date > as_of.date() are excluded.

    Returns
    -------
    pd.Series[float] with the same index as price_dates.
    Factor = 1.0 where no adjustable action applies (on/after ex_date, or
    no actions present).
    """
    if isinstance(as_of, datetime):
        as_of_date = as_of.date()
    else:
        as_of_date = as_of

    if actions.empty:
        return pd.Series(1.0, index=price_dates.index, dtype="float64")

    # PIT filter: only actions effective on or before as_of
    ex_dates_norm = pd.to_datetime(actions["ex_date"]).dt.date
    valid = actions[ex_dates_norm <= as_of_date].copy()

    if valid.empty:
        return pd.Series(1.0, index=price_dates.index, dtype="float64")

    valid = valid.assign(_ex_norm=pd.to_datetime(valid["ex_date"]).dt.date)

    def _factor_for(bar_date: date) -> float:
        # Product of factors for all actions where action.ex_date > bar_date
        mask = valid["_ex_norm"] > bar_date
        if not mask.any():
            return 1.0
        return float(valid.loc[mask, "adjustment_factor"].prod())

    return pd.Series(
        [_factor_for(d) for d in price_dates],
        index=price_dates.index,
        dtype="float64",
    )


# ── AdjustedBhavCopySource ────────────────────────────────────────────────────

class AdjustedBhavCopySource(DataSource):
    """Backward-adjusted daily OHLCV — BACKTEST USE ONLY.

    OHLC prices are multiplied by the cumulative corporate-action adjustment
    factor so the time series is continuous across split and bonus ex-dates.
    The most-recent bar retains the true unadjusted market price (the backward
    adjustment anchors on the present).

    BACKTEST_ONLY = True is a structural discriminator.  Live feature code that
    checks this flag will correctly refuse the adjusted source.  The distinct
    data_source_name ensures the experiment run log always records which source
    was used.

    Requires the corporate_actions table to be populated (via
    CorporateActionFetcher.fetch_and_store or load_corporate_actions_csv) before
    adjusted prices are meaningful; an unpopulated table yields raw prices.
    """

    BACKTEST_ONLY: bool = True   # structural discriminator — not a docstring convention

    data_source_name = "nse_bhavcopy_adjusted"
    data_source_limitations = (
        "BACKTEST USE ONLY — do NOT use for live point-in-time features. "
        "Prices are backward-adjusted for splits and bonuses "
        "(factor = new_FV/old_FV for splits; existing/(existing+bonus) for bonuses). "
        "Dividends are intentionally not price-adjusted (logged as known decision). "
        "Adjustment for date D uses only actions with ex_date <= as_of.date() "
        "(PIT-correct; future splits are excluded). "
        "The most-recent bar retains the true unadjusted market price (backward anchor). "
        "Requires corporate_actions table populated before use."
    )
    data_granularity = "daily"

    def __init__(self, db_path) -> None:
        self._db_path = db_path

    # ── Daily OHLCV ─────────────────────────────────────────────────────────

    def get_ohlcv_daily(
        self,
        symbol: str,
        as_of: datetime,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """Backward-adjusted daily OHLCV with date < as_of.date().

        Returns the same column schema as BhavCopySource.get_ohlcv_daily but
        with 'source' set to 'nse_bhavcopy_adjusted'.  Volume is not adjusted.
        """
        if isinstance(as_of, datetime):
            cutoff = as_of.date()
        else:
            cutoff = as_of

        sql_raw = """
            SELECT date, open, high, low, close, volume, traded_value, source
            FROM ohlcv_daily
            WHERE symbol = ?
              AND date < ?
            ORDER BY date DESC
            LIMIT ?
        """
        sql_ca = """
            SELECT ex_date, adjustment_factor
            FROM corporate_actions
            WHERE symbol = ?
              AND action_type IN ('SPLIT', 'BONUS')
            ORDER BY ex_date ASC
        """

        with DuckDBClient(self._db_path) as db:
            db.execute_schema(DUCKDB_SCHEMA_SQL)
            df_raw = db.query_df(sql_raw, (symbol, cutoff, lookback_days))
            df_ca = db.query_df(sql_ca, (symbol,))

        if df_raw.empty:
            return self._enforce_pit(_empty(OHLCV_DAILY_COLS), "date", as_of)

        df = df_raw.copy()
        df["date"] = pd.to_datetime(df["date"]).dt.date

        if not df_ca.empty:
            factors = compute_adjustment_factors(df_ca, df["date"], as_of)
            for col in ("open", "high", "low", "close"):
                df[col] = (df[col].values * factors.values).round(4)

        df["source"] = "nse_bhavcopy_adjusted"

        return self._enforce_pit(df, "date", as_of)

    # ── All other data types — not available ─────────────────────────────────

    def get_ohlcv_minute(self, symbol, as_of, lookback_bars=390):
        raise DataNotAvailableError(
            f"AdjustedBhavCopySource provides daily bars only for {symbol!r}. "
            "Intraday adjusted data is not available in Phase 1.5."
        )

    def get_ticks(self, symbol, as_of, lookback_seconds=300):
        raise DataNotAvailableError(
            f"AdjustedBhavCopySource does not provide tick data for {symbol!r}."
        )

    def get_news(self, symbol, as_of, lookback_hours=24.0):
        raise DataNotAvailableError(
            f"AdjustedBhavCopySource does not provide news for {symbol!r}."
        )

    def get_option_chain(self, symbol, as_of):
        raise DataNotAvailableError(
            f"AdjustedBhavCopySource does not provide option chain data for {symbol!r}."
        )
