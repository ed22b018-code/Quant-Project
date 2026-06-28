"""NSE equity trading calendar.

Provides:
  - is_trading_day / is_market_open
  - next_trading_day / prev_trading_day
  - trading_days_in_range
  - classify_missing_file — 'expected' (holiday/weekend) vs 'unexpected' (trading day
    with no bhavcopy file); used by the downloader for gap-classification.

Holiday data is loaded from config/nse_holidays.json when present.  If the file is
missing or malformed the module falls back to an embedded list and logs a WARNING.
Call refresh_nse_holidays() to update the cache from NSE's API.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

_log = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
_MARKET_OPEN = time(9, 15)
_MARKET_CLOSE = time(15, 30)   # exclusive — 15:30:00.000 is NOT inside market hours

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_HOLIDAYS_JSON = _PROJECT_ROOT / "config" / "nse_holidays.json"

# ---------------------------------------------------------------------------
# Embedded fallback — approximate NSE equity holidays 2023-2026.
# Source: NSE India circulars (https://www.nseindia.com).
# Lunar-based holidays (Eid, Muharram, Diwali) are approximate — update the
# JSON override once NSE confirms dates for the relevant year.
# Holidays that fall on weekends are omitted (weekend check already handles them).
# ---------------------------------------------------------------------------
_EMBEDDED: dict[int, list[str]] = {
    2023: [
        "2023-01-26",  # Republic Day
        "2023-03-07",  # Holi
        "2023-03-30",  # Ram Navami
        "2023-04-04",  # Mahavir Jayanti
        "2023-04-07",  # Good Friday
        "2023-04-14",  # Dr. Baba Saheb Ambedkar Jayanti
        "2023-04-22",  # Id-ul-Fitr (Ramzan Eid) — approximate
        "2023-06-29",  # Bakri Eid — approximate
        "2023-07-28",  # Muharram — approximate
        "2023-08-15",  # Independence Day
        "2023-10-02",  # Mahatma Gandhi Jayanti
        "2023-10-24",  # Dussehra
        "2023-11-13",  # Diwali Laxmi Puja
        "2023-11-27",  # Gurunanak Jayanti
        "2023-12-25",  # Christmas
    ],
    2024: [
        "2024-01-26",  # Republic Day (Friday)
        "2024-03-19",  # Holi (Tuesday)
        "2024-03-29",  # Good Friday
        "2024-04-17",  # Ram Navami (Wednesday)
        "2024-06-17",  # Bakri Eid (Monday) — approximate
        "2024-07-17",  # Muharram (Wednesday) — approximate
        "2024-08-15",  # Independence Day (Thursday)
        "2024-10-02",  # Mahatma Gandhi Jayanti (Wednesday)
        "2024-11-01",  # Diwali Laxmi Puja (Friday)
        "2024-11-15",  # Gurunanak Jayanti (Friday)
        "2024-12-25",  # Christmas (Wednesday)
    ],
    2025: [
        "2025-02-26",  # Mahashivratri (Wednesday)
        "2025-03-14",  # Holi (Friday)
        "2025-03-31",  # Eid ul-Fitr (Monday) — approximate
        "2025-04-10",  # Mahavir Jayanti (Thursday)
        "2025-04-14",  # Dr. Baba Saheb Ambedkar Jayanti (Monday)
        "2025-04-18",  # Good Friday
        "2025-05-01",  # Maharashtra Day (Thursday)
        "2025-08-15",  # Independence Day (Friday)
        "2025-10-02",  # Mahatma Gandhi Jayanti (Thursday)
        "2025-10-21",  # Diwali Laxmi Puja (Tuesday) — approximate
        "2025-11-05",  # Gurunanak Jayanti (Wednesday)
        "2025-12-25",  # Christmas (Thursday)
    ],
    2026: [
        "2026-01-26",  # Republic Day (Monday)
        "2026-03-03",  # Holi (Tuesday) — approximate
        "2026-03-20",  # Eid ul-Fitr (Friday) — approximate
        "2026-03-27",  # Good Friday (approximate)
        "2026-04-14",  # Dr. Baba Saheb Ambedkar Jayanti (Tuesday)
        "2026-10-02",  # Mahatma Gandhi Jayanti (Friday)
        "2026-11-09",  # Diwali Laxmi Puja (Monday) — approximate
        "2026-12-25",  # Christmas (Friday)
    ],
}


class NSETradingCalendar:
    """NSE equity segment trading calendar.

    Construction
    ------------
    NSETradingCalendar()                           # load from config/nse_holidays.json
    NSETradingCalendar(holidays_json=some_path)    # load from a specific file
    NSETradingCalendar(holidays={date(2024,1,26)}) # inject directly (for testing)
    """

    def __init__(
        self,
        *,
        holidays: set[date] | frozenset[date] | None = None,
        holidays_json: Path | None = None,
    ) -> None:
        if holidays is not None:
            self._holidays: frozenset[date] = frozenset(holidays)
            self._source = "injected"
            # Injected holiday sets are test-specific; suppress forward-coverage warnings.
            self._max_holiday_year: int = 0
        else:
            path = holidays_json if holidays_json is not None else _DEFAULT_HOLIDAYS_JSON
            self._holidays, self._source = _load_holidays(path)
            self._max_holiday_year = max((d.year for d in self._holidays), default=0)
        self._warned_future_years: set[int] = set()

    # ── Core predicates ──────────────────────────────────────────────────────

    def is_trading_day(self, d: date) -> bool:
        """True iff d is a weekday that is not an NSE equity holiday.

        If the date falls in a year beyond the loaded holiday data range a
        WARNING is logged once per unknown year.  The method still returns a
        best-effort answer (weekday + not in loaded holidays), but the answer
        may be wrong for holidays not yet in the cache.  Call
        refresh_nse_holidays(year) to extend coverage.
        """
        if self._max_holiday_year > 0 and d.year > self._max_holiday_year:
            if d.year not in self._warned_future_years:
                _log.warning(
                    "is_trading_day(%s): date is in year %d which is beyond the loaded "
                    "holiday data range (max year in cache: %d). Results may be incorrect "
                    "for holidays in %d. Call refresh_nse_holidays(%d) to update the cache.",
                    d, d.year, self._max_holiday_year, d.year, d.year,
                )
                self._warned_future_years.add(d.year)
        return d.weekday() < 5 and d not in self._holidays

    def is_market_open(self, dt: datetime) -> bool:
        """True iff dt falls within NSE market hours: 09:15 <= t < 15:30 IST.

        dt must be tz-aware (IST or UTC; either is accepted and converted to IST).
        Raises ValueError for tz-naive datetimes — naive datetimes are ambiguous.
        """
        if dt.tzinfo is None:
            raise ValueError(
                "is_market_open requires a tz-aware datetime. "
                "Provide an IST-aware or UTC-aware datetime; naive datetimes are ambiguous."
            )
        ist = dt.astimezone(IST)
        return self.is_trading_day(ist.date()) and _MARKET_OPEN <= ist.time() < _MARKET_CLOSE

    # ── Navigation ───────────────────────────────────────────────────────────

    def next_trading_day(self, d: date) -> date:
        """First trading day strictly after d."""
        d = d + timedelta(days=1)
        while not self.is_trading_day(d):
            d += timedelta(days=1)
        return d

    def prev_trading_day(self, d: date) -> date:
        """Last trading day strictly before d."""
        d = d - timedelta(days=1)
        while not self.is_trading_day(d):
            d -= timedelta(days=1)
        return d

    def trading_days_in_range(self, start: date, end: date) -> list[date]:
        """All trading days d with start <= d <= end (inclusive on both ends)."""
        out: list[date] = []
        d = start
        while d <= end:
            if self.is_trading_day(d):
                out.append(d)
            d += timedelta(days=1)
        return out

    # ── Gap classification ───────────────────────────────────────────────────

    def classify_missing_file(self, d: date) -> Literal["expected", "unexpected"]:
        """Is a missing bhavcopy file on date d expected or a data gap?

        Returns
        -------
        'expected'   — d is a weekend or NSE holiday; no file is normal.
        'unexpected' — d is a trading day; a file should exist but doesn't.
        """
        return "unexpected" if self.is_trading_day(d) else "expected"


# ── Holiday loading ──────────────────────────────────────────────────────────

def _load_holidays(path: Path) -> tuple[frozenset[date], str]:
    """Load holidays from a JSON file; fall back to embedded list on any failure."""
    if path.exists():
        try:
            raw: dict[str, list[str]] = json.loads(path.read_text(encoding="utf-8"))
            holidays: set[date] = set()
            for year_dates in raw.values():
                for d_str in year_dates:
                    holidays.add(date.fromisoformat(d_str))
            _log.debug("NSE holidays loaded from %s (%d dates)", path, len(holidays))
            return frozenset(holidays), str(path)
        except Exception as exc:
            _log.warning(
                "Failed to parse NSE holidays from %s (%s); falling back to embedded list.",
                path,
                exc,
            )
    else:
        _log.warning(
            "NSE holidays file not found at %s; falling back to embedded list. "
            "Run refresh_nse_holidays() to populate it.",
            path,
        )
    return _embedded_holidays(), "embedded"


def _embedded_holidays() -> frozenset[date]:
    out: set[date] = set()
    for year_dates in _EMBEDDED.values():
        for d_str in year_dates:
            out.add(date.fromisoformat(d_str))
    return frozenset(out)


# ── NSE cache refresh ────────────────────────────────────────────────────────

def refresh_nse_holidays(
    year: int,
    save_path: Path = _DEFAULT_HOLIDAYS_JSON,
) -> bool:
    """Try to fetch NSE equity holidays for `year` from the NSE API and update
    the local cache at `save_path`.

    Returns True when the cache is updated, False when fetch or parse fails.
    On failure a WARNING is logged and the existing cache is left unchanged —
    the function never raises.

    Notes
    -----
    NSE's API requires browser-like headers and may return empty results or
    throttle repeated calls.  If this returns False, update `save_path` manually
    using data from https://www.nseindia.com/products-services/equity-market-timings-holidays.
    """
    import urllib.request  # noqa: PLC0415

    url = "https://www.nseindia.com/api/holiday-master?type=trading"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw: dict = json.loads(resp.read().decode("utf-8"))

        # NSE response format: {"CM": [{"tradingDate": "26-Jan-2024", ...}, ...], ...}
        new_dates: list[str] = []
        for entry in raw.get("CM", []):
            date_str = entry.get("tradingDate", "")
            try:
                d = datetime.strptime(date_str, "%d-%b-%Y").date()
                if d.year == year and d.weekday() < 5:
                    new_dates.append(d.isoformat())
            except ValueError:
                continue

        if not new_dates:
            _log.warning(
                "refresh_nse_holidays: parsed 0 weekday holidays for %d from NSE "
                "response — response format may have changed. Cache unchanged.",
                year,
            )
            return False

        existing: dict[str, list[str]] = {}
        if save_path.exists():
            try:
                existing = json.loads(save_path.read_text(encoding="utf-8"))
            except Exception:
                pass

        existing[str(year)] = sorted(set(new_dates))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(
            json.dumps(existing, indent=2, sort_keys=True), encoding="utf-8"
        )
        _log.info(
            "refresh_nse_holidays: cached %d holidays for %d → %s",
            len(new_dates),
            year,
            save_path,
        )
        return True

    except Exception as exc:
        _log.warning(
            "refresh_nse_holidays: fetch failed (%s). "
            "Cache unchanged — update %s manually or rely on the embedded list.",
            exc,
            save_path,
        )
        return False


# ── Module-level default instance ────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_calendar() -> NSETradingCalendar:
    """Return the module-level NSETradingCalendar (loaded once from the JSON cache)."""
    return NSETradingCalendar()
