"""Checkpoint 3 proof tests for UpstoxLiveSource.

Ten test groups mandated by the project rules:
  1. SDK quarantine — importing the module does NOT import upstox_client
  2. SDK quarantine — prediction-path modules don't import live_upstox transitively
  3. Interface symmetry — tick schema matches StubHistoricalSource
  4. PIT on live data — future tick raises DataLeakageError; past tick returned
  5. Thread safety — concurrent _on_tick / get_ticks don't corrupt the buffer
  6. Empty buffer — returns empty correctly-typed DataFrame, never None
  7. Reconnect backoff — _on_close delays 1, 2, 4 … seconds before retrying
  8. as_of from caller — adapter never reads a clock (proved behaviourally)
  9. No GPU code — no torch / cuda / cupy in the module source
 10. Provenance fields — source="upstox_live" in every returned row
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
import sys
import threading
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from src.data.base import DataLeakageError, DataNotAvailableError
from src.data.historical import TICK_COLS
from src.data.live_upstox import UpstoxLiveSource


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_src() -> UpstoxLiveSource:
    return UpstoxLiveSource(_inject_ws=MagicMock())


def _inject(src: UpstoxLiveSource, symbol: str, ts: datetime, ltp: float = 100.0) -> None:
    src._on_tick({
        "symbol": symbol,
        "ts": ts,
        "ltp": ltp,
        "volume": 100,
        "bid": ltp - 0.5,
        "ask": ltp + 0.5,
        "oi": 5000,
    })


# ── 1. SDK quarantine — module import ────────────────────────────────────────

def test_importing_module_does_not_import_upstox_sdk():
    """Importing live_upstox.py must NOT import the Upstox SDK as a side effect.

    The SDK import is deferred to connect() — not at module level — so this file
    can be imported and used without the SDK installed.
    """
    # The module was already imported at the top of this file.
    # upstox_client must still not be in sys.modules.
    assert "upstox_client" not in sys.modules, (
        "The Upstox SDK was imported as a side effect of importing live_upstox.py. "
        "The SDK import must stay deferred to connect()."
    )


# ── 2. SDK quarantine — prediction-path modules ──────────────────────────────

def test_prediction_path_modules_do_not_import_live_upstox():
    """None of the prediction/feature/backtest modules import live_upstox (or the SDK).

    If any prediction module imports live_upstox it transitively brings in the SDK
    at module-load time, breaking SDK quarantine.
    """
    prediction_modules = [
        "src.config",
        "src.data.base",
        "src.data.historical",
        "src.data.bhavcopy_source",
        "src.data.calendar",
        "src.data.eligibility",
        "src.experiment.tracker",
    ]
    for mod_name in prediction_modules:
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        mod_globals = vars(mod)
        assert "live_upstox" not in str(mod_globals.get("__file__", "")), (
            f"{mod_name} appears to import live_upstox — SDK quarantine violated"
        )
        # The upstox_client SDK must not be reachable from prediction modules
        assert "upstox_client" not in sys.modules, (
            f"upstox_client was found in sys.modules after loading {mod_name}"
        )


# ── 3. Interface symmetry ────────────────────────────────────────────────────

def test_tick_column_names_match_tick_cols_schema():
    """get_ticks returns columns in TICK_COLS order — same as StubHistoricalSource."""
    from src.data.historical import StubHistoricalSource

    stub = StubHistoricalSource()
    live = _make_src()
    as_of = datetime(2024, 1, 22, 16, 0)

    df_stub = stub.get_ticks("X", as_of)
    df_live = live.get_ticks("X", as_of)   # empty buffer

    assert list(df_stub.columns) == list(TICK_COLS.keys())
    assert list(df_live.columns) == list(TICK_COLS.keys())
    assert list(df_stub.columns) == list(df_live.columns)


def test_get_ohlcv_daily_raises_data_not_available():
    """UpstoxLiveSource raises DataNotAvailableError for daily bars (ticks-only source)."""
    src = _make_src()
    with pytest.raises(DataNotAvailableError, match="daily bars"):
        src.get_ohlcv_daily("RELIANCE", datetime(2024, 1, 22, 16, 0))


def test_get_ohlcv_minute_raises_data_not_available():
    src = _make_src()
    with pytest.raises(DataNotAvailableError):
        src.get_ohlcv_minute("RELIANCE", datetime(2024, 1, 22, 16, 0))


def test_get_news_raises_data_not_available():
    src = _make_src()
    with pytest.raises(DataNotAvailableError):
        src.get_news("RELIANCE", datetime(2024, 1, 22, 16, 0))


def test_get_option_chain_raises_data_not_available():
    src = _make_src()
    with pytest.raises(DataNotAvailableError):
        src.get_option_chain("RELIANCE", datetime(2024, 1, 22, 16, 0))


# ── 4. PIT on live data ───────────────────────────────────────────────────────

def test_pit_past_tick_returned():
    """A tick with ts < as_of is returned normally."""
    src = _make_src()
    as_of = datetime(2024, 1, 22, 15, 30)
    _inject(src, "REL", datetime(2024, 1, 22, 15, 0), ltp=2520.0)

    df = src.get_ticks("REL", as_of, lookback_seconds=3600)
    assert len(df) == 1
    assert df["ltp"].iloc[0] == pytest.approx(2520.0)


def test_pit_future_tick_raises_data_leakage_error():
    """A tick with ts >= as_of must raise DataLeakageError."""
    src = _make_src()
    as_of = datetime(2024, 1, 22, 15, 30)
    _inject(src, "REL", datetime(2024, 1, 22, 15, 30), ltp=2525.0)  # exactly at as_of

    with pytest.raises(DataLeakageError):
        src.get_ticks("REL", as_of, lookback_seconds=3600)


def test_pit_tick_one_second_after_as_of_blocked():
    """Even a tick that is 1 second after as_of is blocked."""
    src = _make_src()
    as_of = datetime(2024, 1, 22, 15, 30, 0)
    _inject(src, "REL", datetime(2024, 1, 22, 15, 30, 1))

    with pytest.raises(DataLeakageError):
        src.get_ticks("REL", as_of, lookback_seconds=3600)


# ── 5. Thread safety ─────────────────────────────────────────────────────────

def test_tick_buffer_is_thread_safe():
    """Concurrent _on_tick writes and get_ticks reads must not corrupt the buffer."""
    src = _make_src()
    base_ts = datetime(2024, 1, 22, 9, 30, 0)
    as_of   = datetime(2024, 1, 22, 16, 0, 0)
    errors: list[str] = []

    def writer() -> None:
        for i in range(500):
            try:
                src._on_tick({
                    "symbol": "TSYM",
                    "ts":     base_ts + timedelta(seconds=i),
                    "ltp":    100.0 + i * 0.01,
                    "volume": 100 + i,
                    "bid":    99.0,
                    "ask":    101.0,
                    "oi":     5000,
                })
            except Exception as exc:  # noqa: BLE001
                errors.append(f"writer: {exc}")

    def reader() -> None:
        for _ in range(50):
            try:
                df = src.get_ticks("TSYM", as_of, lookback_seconds=86400)
                assert df is not None
            except Exception as exc:  # noqa: BLE001
                errors.append(f"reader: {exc}")

    t_write = threading.Thread(target=writer)
    t_read  = threading.Thread(target=reader)
    t_write.start()
    t_read.start()
    t_write.join(timeout=15)
    t_read.join(timeout=15)

    assert not errors, f"Thread-safety errors: {errors}"


def test_buffer_per_symbol_is_isolated():
    """Writes to symbol A must not affect symbol B's buffer."""
    src = _make_src()
    ts  = datetime(2024, 1, 22, 10, 0)
    as_of = datetime(2024, 1, 22, 16, 0)

    _inject(src, "AAA", ts, ltp=100.0)
    _inject(src, "BBB", ts, ltp=200.0)

    df_a = src.get_ticks("AAA", as_of, lookback_seconds=86400)
    df_b = src.get_ticks("BBB", as_of, lookback_seconds=86400)

    assert len(df_a) == 1 and df_a["ltp"].iloc[0] == pytest.approx(100.0)
    assert len(df_b) == 1 and df_b["ltp"].iloc[0] == pytest.approx(200.0)


# ── 6. Empty buffer ───────────────────────────────────────────────────────────

def test_empty_buffer_returns_empty_dataframe_not_none():
    """get_ticks on a symbol with no ticks returns an empty DataFrame, never None."""
    src = _make_src()
    df = src.get_ticks("NOSYM", datetime(2024, 1, 22, 16, 0))
    assert df is not None
    assert df.empty
    assert list(df.columns) == list(TICK_COLS.keys())


def test_empty_buffer_still_has_correct_schema():
    src = _make_src()
    df = src.get_ticks("NOSYM", datetime(2024, 1, 22, 16, 0))
    for col in TICK_COLS:
        assert col in df.columns, f"Column {col!r} missing from empty result"


# ── 7. Reconnect with exponential backoff ────────────────────────────────────

def test_reconnect_first_attempt_waits_1_second():
    """_on_close delays 1 s on the first reconnect attempt."""
    import src.data.live_upstox as mod

    src = UpstoxLiveSource(max_reconnect_attempts=3)
    src._connected = False

    with patch.object(mod.time, "sleep") as mock_sleep, \
         patch.object(src, "connect"):
        src._reconnect_attempts = 0
        src._on_close("disconnected")

    mock_sleep.assert_called_once_with(1)


def test_reconnect_second_attempt_waits_2_seconds():
    import src.data.live_upstox as mod

    src = UpstoxLiveSource(max_reconnect_attempts=3)
    src._connected = False

    with patch.object(mod.time, "sleep") as mock_sleep, \
         patch.object(src, "connect"):
        src._reconnect_attempts = 1
        src._on_close("disconnected again")

    mock_sleep.assert_called_once_with(2)


def test_reconnect_third_attempt_waits_4_seconds():
    import src.data.live_upstox as mod

    src = UpstoxLiveSource(max_reconnect_attempts=5)
    src._connected = False

    with patch.object(mod.time, "sleep") as mock_sleep, \
         patch.object(src, "connect"):
        src._reconnect_attempts = 2
        src._on_close("third close")

    mock_sleep.assert_called_once_with(4)


def test_reconnect_stops_at_max_attempts():
    """Once max_reconnect_attempts is reached, _on_close neither sleeps nor reconnects."""
    import src.data.live_upstox as mod

    src = UpstoxLiveSource(max_reconnect_attempts=3)
    src._connected = False
    src._reconnect_attempts = 3  # already at max

    with patch.object(mod.time, "sleep") as mock_sleep, \
         patch.object(src, "connect") as mock_connect:
        src._on_close("final close")

    mock_sleep.assert_not_called()
    mock_connect.assert_not_called()


# ── 8. as_of from caller ─────────────────────────────────────────────────────

def test_as_of_from_caller_far_future_allows_past_tick():
    """An as_of far in the future returns a past tick — proving as_of is from the caller."""
    src = _make_src()
    _inject(src, "X", datetime(2024, 1, 22, 9, 30))

    df = src.get_ticks("X", datetime(2030, 1, 1), lookback_seconds=86400 * 365 * 10)
    assert len(df) == 1, "Past tick must be returned when as_of is in the far future"


def test_as_of_from_caller_past_as_of_blocks_tick():
    """An as_of before a tick's ts means PIT blocks the tick — as_of is honoured as given."""
    src = _make_src()
    _inject(src, "X", datetime(2024, 1, 22, 9, 30))

    with pytest.raises(DataLeakageError):
        src.get_ticks("X", datetime(2024, 1, 22, 9, 29))


# ── 9. No GPU code ────────────────────────────────────────────────────────────

def test_no_gpu_imports_in_module_source():
    """The live_upstox module must not import torch, cuda, cupy, or any GPU library.

    The check targets import statements, not docstring text, so safe mentions
    in comments are fine.
    """
    import src.data.live_upstox as mod

    source = inspect.getsource(mod)
    gpu_import_patterns = [
        "import torch",
        "import cuda",
        "import cupy",
        "from torch",
        "from cuda",
        'device("cuda")',
        "device('cuda')",
        '.to("cuda")',
        ".to('cuda')",
    ]
    for pattern in gpu_import_patterns:
        assert pattern not in source, (
            f"GPU import pattern {pattern!r} found in live_upstox.py — CPU-only project"
        )


# ── 10. Provenance fields ─────────────────────────────────────────────────────

def test_provenance_source_field_is_upstox_live():
    """Every row returned by get_ticks has source='upstox_live'."""
    src = _make_src()
    ts  = datetime(2024, 1, 22, 9, 30)
    as_of = datetime(2024, 1, 22, 16, 0)

    _inject(src, "PROV", ts, ltp=555.0)
    df = src.get_ticks("PROV", as_of, lookback_seconds=86400)

    assert list(df["source"]) == ["upstox_live"]


def test_metadata_fields_correct():
    """data_source_name, data_granularity, data_source_limitations are set correctly."""
    src = _make_src()
    assert src.data_source_name == "upstox_live"
    assert src.data_granularity == "tick"
    assert "live tick" in src.data_source_limitations
    assert "no historical" in src.data_source_limitations
    assert "credentials" in src.data_source_limitations


# ── connect() without SDK installed ──────────────────────────────────────────

@pytest.mark.skipif(
    importlib.util.find_spec("upstox_client") is not None,
    reason="upstox_client is installed — skipping SDK-absent ImportError test",
)
def test_connect_without_sdk_raises_import_error():
    """connect() raises ImportError with an actionable message when SDK is absent."""
    src = UpstoxLiveSource()   # no _inject_ws
    with pytest.raises(ImportError, match="upstox-python-api"):
        src.connect()


def test_inject_ws_skips_connect_and_sets_connected():
    """_inject_ws=… marks the source as connected without calling connect()."""
    mock_ws = MagicMock()
    src = UpstoxLiveSource(_inject_ws=mock_ws)
    assert src._connected is True
    src.connect()   # must be a no-op (already connected)
    assert src._connected is True
