"""Live data source factory — config-driven adapter selection.

Swapping from Dhan live to Upstox free-tier after the subscription expires
is a one-line config change:

    # config/config.toml or .env override
    [data]
    live_data_source = "upstox"   # was "dhan"

No code changes required upstream of this factory.

Supported sources
-----------------
  "dhan"   — DhanDataSource (from dhan_adapter.py)
             Credentials: DHAN_CLIENT_ID + DHAN_ACCESS_TOKEN in .env
  "upstox" — UpstoxLiveSource (from live_upstox.py)
             Credentials: UPSTOX_API_KEY + UPSTOX_ACCESS_TOKEN in .env

Both implement the full DataSource interface; neither is imported at module level
(deferred to get_live_source call) so this factory is importable without either
SDK installed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.data.base import DataSource


_SUPPORTED = {"dhan", "upstox"}


def get_live_source(config: Any, db_path: str | None = None) -> "DataSource":
    """Return the configured live DataSource adapter.

    Parameters
    ----------
    config:
        Project config object (pydantic model from src/config.py).
        Reads config.data.live_data_source and config.data.duckdb_path.
    db_path:
        Override for the DuckDB path (used in tests).

    Raises
    ------
    ValueError
        If config.data.live_data_source is not a recognised source name.
    """
    source_name = getattr(config.data, "live_data_source", "dhan").strip().lower()

    if source_name not in _SUPPORTED:
        raise ValueError(
            f"Unknown live_data_source {source_name!r}.  "
            f"Valid options: {sorted(_SUPPORTED)}.  "
            "Check [data] live_data_source in config.toml."
        )

    resolved_db = db_path or str(getattr(config.data, "duckdb_path", "data/market.duckdb"))

    if source_name == "dhan":
        from src.data.dhan_adapter import DhanDataSource  # noqa: PLC0415
        return DhanDataSource(db_path=resolved_db)

    # source_name == "upstox"
    from src.data.live_upstox import UpstoxLiveSource  # noqa: PLC0415
    return UpstoxLiveSource()
