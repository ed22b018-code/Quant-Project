"""Configuration loader — single source of truth for all runtime settings.

Load order: src/config.py reads config/config.toml via stdlib tomllib (Python 3.11+),
validates with pydantic v2, and caches the result.  All other modules import from
here; nothing else reads config.toml directly.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from typing import Literal

from pydantic import BaseModel, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.toml"


def _resolve(raw: str) -> Path:
    """Return absolute path; resolve relative to PROJECT_ROOT."""
    p = Path(raw)
    return p if p.is_absolute() else PROJECT_ROOT / p


# ── Sub-models ────────────────────────────────────────────────────────────────

class ProjectConfig(BaseModel):
    name: str
    version: str


class DataConfig(BaseModel):
    historical_source: str
    data_dir: Path
    snapshots_dir: Path
    duckdb_path: Path
    sqlite_path: Path

    @field_validator("data_dir", "snapshots_dir", "duckdb_path", "sqlite_path", mode="before")
    @classmethod
    def _to_abs(cls, v: str) -> Path:
        return _resolve(v)


class UniverseConfig(BaseModel):
    symbols: list[str]

    @field_validator("symbols")
    @classmethod
    def _no_duplicates(cls, v: list[str]) -> list[str]:
        seen: set[str] = set()
        for s in v:
            if s in seen:
                raise ValueError(f"Duplicate symbol in universe: {s!r}")
            seen.add(s)
        return v


class CostConfig(BaseModel):
    """Rate parameters for NSE intraday equity cost simulation.

    All rates sourced from published regulatory/exchange schedules.
    When official rates change, update config.toml [trading.costs] only —
    no code changes needed anywhere else.
    """
    brokerage_model: Literal["flat", "percentage"] = "percentage"
    brokerage_pct: float = 0.0003
    brokerage_flat_cap: float = 20.0
    stt_intraday_sell_pct: float = 0.00025
    exchange_charges_pct: float = 0.0000297
    sebi_fees_pct: float = 0.000001
    gst_pct: float = 0.18
    stamp_duty_intraday_buy_pct: float = 0.00003

    @field_validator(
        "brokerage_pct", "brokerage_flat_cap", "stt_intraday_sell_pct",
        "exchange_charges_pct", "sebi_fees_pct", "gst_pct",
        "stamp_duty_intraday_buy_pct",
    )
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"Cost rate must be >= 0, got {v}")
        return v


class TradingConfig(BaseModel):
    token_pool: int
    primary_horizon_min: int
    additional_horizons_min: list[int]
    market_open: str
    market_close: str
    square_off_time: str
    max_position_pct: float
    max_intraday_exposure_pct: float
    max_delivery_exposure_pct: float
    min_cash_reserve_pct: float
    drawdown_pause_pct: float
    max_sector_exposure_pct: float
    costs: CostConfig

    @model_validator(mode="after")
    def _primary_in_all_horizons(self) -> "TradingConfig":
        all_h = set(self.additional_horizons_min) | {self.primary_horizon_min}
        if self.primary_horizon_min not in all_h:
            raise ValueError("primary_horizon_min must appear in additional_horizons_min")
        return self


class ModelConfig(BaseModel):
    random_seed: int
    quantile_levels: list[float]
    max_cpu_cores: int = -1

    @field_validator("quantile_levels")
    @classmethod
    def _sorted_and_valid(cls, v: list[float]) -> list[float]:
        if not all(0.0 < q < 1.0 for q in v):
            raise ValueError("quantile_levels must all be in the open interval (0, 1)")
        if 0.50 not in v:
            raise ValueError("quantile_levels must include 0.50 (median = expected drift)")
        return sorted(v)


class ExperimentConfig(BaseModel):
    db_path: Path

    @field_validator("db_path", mode="before")
    @classmethod
    def _to_abs(cls, v: str) -> Path:
        return _resolve(v)


class LoggingConfig(BaseModel):
    level: str
    log_dir: Path

    @field_validator("log_dir", mode="before")
    @classmethod
    def _to_abs(cls, v: str) -> Path:
        return _resolve(v)


# ── Top-level config ──────────────────────────────────────────────────────────

class Config(BaseModel):
    project: ProjectConfig
    data: DataConfig
    universe: UniverseConfig
    trading: TradingConfig
    model: ModelConfig
    experiment: ExperimentConfig
    logging: LoggingConfig

    def as_dict(self) -> dict[str, Any]:
        """Fully serialisable dict (Paths → strings)."""
        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        """SHA-256 of the canonical JSON serialisation — fingerprints experiment runs."""
        canonical = json.dumps(self.as_dict(), sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()


# ── Loader (cached) ───────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def load_config(path: Path = _DEFAULT_CONFIG_PATH) -> Config:
    """Load, validate, and cache the TOML config.  Call once; re-use everywhere."""
    with open(path, "rb") as fh:
        raw = tomllib.load(fh)
    return Config.model_validate(raw)
