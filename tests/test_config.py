"""Tests for config loading and validation."""
from __future__ import annotations

from src.config import Config, load_config


def test_config_loads_successfully() -> None:
    cfg = load_config()
    assert isinstance(cfg, Config)


def test_token_pool_is_500k() -> None:
    """Token pool must come from config — never hard-coded elsewhere."""
    cfg = load_config()
    assert cfg.trading.token_pool == 500_000


def test_universe_has_at_least_50_symbols() -> None:
    cfg = load_config()
    assert len(cfg.universe.symbols) >= 50


def test_universe_contains_nifty50_names() -> None:
    cfg = load_config()
    syms = set(cfg.universe.symbols)
    for expected in ("RELIANCE", "TCS", "INFY", "HDFCBANK", "SBIN"):
        assert expected in syms, f"{expected!r} missing from universe"


def test_no_duplicate_symbols() -> None:
    cfg = load_config()
    syms = cfg.universe.symbols
    assert len(syms) == len(set(syms))


def test_all_three_horizons_present() -> None:
    """15 / 30 / 60 min must all be present so cone-widening can be validated."""
    cfg = load_config()
    all_horizons = set(cfg.trading.additional_horizons_min) | {cfg.trading.primary_horizon_min}
    assert {15, 30, 60}.issubset(all_horizons)


def test_primary_horizon_is_30() -> None:
    cfg = load_config()
    assert cfg.trading.primary_horizon_min == 30


def test_quantile_levels_sorted_and_include_median() -> None:
    cfg = load_config()
    qls = cfg.model.quantile_levels
    assert qls == sorted(qls)
    assert 0.50 in qls


def test_quantile_levels_include_full_cone() -> None:
    cfg = load_config()
    qls = set(cfg.model.quantile_levels)
    for q in (0.10, 0.25, 0.50, 0.75, 0.90):
        assert q in qls, f"quantile level {q} missing"


def test_config_hash_is_stable_across_calls() -> None:
    cfg = load_config()
    assert cfg.config_hash() == cfg.config_hash()


def test_config_hash_is_hex_string() -> None:
    h = load_config().config_hash()
    assert len(h) == 64
    int(h, 16)   # raises ValueError if not valid hex


def test_data_dir_is_absolute() -> None:
    cfg = load_config()
    assert cfg.data.data_dir.is_absolute()


def test_experiment_db_path_is_absolute() -> None:
    cfg = load_config()
    assert cfg.experiment.db_path.is_absolute()
