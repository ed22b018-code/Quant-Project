# tools/

One-off operational scripts used during the 3060 data-collection run.
These are NOT part of the production pipeline and are NOT imported by any `src/` module.

| Script | Purpose | Notes |
|---|---|---|
| `filter_symbols.py` | Queries `download_jobs` for successfully-downloaded symbols, writes `formatted_symbols.txt` | Used to assemble the 111-symbol list in `config.toml` |
| `get_symbols.py` | Queries `dhan_instruments` for FUTSTK symbols, strips expiry suffixes | Used to inspect available futures tickers |
| `verify.py` | Prints row counts and date ranges from `ohlcv_intraday` | Hardcodes `data/market.duckdb` — edit before use |
| `verify_status.py` | Prints `download_jobs` status counts | Hardcodes `data/market.duckdb` — edit before use |
| `test_labeling.py` | Ad-hoc labeling test against real downloaded data | Not a pytest test — run manually |

All scripts hardcode `data/market.duckdb` which only exists on the 3060 training machine.
They will not run on the dev laptop without that file present.

`update_config.py` was deleted: it had a Python string-quoting bug that made it
unrunnable (`''FUTSTK''` compiled to a SQL identifier rather than a quoted string).
The 111-symbol `config.toml` list was assembled via `filter_symbols.py` + manual editing.
