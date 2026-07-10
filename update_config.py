
import duckdb
import tomli
import tomli_w
from pathlib import Path

conn = duckdb.connect('data/market.duckdb', read_only=True)
symbols = [row[0] for row in conn.execute('SELECT DISTINCT symbol FROM dhan_instruments WHERE instrument_type=''FUTSTK''').fetchall()]

config_path = Path('config/config.toml')
with open(config_path, 'rb') as f:
    config = tomli.load(f)

config['universe']['symbols'] = symbols

with open(config_path, 'wb') as f:
    tomli_w.dump(config, f)
print(f'Updated config.toml with {len(symbols)} symbols.')

