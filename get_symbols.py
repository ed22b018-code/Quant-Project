
import duckdb
conn = duckdb.connect('data/market.duckdb', read_only=True)
res = conn.execute('SELECT DISTINCT symbol FROM dhan_instruments WHERE instrument_type=\'FUTSTK\'').fetchall()
symbols = sorted(list(set('-'.join(row[0].split('-')[:-2]) for row in res if 'TEST' not in row[0])))
print(repr(symbols))
print('Count:', len(symbols))

