
import duckdb
with duckdb.connect('data/market.duckdb', read_only=True) as db:
    symbols = db.execute('SELECT COUNT(DISTINCT symbol) FROM ohlcv_intraday').fetchone()[0]
    timeframes = db.execute('SELECT DISTINCT timeframe FROM ohlcv_intraday ORDER BY timeframe').fetchall()
    min_date = db.execute('SELECT CAST(MIN(ts) AS VARCHAR) FROM ohlcv_intraday').fetchone()[0]
    max_date = db.execute('SELECT CAST(MAX(ts) AS VARCHAR) FROM ohlcv_intraday').fetchone()[0]
    total_rows = db.execute('SELECT COUNT(*) FROM ohlcv_intraday').fetchone()[0]
    print(f'Symbols present: {symbols}')
    print(f'Timeframes present: {[t[0] for t in timeframes]}')
    print(f'Date Range: {min_date} to {max_date}')
    print(f'Total Rows: {total_rows}')

