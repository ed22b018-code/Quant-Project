
import duckdb
with duckdb.connect('data/market.duckdb', read_only=True) as db:
    statuses = db.execute('SELECT status, COUNT(*) FROM download_jobs GROUP BY status').fetchall()
    print('Statuses:', statuses)

