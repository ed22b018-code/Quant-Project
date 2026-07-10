import time
import pandas as pd
from datetime import datetime, timezone
import logging
from src.models.labels import TripleBarrierLabeler
from src.models.dataset import DatasetBuilder

UTC = timezone.utc

def test_full_worker():
    db_path = r"data\market.duckdb"
    sym = "RELIANCE"
    
    # generate some timestamps
    start_date = "2023-07-04"
    end_date = "2026-07-03"
    bdays = pd.bdate_range(start=start_date, end=end_date)
    timestamps = [datetime(d.year, d.month, d.day, 4, 0, tzinfo=UTC) for d in bdays]

    labeler = TripleBarrierLabeler(profit_pct=0.01, stop_pct=0.005, horizon_bars=30)
    builder = DatasetBuilder(feature_names=[], gates_open=None)
    
    print("Starting worker...")
    t0 = time.time()
    
    import duckdb
    with duckdb.connect(db_path, read_only=True) as con:
        df_intra = con.execute(
            f"SELECT ts, close FROM ohlcv_intraday WHERE symbol = '{sym}' ORDER BY ts"
        ).df()
        
    print(f"Data fetched in {time.time() - t0:.2f}s")
    
    t1 = time.time()
    labels_df = None
    if not df_intra.empty:
        labels_df = labeler.label_series(df_intra["close"], df_intra["ts"])
    print(f"Labels computed in {time.time() - t1:.2f}s")
    
    t2 = time.time()
    res = builder.build(sym, timestamps, db_path, labels_df=labels_df)
    print(f"Builder finished in {time.time() - t2:.2f}s")
    
    print(f"Total time: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_full_worker()
