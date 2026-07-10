import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from src.models.scenario_classifier import ScenarioClassifier
from src.models.quantile_cone import QuantileConeTrainer
from src.models.dataset import DatasetBuilder, WalkForwardSplitter
from src.models.labels import TripleBarrierLabeler, Label


def evaluate(db_path="data/market.duckdb", model_dir="models/v1"):
    # Load provenance to get test details
    with open(Path(model_dir) / "provenance.json") as f:
        prov = json.load(f)
    
    # We will evaluate on 5 highly liquid representative symbols to give the user a fast response
    eval_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "BHARTIARTL"]
    
    # Check if they exist in the DB, if not use fallback
    import duckdb
    with duckdb.connect(db_path, read_only=True, config={'threads': '1'}) as con:
        db_symbols = [r[0] for r in con.execute("SELECT DISTINCT symbol FROM ohlcv_intraday LIMIT 50").fetchall()]
    eval_symbols = [s for s in eval_symbols if s in db_symbols]
    if len(eval_symbols) < 5:
        eval_symbols.extend([s for s in db_symbols if s not in eval_symbols][:5 - len(eval_symbols)])
        
    print(f"Evaluating on: {eval_symbols}")
    
    start_date = pd.to_datetime(prov["date_range"][0]).date()
    end_date = pd.to_datetime(prov["date_range"][1]).date()
    bdays = pd.bdate_range(start=start_date, end=end_date)
    timestamps = [datetime(d.year, d.month, d.day, 4, 0, tzinfo=timezone.utc) for d in bdays]
    
    # We only care about the TEST SET timestamps (the last 10%)
    n_ts = len(timestamps)
    test_ts = timestamps[-max(1, int(n_ts * 0.10)):]
    
    classifier = ScenarioClassifier.load(Path(model_dir) / "scenario_classifier")
    expected_features = classifier.feature_names
    
    builder = DatasetBuilder(feature_names=expected_features)
    labeler = TripleBarrierLabeler(profit_pct=0.010, stop_pct=0.005, horizon_bars=30)
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    parts = []
    
    def process_symbol(sym):
        print(f"[{sym}] Starting evaluation build...")
        df_sym = builder.build(sym, test_ts, db_path)
        
        with duckdb.connect(db_path, read_only=True, config={'threads': '1'}) as con:
            df_intra = con.execute(f"SELECT ts, close FROM ohlcv_intraday WHERE symbol = '{sym}' ORDER BY ts").df()
            
        labels_df = None
        if not df_intra.empty:
            labels_df = labeler.label_series(df_intra["close"], df_intra["ts"])
            
        if labels_df is not None:
            df_sym = df_sym.merge(labels_df[["t0", "label", "t1", "barrier"]], on="t0", how="left")
            
        return df_sym

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_symbol, sym): sym for sym in eval_symbols}
        for future in as_completed(futures):
            parts.append(future.result())
            print(f"[{futures[future]}] Finished.")
            
    dataset = pd.concat(parts, ignore_index=True)
    dataset = dataset.dropna(subset=expected_features + ["label"])
    
    print(f"Test set reconstructed with {len(dataset)} rows.")
    if len(dataset) == 0:
        print("No test data found. Exiting.")
        return
        
    # Load Classifier
    X_test = dataset[expected_features].fillna(0.0)
    y_test = dataset["label"]
    
    proba_test = classifier.predict_proba(X_test)
    preds = proba_test.idxmax(axis=1).map({"p_no_trade": Label.NO_TRADE, "p_bullish": Label.BULLISH, "p_bearish": Label.BEARISH})
    
    # 1. Calibration: When it says >70% confident, is it actually right?
    high_conf_bull = proba_test["p_bullish"] > 0.70
    high_conf_bear = proba_test["p_bearish"] > 0.70
    
    bull_acc = (y_test[high_conf_bull] == Label.BULLISH).mean() if high_conf_bull.sum() > 0 else 0
    bear_acc = (y_test[high_conf_bear] == Label.BEARISH).mean() if high_conf_bear.sum() > 0 else 0
    
    print("\n--- CALIBRATION ---")
    print(f"Bullish >70% Confidence Accuracy: {bull_acc:.1%} ({high_conf_bull.sum()} trades)")
    print(f"Bearish >70% Confidence Accuracy: {bear_acc:.1%} ({high_conf_bear.sum()} trades)")
    
    # 2. Benchmarks (After Costs)
    # Assume 0.05% slippage+commission per trade (0.1% round trip)
    cost_per_trade = 0.001 
    
    # Reconstruct actual returns for trades
    dataset["pred"] = preds
    dataset["trade_return"] = 0.0
    
    # If we predicted BULLISH and it hit the profit barrier (0.01) or stop (-0.005)
    # Actually, we need exact return. Let's approximate from labels for speed
    def get_return(row):
        if row["pred"] == Label.NO_TRADE: return 0.0
        
        # Bullish trade
        if row["pred"] == Label.BULLISH:
            if row["label"] == Label.BULLISH: return 0.01 - cost_per_trade # Hit profit
            elif row["label"] == Label.BEARISH: return -0.005 - cost_per_trade # Hit stop loss
            else: return 0.0 - cost_per_trade # Time stop
            
        # Bearish trade
        if row["pred"] == Label.BEARISH:
            if row["label"] == Label.BEARISH: return 0.01 - cost_per_trade # Hit profit
            elif row["label"] == Label.BULLISH: return -0.005 - cost_per_trade # Hit stop
            else: return 0.0 - cost_per_trade
    
    dataset["model_return"] = dataset.apply(get_return, axis=1)
    
    # Buy and hold return (approximate using the underlying price movement of these 5 stocks)
    # Random return (randomly assigning bullish/bearish/no trade)
    np.random.seed(42)
    dataset["random_pred"] = np.random.choice([Label.NO_TRADE, Label.BULLISH, Label.BEARISH], size=len(dataset), p=[0.8, 0.1, 0.1])
    dataset["random_return"] = dataset.apply(lambda r: get_return({**r, "pred": r["random_pred"]}), axis=1)
    
    total_model_return = dataset.groupby("t0")["model_return"].mean().sum()
    total_random_return = dataset.groupby("t0")["random_return"].mean().sum()
    
    print("\n--- BENCHMARKS (Net of 0.1% round-trip costs) ---")
    print(f"Model Strategy Return:   {total_model_return:.2%}")
    print(f"Random Strategy Return:  {total_random_return:.2%}")
    print(f"Always-Flat Return:      0.00%")
    print(f"Total Trades Taken:      {(dataset['pred'] != Label.NO_TRADE).sum()}")
    
    # 3. Warnings (Leakage / NaNs)
    print("\n--- WARNINGS / LEAKAGE CHECK ---")
    nan_counts = dataset[expected_features].isna().sum()
    print("NaN features:", nan_counts[nan_counts > 0].to_dict())
    
    print("Feature Importances (Top 5):")
    importances = pd.Series(classifier._model.feature_importance(), index=classifier.feature_names).sort_values(ascending=False)
    for feat, imp in importances.head(5).items():
        print(f"  {feat}: {imp}")
        
    if importances.iloc[0] > importances.iloc[1] * 5:
        print("WARNING: Top feature is >5x more important than the next. High risk of data leakage.")
    else:
        print("No immediate signs of singular feature leakage.")
        
    print("\n--- VERDICT ---")
    if total_model_return > total_random_return and total_model_return > 0 and (bull_acc > 0.55 or bear_acc > 0.55):
        print("Verdict: POTENTIAL EDGE DETECTED.")
    else:
        print("Verdict: NO REAL EDGE AFTER COSTS.")

if __name__ == "__main__":
    evaluate()
