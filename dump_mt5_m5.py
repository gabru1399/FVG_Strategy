r"""
Dump M5 historical data from live MT5 Blueberry Markets account.
Saves to C:\Users\SIGMA\Documents\FVG_Strategy\Blueberry_Data as <symbol>_5M.parquet
"""

import os
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timezone

DATA_DIR = r"C:\Users\SIGMA\Documents\FVG_Strategy\Blueberry_Data"
SYMBOLS = ["EURUSD.pi", "GBPUSD.pi", "USDJPY.pi", "XAUUSD.pi"]

def dump_symbol_m5(sym):
    print(f"Fetching M5 data for {sym}...")
    cur_dt = datetime.now(timezone.utc)
    all_chunks = []
    
    for step in range(5):
        rates = mt5.copy_rates_from(sym, mt5.TIMEFRAME_M5, cur_dt, 50000)
        if rates is None or len(rates) == 0:
            break
        df_chunk = pd.DataFrame(rates)
        df_chunk['time'] = pd.to_datetime(df_chunk['time'], unit='s', utc=True)
        all_chunks.append(df_chunk)
        
        t0 = df_chunk['time'].iloc[0]
        if len(rates) < 1000 or (t0 >= cur_dt):
            break
        cur_dt = t0
        
    if not all_chunks:
        print(f"  No data retrieved for {sym}")
        return None
        
    df_all = pd.concat(all_chunks, ignore_index=True)
    df_all = df_all.drop_duplicates(subset=['time']).sort_values('time').reset_index(drop=True)
    
    out_path = os.path.join(DATA_DIR, f"{sym}_5M.parquet")
    df_all.to_parquet(out_path, index=False)
    print(f"  Saved {len(df_all)} bars to {os.path.basename(out_path)} ({df_all['time'].iloc[0]} to {df_all['time'].iloc[-1]})")
    return out_path

def main():
    if not mt5.initialize():
        print("Failed to initialize MT5:", mt5.last_error())
        return
        
    acc = mt5.account_info()
    print(f"Connected to MT5 Account: {acc.login} ({acc.server})")
    
    for sym in SYMBOLS:
        dump_symbol_m5(sym)
        
    mt5.shutdown()
    print("M5 data dump completed successfully.")

if __name__ == "__main__":
    main()
