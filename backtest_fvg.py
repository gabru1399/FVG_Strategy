"""
Multi-Asset A/A+ Momentum Fair Value Gap (FVG) Strategy Backtester
Tested on Blueberry Markets MT5 data (C:\\Users\\SIGMA\\Documents\\FVG_Strategy\\Blueberry_Data)
Based on: FVG_Strategy.pdf (Teghjot Singh Bindra, 2025)
Author: Antigravity (Ponytail Minimal Code Protocol)
"""

import os
import glob
import hashlib
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\SIGMA\Documents\FVG_Strategy\Blueberry_Data"
CORE_ASSETS = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD"]
EXTENDED_ASSETS = [
    "EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "AUDUSD", "NZDUSD", 
    "USDCAD", "USDCHF", "EURJPY", "GBPJPY", "AUDJPY", "EURGBP"
]

def load_data(symbol, tf="1H"):
    path = os.path.join(DATA_DIR, f"{symbol}.pi_{tf}.parquet")
    if not os.path.exists(path):
        path = os.path.join(DATA_DIR, f"{symbol}.p_{tf}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.sort_values('time').reset_index(drop=True)
        if tf == "1H":
            df = df[df['time'] >= '2023-03-22'].reset_index(drop=True)
    return df

def compute_htf_indicators(df_4h):
    """Computes EMA-20, EMA-50, EMA-233, and 30-period rolling Point of Control (POC)."""
    df = df_4h.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema233_4h'] = df['close'].ewm(span=233, adjust=False).mean()
    
    # 30-period rolling POC via price histogram
    pocs = [np.nan] * len(df)
    closes = df['close'].values
    for i in range(30, len(df)):
        window = closes[i-30:i]
        counts, bin_edges = np.histogram(window, bins=15)
        max_idx = np.argmax(counts)
        pocs[i] = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2.0
    df['poc30'] = pocs
    return df

def get_pip_size(symbol, price):
    """Return pip size for distance calculations."""
    if "JPY" in symbol:
        return 0.01
    elif "XAU" in symbol or "GOLD" in symbol:
        return 0.10
    elif any(c in symbol for c in ["BTC", "ETH", "BNB"]):
        return 1.0
    elif price > 500:
        return 0.1
    else:
        return 0.0001

def find_setups(df_1h, df_htf, symbol, ema233_mode=None):
    """
    Scans for Momentum Legs, FVGs, and HTF confluence per FVG_Strategy.pdf.
    Optionally filters by EMA 233 trend:
      - ema233_mode='1H': Long only if price > 1H EMA233, Short only if price < 1H EMA233.
      - ema233_mode='4H': Long only if price > 4H EMA233, Short only if price < 4H EMA233.
    """
    pip_size = get_pip_size(symbol, df_1h['close'].mean())
    df_1h = df_1h.copy()
    
    # 1H EMA 233
    df_1h['ema233_1h'] = df_1h['close'].ewm(span=233, adjust=False).mean()
    
    # 1H Body ratio & Direction
    body = (df_1h['close'] - df_1h['open']).abs()
    rng = (df_1h['high'] - df_1h['low']).replace(0, 1e-6)
    df_1h['body_ratio'] = body / rng
    df_1h['is_bull'] = df_1h['close'] > df_1h['open']
    df_1h['is_bear'] = df_1h['close'] < df_1h['open']
    
    # Swing High/Low over 10 prior bars
    df_1h['swing_high_10'] = df_1h['high'].shift(2).rolling(10).max()
    df_1h['swing_low_10'] = df_1h['low'].shift(2).rolling(10).min()
    
    # Merge HTF indicators using merge_asof
    htf_cols = ['time', 'ema20', 'ema50', 'poc30']
    if 'ema233_4h' in df_htf.columns:
        htf_cols.append('ema233_4h')
    df_htf_clean = df_htf[htf_cols].dropna()
    df_merged = pd.merge_asof(df_1h, df_htf_clean, on='time', direction='backward')
    
    setups = []
    n = len(df_merged)
    
    for i in range(12, n - 25):
        # 1. Momentum Leg Check (2 consecutive directional candles)
        br1 = df_merged.at[i-1, 'body_ratio']
        br2 = df_merged.at[i, 'body_ratio']
        bull1 = df_merged.at[i-1, 'is_bull']
        bull2 = df_merged.at[i, 'is_bull']
        bear1 = df_merged.at[i-1, 'is_bear']
        bear2 = df_merged.at[i, 'is_bear']
        c2 = df_merged.at[i, 'close']
        sh = df_merged.at[i, 'swing_high_10']
        sl = df_merged.at[i, 'swing_low_10']
        
        is_bull_leg = bull1 and bull2 and (c2 > sh)
        is_bear_leg = bear1 and bear2 and (c2 < sl)
        
        if not (is_bull_leg or is_bear_leg):
            continue
            
        direction = "Long" if is_bull_leg else "Short"
        
        # Grading threshold
        grade_candidate = "A+" if (br1 >= 0.70 and br2 >= 0.70) else ("A" if (br1 >= 0.52 and br2 >= 0.52) else None)
        if grade_candidate is None:
            continue
            
        # 2. FVG Search in subsequent window (up to 24 bars)
        for k in range(i + 1, min(i + 25, n - 1)):
            fvg_found = False
            midpoint = None
            fvg_sl = None
            
            if direction == "Long":
                h_k2 = df_merged.at[k-2, 'high']
                l_k = df_merged.at[k, 'low']
                if h_k2 < l_k:
                    fvg_found = True
                    midpoint = (h_k2 + l_k) / 2.0
                    fvg_sl = h_k2
            else:
                l_k2 = df_merged.at[k-2, 'low']
                h_k = df_merged.at[k, 'high']
                if l_k2 > h_k:
                    fvg_found = True
                    midpoint = (l_k2 + h_k) / 2.0
                    fvg_sl = l_k2
                    
            if not fvg_found:
                continue
                
            # 3. Check for entry: Price taps midpoint in subsequent bars
            entry_idx = None
            for m in range(k + 1, min(k + 25, n)):
                bar_low = df_merged.at[m, 'low']
                bar_high = df_merged.at[m, 'high']
                if bar_low <= midpoint <= bar_high:
                    entry_idx = m
                    break
                    
            if entry_idx is None:
                continue
                
            entry_time = df_merged.at[entry_idx, 'time']
            hour_utc = entry_time.hour
            
            # 4. Session Eligibility
            session = None
            wr_bonus = 0.0
            if 8 <= hour_utc < 11:
                session = "London"
                wr_bonus = 0.05
            elif 13 <= hour_utc < 15:
                session = "Overlap"
                wr_bonus = 0.06
            elif 15 <= hour_utc < 16:
                session = "New York"
                wr_bonus = 0.02
            else:
                continue
                
            # 5. HTF Confluence Check
            ema20 = df_merged.at[entry_idx, 'ema20']
            ema50 = df_merged.at[entry_idx, 'ema50']
            poc = df_merged.at[entry_idx, 'poc30']
            
            htf_aligned = False
            prox_limit = max(15 * pip_size, midpoint * 0.0015)
            for ref in [ema20, ema50, poc]:
                if pd.notna(ref) and abs(midpoint - ref) <= prox_limit:
                    htf_aligned = True
                    break
                    
            if pd.notna(ema50):
                if direction == "Long" and midpoint < ema50 * 0.995:
                    htf_aligned = False
                elif direction == "Short" and midpoint > ema50 * 1.005:
                    htf_aligned = False
                    
            if grade_candidate == "A+" and htf_aligned:
                final_grade = "A+"
            elif grade_candidate in ["A", "A+"]:
                final_grade = "A"
            else:
                continue
                
            # 6. EMA 233 Trend Filter (User Constraint)
            if ema233_mode == "1H":
                ema233 = df_merged.at[entry_idx, 'ema233_1h']
                if pd.notna(ema233):
                    if direction == "Long" and midpoint <= ema233:
                        continue
                    elif direction == "Short" and midpoint >= ema233:
                        continue
            elif ema233_mode == "4H":
                if 'ema233_4h' in df_merged.columns:
                    ema233 = df_merged.at[entry_idx, 'ema233_4h']
                    if pd.notna(ema233):
                        if direction == "Long" and midpoint <= ema233:
                            continue
                        elif direction == "Short" and midpoint >= ema233:
                            continue
                
            risk_dist = abs(midpoint - fvg_sl)
            min_risk = 9 * pip_size
            if risk_dist < min_risk:
                risk_dist = min_risk
                fvg_sl = midpoint - risk_dist if direction == "Long" else midpoint + risk_dist
                
            setups.append({
                'symbol': symbol,
                'entry_idx': entry_idx,
                'entry_time': entry_time,
                'direction': direction,
                'grade': final_grade,
                'session': session,
                'htf_aligned': htf_aligned,
                'entry_price': midpoint,
                'stop_loss': fvg_sl,
                'risk_dist': risk_dist,
                'wr_bonus': wr_bonus,
                'pip_size': pip_size
            })
            break
            
    return df_merged, pd.DataFrame(setups)

def simulate_real_outcomes(df, setups_df, rr_ratio=1.0, max_holding_bars=48):
    if setups_df.empty:
        return setups_df
        
    trades = []
    n = len(df)
    
    for _, s in setups_df.iterrows():
        idx = s['entry_idx']
        entry_p = s['entry_price']
        sl = s['stop_loss']
        risk = s['risk_dist']
        direction = s['direction']
        tp = entry_p + (risk * rr_ratio) if direction == "Long" else entry_p - (risk * rr_ratio)
        
        result = None
        for f in range(idx + 1, min(idx + max_holding_bars + 1, n)):
            high = df.at[f, 'high']
            low = df.at[f, 'low']
            
            if direction == "Long":
                hit_sl = low <= sl
                hit_tp = high >= tp
                if hit_sl and hit_tp:
                    result = "Loss"
                    break
                elif hit_sl:
                    result = "Loss"
                    break
                elif hit_tp:
                    result = "Win"
                    break
            else:
                hit_sl = high >= sl
                hit_tp = low <= tp
                if hit_sl and hit_tp:
                    result = "Loss"
                    break
                elif hit_sl:
                    result = "Loss"
                    break
                elif hit_tp:
                    result = "Win"
                    break
                    
        if result is None:
            last_c = df.at[min(idx + max_holding_bars, n - 1), 'close']
            r_return = (last_c - entry_p) / risk if direction == "Long" else (entry_p - last_c) / risk
            result = "Win" if r_return > 0 else "Loss"
            r_captured = r_return
        else:
            r_captured = rr_ratio if result == "Win" else -1.0
            
        t_record = dict(s)
        t_record['real_result'] = result
        t_record['real_r'] = r_captured
        t_record['real_pnl'] = r_captured * 250.0
        trades.append(t_record)
        
    return pd.DataFrame(trades)

def simulate_paper_hash_model(trades_df):
    if trades_df.empty:
        return trades_df
        
    sim_trades = []
    for _, t in trades_df.iterrows():
        base_wr = 0.76 if t['grade'] == "A+" else 0.64
        session_bonus = t.get('wr_bonus', 0.0)
        htf_bonus = 0.04 if t['htf_aligned'] else 0.0
        win_prob = base_wr + session_bonus + htf_bonus
        
        hash_input = f"{t['symbol']}_{t['entry_time']}_{t['direction']}_{t['grade']}"
        h_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 10000 / 10000.0
        
        is_win = h_val < win_prob
        r_mult = 1.0 if is_win else -1.0
        
        record = dict(t)
        record['hash_result'] = "Trail 1:1" if is_win else "Loss"
        record['hash_r'] = r_mult
        record['hash_pnl'] = r_mult * 250.0
        sim_trades.append(record)
        
    return pd.DataFrame(sim_trades)

def compute_metrics(df_trades, r_col, pnl_col):
    if df_trades.empty:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0, 'total_r': 0.0, 
                'net_pnl': 0.0, 'return_pct': 0.0, 'profit_factor': 0.0, 'max_dd_dollars': 0.0, 
                'max_dd_pct': 0.0, 'expectancy_r': 0.0}
    n = len(df_trades)
    wins = (df_trades[r_col] > 0).sum()
    losses = (df_trades[r_col] <= 0).sum()
    wr = (wins / n) * 100.0 if n > 0 else 0.0
    tot_r = df_trades[r_col].sum()
    tot_pnl = df_trades[pnl_col].sum()
    
    gross_win = df_trades[df_trades[pnl_col] > 0][pnl_col].sum()
    gross_loss = abs(df_trades[df_trades[pnl_col] < 0][pnl_col].sum())
    pf = (gross_win / gross_loss) if gross_loss > 0 else (99.0 if gross_win > 0 else 0.0)
    
    equity = 50000.0 + df_trades[pnl_col].cumsum()
    peak = equity.cummax()
    dd = equity - peak
    max_dd = dd.min()
    max_dd_pct = (max_dd / peak.max()) * 100.0 if peak.max() > 0 else 0.0
    
    avg_r = df_trades[r_col].mean()
    
    return {
        'trades': n,
        'wins': wins,
        'losses': losses,
        'win_rate': wr,
        'total_r': tot_r,
        'net_pnl': tot_pnl,
        'return_pct': (tot_pnl / 50000.0) * 100.0,
        'profit_factor': pf,
        'max_dd_dollars': max_dd,
        'max_dd_pct': max_dd_pct,
        'expectancy_r': avg_r
    }

def run_comprehensive_study():
    print("=" * 85)
    print("      FVG STRATEGY COMPREHENSIVE PERFORMANCE AUDIT ON BLUEBERRY MT5 DATA")
    print("               Paper Spec: Teghjot Singh Bindra (2025)")
    print("=" * 85)
    
    all_data = {}
    for sym in EXTENDED_ASSETS:
        df_1h = load_data(sym, "1H")
        df_4h = load_data(sym, "4H")
        if df_1h is not None and df_4h is not None:
            df_htf = compute_htf_indicators(df_4h)
            df_merged, setups_df = find_setups(df_1h, df_htf, sym)
            all_data[sym] = (df_merged, setups_df)
            print(f"Loaded {sym:7s} | 1H bars: {len(df_1h):5d} | Setups: {len(setups_df):4d}")
            
    # SECTION 1: 4 Core Paper Assets (2025 vs All-Time)
    print("\n" + "=" * 85)
    print("SECTION 1: CORE 4 ASSETS (EURUSD, GBPUSD, USDJPY, XAUUSD)")
    print("=" * 85)
    
    core_trades_real = []
    for sym in CORE_ASSETS:
        df_merged, setups = all_data[sym]
        real_t = simulate_real_outcomes(df_merged, setups, rr_ratio=1.0)
        full_t = simulate_paper_hash_model(real_t)
        core_trades_real.append(full_t)
        
    core_df = pd.concat(core_trades_real, ignore_index=True)
    core_df['year'] = core_df['entry_time'].dt.year
    df_2025 = core_df[core_df['year'] == 2025].reset_index(drop=True)
    
    m_paper_25 = compute_metrics(df_2025, 'hash_r', 'hash_pnl')
    m_real_25 = compute_metrics(df_2025, 'real_r', 'real_pnl')
    m_real_all = compute_metrics(core_df, 'real_r', 'real_pnl')
    
    print("\n[TABLE 1: HEAD-TO-HEAD COMPARISON - 2025 PERIOD]")
    print(f"{'Metric':<25} | {'Paper Claims':<18} | {'Paper Model (Blueberry)':<24} | {'Real MT5 Execution'}")
    print("-" * 85)
    print(f"{'Trades':<25} | {'473':<18} | {m_paper_25['trades']:<24d} | {m_real_25['trades']}")
    print(f"{'Win Rate':<25} | {'78.0%':<18} | {m_paper_25['win_rate']:<23.1f}% | {m_real_25['win_rate']:.1f}%")
    print(f"{'Total R':<25} | {'+265.0 R':<18} | {m_paper_25['total_r']:<+23.1f}R | {m_real_25['total_r']:+.1f}R")
    print(f"{'Net P&L ($50k Acc)':<25} | {'+$66,250':<18} | ${m_paper_25['net_pnl']:<+23,.0f} | ${m_real_25['net_pnl']:+,.0f}")
    print(f"{'Return on Capital':<25} | {'+132.5%':<18} | {m_paper_25['return_pct']:<+23.1f}% | {m_real_25['return_pct']:+.1f}%")
    print(f"{'Max Drawdown':<25} | {'-2.0% (-$1,000)':<18} | {m_paper_25['max_dd_pct']:<23.1f}% | {m_real_25['max_dd_pct']:.1f}% (${m_real_25['max_dd_dollars']:,.0f})")
    print(f"{'Profit Factor':<25} | {'N/A (Trail 1:1)':<18} | {m_paper_25['profit_factor']:<24.2f} | {m_real_25['profit_factor']:.2f}")
    print(f"{'Expectancy per Trade':<25} | {'+0.56 R':<18} | {m_paper_25['expectancy_r']:<+23.2f}R | {m_real_25['expectancy_r']:+.2f}R")

    print("\n[TABLE 2: REAL MT5 ASSET BREAKDOWN (2023 - 2026)]")
    print(f"{'Asset':<10} | {'Trades':<6} | {'Win Rate':<8} | {'Total R':<8} | {'Net P&L':<10} | {'Max DD ($)':<12} | {'PF':<5}")
    print("-" * 75)
    for sym, grp in core_df.groupby('symbol'):
        m = compute_metrics(grp, 'real_r', 'real_pnl')
        print(f"{sym:<10} | {m['trades']:<6d} | {m['win_rate']:>6.1f}% | {m['total_r']:>+7.1f}R | ${m['net_pnl']:>+9,.0f} | ${m['max_dd_dollars']:>10,.0f} | {m['profit_factor']:>4.2f}")

    # SECTION 2: R:R Sensitivity Analysis
    print("\n" + "=" * 85)
    print("SECTION 2: R:R SENSITIVITY TEST (REAL MT5 EXECUTION ON EURUSD & GBPUSD)")
    print("=" * 85)
    for sym in ["EURUSD", "GBPUSD"]:
        df_merged, setups = all_data[sym]
        print(f"\nAsset: {sym}")
        print(f"{'R:R Target':<12} | {'Win Rate':<10} | {'Total R':<10} | {'Net P&L':<12} | {'Max DD':<12} | {'PF':<5}")
        print("-" * 70)
        for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
            res = simulate_real_outcomes(df_merged, setups, rr_ratio=rr)
            m = compute_metrics(res, 'real_r', 'real_pnl')
            print(f"{rr:<12.1f} | {m['win_rate']:>8.1f}% | {m['total_r']:>+8.1f}R | ${m['net_pnl']:>+10,.0f} | {m['max_dd_pct']:>10.1f}% | {m['profit_factor']:>4.2f}")

    # SECTION 3: Extended FX & Commodities Scan
    print("\n" + "=" * 85)
    print("SECTION 3: EXTENDED UNIVERSE SCAN (REAL MT5 1:1.5 RR EXECUTION)")
    print("=" * 85)
    print(f"{'Symbol':<10} | {'Trades':<6} | {'Win Rate':<8} | {'Total R':<8} | {'Net P&L':<10} | {'PF':<5}")
    print("-" * 60)
    for sym in EXTENDED_ASSETS:
        if sym in all_data:
            df_merged, setups = all_data[sym]
            res = simulate_real_outcomes(df_merged, setups, rr_ratio=1.5)
            m = compute_metrics(res, 'real_r', 'real_pnl')
            print(f"{sym:<10} | {m['trades']:<6d} | {m['win_rate']:>6.1f}% | {m['total_r']:>+7.1f}R | ${m['net_pnl']:>+9,.0f} | {m['profit_factor']:>4.2f}")

    # SECTION 4: EMA 233 TREND FILTER IMPACT AUDIT
    print("\n" + "=" * 85)
    print("SECTION 4: EMA 233 TREND FILTER IMPACT (REAL MT5 PRICE-ACTION EXECUTION)")
    print("Rule: Long ONLY if Price > EMA 233 | Short ONLY if Price < EMA 233")
    print("=" * 85)
    
    # Precompute setups once for each mode
    modes = [("Baseline (No 233)", None), ("1H EMA 233", "1H"), ("4H EMA 233", "4H")]
    cached_mode_setups = {}
    for mode_name, mode_val in modes:
        cached_mode_setups[mode_name] = []
        for sym in CORE_ASSETS:
            df_1h = load_data(sym, "1H")
            df_4h = load_data(sym, "4H")
            df_htf = compute_htf_indicators(df_4h)
            df_merged, setups = find_setups(df_1h, df_htf, sym, ema233_mode=mode_val)
            cached_mode_setups[mode_name].append((sym, df_merged, setups))
            
    for rr in [1.0, 1.5, 2.0]:
        print(f"\n--- Performance at Target R:R = {rr:.1f} ---")
        print(f"{'Filter Mode':<18} | {'Trades':<6} | {'Win Rate':<8} | {'Total R':<8} | {'Net P&L':<12} | {'Max DD':<10} | {'PF':<5}")
        print("-" * 75)
        
        for mode_name, _ in modes:
            mode_trades = []
            for sym, df_merged, setups in cached_mode_setups[mode_name]:
                real_t = simulate_real_outcomes(df_merged, setups, rr_ratio=rr)
                mode_trades.append(real_t)
                
            combined = pd.concat(mode_trades, ignore_index=True)
            m = compute_metrics(combined, 'real_r', 'real_pnl')
            print(f"{mode_name:<18} | {m['trades']:<6d} | {m['win_rate']:>6.1f}% | {m['total_r']:>+7.1f}R | ${m['net_pnl']:>+10,.0f} | {m['max_dd_pct']:>8.1f}% | {m['profit_factor']:>4.2f}")

    # Breakdown per symbol with 1H EMA 233 at 1.5 RR
    print("\n--- Asset Breakdown with 1H EMA 233 Filter (at 1:1.5 RR) ---")
    print(f"{'Symbol':<10} | {'Trades':<6} | {'Win Rate':<8} | {'Total R':<8} | {'Net P&L':<12} | {'Max DD':<10} | {'PF':<5}")
    print("-" * 75)
    for sym, df_merged, setups in cached_mode_setups["1H EMA 233"]:
        real_t = simulate_real_outcomes(df_merged, setups, rr_ratio=1.5)
        m = compute_metrics(real_t, 'real_r', 'real_pnl')
        print(f"{sym:<10} | {m['trades']:<6d} | {m['win_rate']:>6.1f}% | {m['total_r']:>+7.1f}R | ${m['net_pnl']:>+10,.0f} | {m['max_dd_pct']:>8.1f}% | {m['profit_factor']:>4.2f}")

if __name__ == "__main__":
    run_comprehensive_study()
