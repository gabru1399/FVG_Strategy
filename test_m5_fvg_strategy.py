r"""
Exact 5-Minute (M5) FVG Momentum Strategy Backtester on MT5 Blueberry Data
Directly tests FVG_Strategy.pdf (Teghjot Singh Bindra, 2025) on real 5-minute ticks.
Author: Antigravity (Ponytail Minimal Code Protocol)
"""

import os
import hashlib
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\SIGMA\Documents\FVG_Strategy\Blueberry_Data"
CORE_SYMBOLS = ["EURUSD.pi", "GBPUSD.pi", "USDJPY.pi", "XAUUSD.pi"]

def load_data(symbol):
    f_m5 = os.path.join(DATA_DIR, f"{symbol}_5M.parquet")
    f_1h = os.path.join(DATA_DIR, f"{symbol}_1H.parquet")
    if not (os.path.exists(f_m5) and os.path.exists(f_1h)):
        return None, None
    df_m5 = pd.read_parquet(f_m5).sort_values('time').reset_index(drop=True)
    df_1h = pd.read_parquet(f_1h).sort_values('time').reset_index(drop=True)
    return df_m5, df_1h

def compute_htf_1h(df_1h):
    df = df_1h.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema233_1h'] = df['close'].ewm(span=233, adjust=False).mean()
    
    # 30-period rolling Point of Control (POC)
    pocs = [np.nan] * len(df)
    closes = df['close'].values
    for i in range(30, len(df)):
        window = closes[i-30:i]
        counts, bin_edges = np.histogram(window, bins=15)
        max_idx = np.argmax(counts)
        pocs[i] = (bin_edges[max_idx] + bin_edges[max_idx+1]) / 2.0
    df['poc30'] = pocs
    return df

def get_pip_size(symbol):
    if "JPY" in symbol:
        return 0.01
    elif "XAU" in symbol:
        return 0.10
    else:
        return 0.0001

def find_m5_setups(df_m5, df_htf_1h, symbol, ema233_mode=None):
    pip_size = get_pip_size(symbol)
    df = df_m5.copy()
    
    # 5M Indicators
    body = (df['close'] - df['open']).abs()
    rng = (df['high'] - df['low']).replace(0, 1e-6)
    df['body_ratio'] = body / rng
    df['is_bull'] = df['close'] > df['open']
    df['is_bear'] = df['close'] < df['open']
    df['ema233_5m'] = df['close'].ewm(span=233, adjust=False).mean()
    
    # 10 prior candles swing high/low (50 mins)
    df['swing_high_10'] = df['high'].shift(2).rolling(10).max()
    df['swing_low_10'] = df['low'].shift(2).rolling(10).min()
    
    # Merge 1H HTF indicators backwards
    htf_clean = df_htf_1h[['time', 'ema20', 'ema50', 'poc30', 'ema233_1h']].dropna()
    merged = pd.merge_asof(df, htf_clean, on='time', direction='backward')
    
    # Extract numpy arrays for ultra-fast indexing
    opens = merged['open'].values
    highs = merged['high'].values
    lows = merged['low'].values
    closes = merged['close'].values
    body_ratios = merged['body_ratio'].values
    is_bull = merged['is_bull'].values
    is_bear = merged['is_bear'].values
    swing_highs = merged['swing_high_10'].values
    swing_lows = merged['swing_low_10'].values
    ema20s = merged['ema20'].values
    ema50s = merged['ema50'].values
    poc30s = merged['poc30'].values
    ema233_5m_arr = merged['ema233_5m'].values
    ema233_1h_arr = merged['ema233_1h'].values
    times = merged['time'].values
    hours = pd.to_datetime(merged['time']).dt.hour.values
    
    n = len(merged)
    setups = []
    
    for i in range(12, n - 25):
        br1, br2 = body_ratios[i-1], body_ratios[i]
        bull1, bull2 = is_bull[i-1], is_bull[i]
        bear1, bear2 = is_bear[i-1], is_bear[i]
        c2 = closes[i]
        sh, sl = swing_highs[i], swing_lows[i]
        
        is_bull_leg = bull1 and bull2 and (c2 > sh)
        is_bear_leg = bear1 and bear2 and (c2 < sl)
        
        if not (is_bull_leg or is_bear_leg):
            continue
            
        direction = "Long" if is_bull_leg else "Short"
        grade_candidate = "A+" if (br1 >= 0.70 and br2 >= 0.70) else ("A" if (br1 >= 0.52 and br2 >= 0.52) else None)
        if grade_candidate is None:
            continue
            
        # Scan forward 24 bars (exactly 2 hours of forward M5 price action per paper)
        for k in range(i + 1, min(i + 25, n - 1)):
            fvg_found = False
            midpoint = 0.0
            fvg_sl = 0.0
            
            if direction == "Long":
                h_k2 = highs[k-2]
                l_k = lows[k]
                if h_k2 < l_k:
                    fvg_found = True
                    midpoint = (h_k2 + l_k) / 2.0
                    fvg_sl = h_k2
            else:
                l_k2 = lows[k-2]
                h_k = highs[k]
                if l_k2 > h_k:
                    fvg_found = True
                    midpoint = (l_k2 + h_k) / 2.0
                    fvg_sl = l_k2
                    
            if not fvg_found:
                continue
                
            # Check for entry tap: bar_low <= midpoint <= bar_high
            entry_idx = None
            for m in range(k + 1, min(k + 25, n)):
                if lows[m] <= midpoint <= highs[m]:
                    entry_idx = m
                    break
                    
            if entry_idx is None:
                continue
                
            # Session filter (UTC)
            h_utc = hours[entry_idx]
            session = None
            wr_bonus = 0.0
            if 8 <= h_utc < 11:
                session = "London"
                wr_bonus = 0.05
            elif 13 <= h_utc < 15:
                session = "Overlap"
                wr_bonus = 0.06
            elif 15 <= h_utc < 16:
                session = "New York"
                wr_bonus = 0.02
            else:
                continue
                
            # Optional EMA 233 Trend Filter
            if ema233_mode == "5M":
                e233 = ema233_5m_arr[entry_idx]
                if direction == "Long" and midpoint <= e233:
                    continue
                elif direction == "Short" and midpoint >= e233:
                    continue
            elif ema233_mode == "1H":
                e233 = ema233_1h_arr[entry_idx]
                if direction == "Long" and midpoint <= e233:
                    continue
                elif direction == "Short" and midpoint >= e233:
                    continue
                    
            # HTF Confluence (within 12 pips of EMA20, EMA50, or POC)
            ema20 = ema20s[entry_idx]
            ema50 = ema50s[entry_idx]
            poc = poc30s[entry_idx]
            
            htf_aligned = False
            prox = 12.0 * pip_size
            for ref in [ema20, ema50, poc]:
                if not np.isnan(ref) and abs(midpoint - ref) <= prox:
                    htf_aligned = True
                    break
                    
            if not np.isnan(ema50):
                if direction == "Long" and midpoint < ema50 - prox:
                    htf_aligned = False
                elif direction == "Short" and midpoint > ema50 + prox:
                    htf_aligned = False
                    
            if grade_candidate == "A+" and htf_aligned:
                final_grade = "A+"
            elif grade_candidate in ["A", "A+"]:
                final_grade = "A"
            else:
                continue
                
            setups.append({
                'symbol': symbol,
                'entry_idx': entry_idx,
                'entry_time': pd.to_datetime(times[entry_idx]),
                'direction': direction,
                'grade': final_grade,
                'session': session,
                'htf_aligned': htf_aligned,
                'entry_price': midpoint,
                'structural_sl': fvg_sl,
                'pip_sl_9': midpoint - (9 * pip_size) if direction == "Long" else midpoint + (9 * pip_size),
                'pip_size': pip_size,
                'wr_bonus': wr_bonus
            })
            break
            
    return merged, pd.DataFrame(setups)

def simulate_m5_trades(merged_df, setups_df, mode="paper_pips", rr=1.0, max_holding_bars=96):
    """
    Simulates trades bar-by-bar on 5-minute candles.
    mode='paper_pips': Exactly replicates paper (SL=9 pips, TP1=9 pips).
    mode='structural': Uses FVG structural boundary with specified R:R.
    """
    if setups_df.empty:
        return setups_df
        
    highs = merged_df['high'].values
    lows = merged_df['low'].values
    closes = merged_df['close'].values
    n = len(merged_df)
    
    trades = []
    for _, s in setups_df.iterrows():
        idx = s['entry_idx']
        direction = s['direction']
        entry_p = s['entry_price']
        pip = s['pip_size']
        
        if mode == "paper_pips":
            # Paper parameters: 9 pips SL, 9 pips TP1 (or TP2=20/35 pips)
            sl_dist = 9.0 * pip
            sl = entry_p - sl_dist if direction == "Long" else entry_p + sl_dist
            tp_dist = (9.0 * rr) * pip
            tp = entry_p + tp_dist if direction == "Long" else entry_p - tp_dist
        else: # structural
            sl = s['structural_sl']
            sl_dist = abs(entry_p - sl)
            if sl_dist < 5.0 * pip:
                sl_dist = 5.0 * pip
                sl = entry_p - sl_dist if direction == "Long" else entry_p + sl_dist
            tp = entry_p + (sl_dist * rr) if direction == "Long" else entry_p - (sl_dist * rr)
            
        result = None
        for f in range(idx + 1, min(idx + max_holding_bars + 1, n)):
            h = highs[f]
            l = lows[f]
            
            if direction == "Long":
                hit_sl = l <= sl
                hit_tp = h >= tp
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
                hit_sl = h >= sl
                hit_tp = l <= tp
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
            last_c = closes[min(idx + max_holding_bars, n - 1)]
            r_ret = (last_c - entry_p) / sl_dist if direction == "Long" else (entry_p - last_c) / sl_dist
            result = "Win" if r_ret > 0 else "Loss"
            r_cap = r_ret
        else:
            r_cap = rr if result == "Win" else -1.0
            
        rec = dict(s)
        rec['real_result'] = result
        rec['real_r'] = r_cap
        rec['real_pnl'] = r_cap * 250.0
        trades.append(rec)
        
    return pd.DataFrame(trades)

def simulate_paper_hash(trades_df):
    if trades_df.empty:
        return trades_df
    out = []
    for _, t in trades_df.iterrows():
        base_wr = 0.76 if t['grade'] == "A+" else 0.64
        bonus = t.get('wr_bonus', 0.0) + (0.04 if t['htf_aligned'] else 0.0)
        p = base_wr + bonus
        h_str = f"{t['symbol']}_{t['entry_time']}_{t['direction']}_{t['grade']}"
        h_val = int(hashlib.md5(h_str.encode()).hexdigest(), 16) % 10000 / 10000.0
        win = h_val < p
        r = 1.0 if win else -1.0
        rec = dict(t)
        rec['hash_r'] = r
        rec['hash_pnl'] = r * 250.0
        out.append(rec)
    return pd.DataFrame(out)

def compute_metrics(df_trades, r_col, pnl_col):
    if df_trades.empty:
        return {'trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0.0, 'total_r': 0.0, 
                'net_pnl': 0.0, 'return_pct': 0.0, 'profit_factor': 0.0, 'max_dd_dollars': 0.0, 
                'max_dd_pct': 0.0}
    n = len(df_trades)
    wins = (df_trades[r_col] > 0).sum()
    losses = (df_trades[r_col] <= 0).sum()
    wr = (wins / n) * 100.0 if n > 0 else 0.0
    tot_r = df_trades[r_col].sum()
    tot_pnl = df_trades[pnl_col].sum()
    
    gw = df_trades[df_trades[pnl_col] > 0][pnl_col].sum()
    gl = abs(df_trades[df_trades[pnl_col] < 0][pnl_col].sum())
    pf = (gw / gl) if gl > 0 else (99.0 if gw > 0 else 0.0)
    
    eq = 50000.0 + df_trades[pnl_col].cumsum()
    peak = eq.cummax()
    dd = eq - peak
    max_dd = dd.min()
    max_dd_pct = (max_dd / peak.max()) * 100.0 if peak.max() > 0 else 0.0
    
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
        'max_dd_pct': max_dd_pct
    }

def run_study():
    print("=" * 85)
    print("      EXACT 5-MINUTE (M5) FVG STRATEGY TEST ON BLUEBERRY MARKETS MT5 DATA")
    print("               Paper Spec: Teghjot Singh Bindra (2025)")
    print("=" * 85)
    
    m5_cache = {}
    for sym in CORE_SYMBOLS:
        df_m5, df_1h = load_data(sym)
        if df_m5 is not None and df_1h is not None:
            df_htf = compute_htf_1h(df_1h)
            m5_cache[sym] = (df_m5, df_htf)
            print(f"Loaded {sym:10s} | M5 bars: {len(df_m5):6d} ({df_m5['time'].iloc[0].strftime('%Y-%m-%d')} to {df_m5['time'].iloc[-1].strftime('%Y-%m-%d')})")
            
    # PART 1: Paper's Exact 9 Pip SL / 9 Pip TP1 (Trail 1:1) Setup
    print("\n" + "=" * 85)
    print("PART 1: REAL 5M PRICE ACTION — EXACT PAPER PARAMETERS (9 PIP SL / 9 PIP TP1)")
    print("=" * 85)
    
    all_paper_pips = []
    for sym in CORE_SYMBOLS:
        df_m5, df_htf = m5_cache[sym]
        merged, setups = find_m5_setups(df_m5, df_htf, sym)
        real_t = simulate_m5_trades(merged, setups, mode="paper_pips", rr=1.0)
        full_t = simulate_paper_hash(real_t)
        all_paper_pips.append(full_t)
        
    df_pips = pd.concat(all_paper_pips, ignore_index=True)
    m_hash = compute_metrics(df_pips, 'hash_r', 'hash_pnl')
    m_real = compute_metrics(df_pips, 'real_r', 'real_pnl')
    
    print("\n[TABLE 1: 5M HEAD-TO-HEAD AUDIT (PAPER HASH SIMULATION VS REAL MT5 TICKS)]")
    print(f"{'Metric':<25} | {'Paper Theoretical Model':<26} | {'Real MT5 5M Price Action'}")
    print("-" * 80)
    print(f"{'Total Trades':<25} | {m_hash['trades']:<26d} | {m_real['trades']}")
    print(f"{'Win Rate':<25} | {m_hash['win_rate']:<25.1f}% | {m_real['win_rate']:.1f}%")
    print(f"{'Total R Captured':<25} | {m_hash['total_r']:<+25.1f}R | {m_real['total_r']:+.1f}R")
    print(f"{'Net P&L ($50k Base)':<25} | ${m_hash['net_pnl']:<+25,.0f} | ${m_real['net_pnl']:+,.0f}")
    print(f"{'Return on Capital':<25} | {m_hash['return_pct']:<+25.1f}% | {m_real['return_pct']:+.1f}%")
    print(f"{'Max Drawdown':<25} | {m_hash['max_dd_pct']:<25.1f}% | {m_real['max_dd_pct']:.1f}% (${m_real['max_dd_dollars']:,.0f})")
    print(f"{'Profit Factor':<25} | {m_hash['profit_factor']:<26.2f} | {m_real['profit_factor']:.2f}")

    print("\n[TABLE 2: REAL 5M EXECUTION BY ASSET (9 PIP SL / 9 PIP TP1)]")
    print(f"{'Asset':<12} | {'Trades':<6} | {'Win Rate':<8} | {'Total R':<8} | {'Net P&L':<12} | {'Max DD':<10} | {'PF':<5}")
    print("-" * 75)
    for sym, grp in df_pips.groupby('symbol'):
        m = compute_metrics(grp, 'real_r', 'real_pnl')
        print(f"{sym:<12} | {m['trades']:<6d} | {m['win_rate']:>6.1f}% | {m['total_r']:>+7.1f}R | ${m['net_pnl']:>+10,.0f} | {m['max_dd_pct']:>8.1f}% | {m['profit_factor']:>4.2f}")

    # PART 2: EMA 233 Filter on 5M Price Action
    print("\n" + "=" * 85)
    print("PART 2: EMA 233 FILTER ON REAL 5M PRICE ACTION")
    print("Rule: Long ONLY if Price > EMA 233 | Short ONLY if Price < EMA 233")
    print("=" * 85)
    
    for mode_name, mode_val in [("Baseline (No 233)", None), ("5M EMA 233", "5M"), ("1H EMA 233", "1H")]:
        trades_list = []
        for sym in CORE_SYMBOLS:
            df_m5, df_htf = m5_cache[sym]
            merged, setups = find_m5_setups(df_m5, df_htf, sym, ema233_mode=mode_val)
            real_t = simulate_m5_trades(merged, setups, mode="paper_pips", rr=1.0)
            trades_list.append(real_t)
        comb = pd.concat(trades_list, ignore_index=True)
        m = compute_metrics(comb, 'real_r', 'real_pnl')
        print(f"{mode_name:<20} | Trades: {m['trades']:<5d} | WR: {m['win_rate']:>5.1f}% | Total R: {m['total_r']:>+7.1f}R | Net P&L: ${m['net_pnl']:>+9,.0f} | Max DD: {m['max_dd_pct']:>6.1f}% | PF: {m['profit_factor']:>4.2f}")

    # PART 3: Asymmetric Payoff on 5M (1:1.5 and 1:2.0 R:R)
    print("\n" + "=" * 85)
    print("PART 3: REAL 5M EXECUTION WITH ASYMMETRIC R:R (STRUCTURAL & EXTENSION EXITS)")
    print("=" * 85)
    for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
        t_list = []
        for sym in ["EURUSD.pi", "GBPUSD.pi"]:
            df_m5, df_htf = m5_cache[sym]
            merged, setups = find_m5_setups(df_m5, df_htf, sym)
            real_t = simulate_m5_trades(merged, setups, mode="paper_pips", rr=rr)
            t_list.append(real_t)
        comb = pd.concat(t_list, ignore_index=True)
        m = compute_metrics(comb, 'real_r', 'real_pnl')
        print(f"EUR/GBP 5M Target R:R = {rr:.1f} | Trades: {m['trades']:<5d} | WR: {m['win_rate']:>5.1f}% | Total R: {m['total_r']:>+7.1f}R | Net P&L: ${m['net_pnl']:>+9,.0f} | Max DD: {m['max_dd_pct']:>6.1f}% | PF: {m['profit_factor']:>4.2f}")

if __name__ == "__main__":
    run_study()
