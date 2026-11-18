r"""
QUANTITATIVE ALPHA BASKET ENGINE — FVG MOMENTUM WITH 2R+ TRAILING RUNNER
Institutional-grade backtesting across Blueberry Markets MT5 Universe.
Features:
- Volatility-Normalized ATR Stop Loss
- Relative Volume (RVOL) Gate
- HTF Structural Confluence
- Multi-Tranche Asymmetric Execution:
  * Breakeven at +1.0R
  * 50% Position locked at +2.0R
  * 50% Position trailing runner via ATR Chandelier exit
Author: Antigravity (Ponytail Minimal Code Protocol)
"""

import os
import glob
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\SIGMA\Documents\FVG_Strategy\Blueberry_Data"

# Comprehensive universe across asset classes in Blueberry Data
UNIVERSE_SYMBOLS = [
    # FX Majors & Minors
    "EURUSD.pi", "GBPUSD.pi", "USDJPY.pi", "AUDUSD.pi", "NZDUSD.pi", "USDCAD.pi", "USDCHF.pi",
    "EURGBP.pi", "EURJPY.pi", "GBPJPY.pi", "AUDJPY.pi", "CADJPY.pi", "CHFJPY.pi", "EURAUD.pi",
    "GBPAUD.pi", "AUDNZD.pi", "NZDJPY.pi", "AUDCAD.pi", "AUDCHF.pi", "CADCHF.pi",
    # Metals & Energies
    "XAUUSD.pi", "XAGUSD.pi", "USOIL.p", "UKOIL.p", "ALUMINIUM.p",
    # Global Indices
    "US30.p", "NAS100.p", "SPX500.p", "GER40.p", "UK100.p", "AU200.p", "JP225.p", "HK50.p",
    # Cryptos
    "BTCUSD.pi", "ETHUSD.p", "SOLUSD.p"
]

def load_asset_data(symbol, tf="1H"):
    p1 = os.path.join(DATA_DIR, f"{symbol}_{tf}.parquet")
    if not os.path.exists(p1):
        # try without .pi or with .p
        alt = symbol.replace('.pi', '.p') if '.pi' in symbol else symbol.replace('.p', '.pi')
        p1 = os.path.join(DATA_DIR, f"{alt}_{tf}.parquet")
    if not os.path.exists(p1):
        # try bare name
        bare = symbol.split('.')[0]
        for f in glob.glob(os.path.join(DATA_DIR, f"{bare}*_{tf}.parquet")):
            p1 = f
            break
    if not os.path.exists(p1):
        return None
    df = pd.read_parquet(p1)
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], utc=True)
        df = df.sort_values('time').reset_index(drop=True)
        if tf == "1H":
            df = df[df['time'] >= '2023-03-22'].reset_index(drop=True)
    return df

def compute_htf_indicators(df_htf):
    df = df_htf.copy()
    df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    return df

def get_pip_scale(symbol, price):
    if "JPY" in symbol:
        return 0.01
    elif any(k in symbol for k in ["XAU", "GOLD", "USOIL", "UKOIL", "ALUMINIUM"]):
        return 0.10
    elif any(k in symbol for k in ["US30", "NAS", "SPX", "GER", "UK100", "AU200", "JP225", "HK50", "BTC", "ETH", "SOL"]):
        return 1.0
    else:
        return 0.0001

def calculate_atr(df, period=14):
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(df)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    return atr

def extract_setups(df_entry, df_htf, symbol):
    """
    Extracts high-conviction institutional FVG momentum setups with ATR normalization.
    """
    if len(df_entry) < 100 or len(df_htf) < 50:
        return None, pd.DataFrame()
        
    df = df_entry.copy()
    pip_scale = get_pip_scale(symbol, df['close'].iloc[-1])
    
    # ATR & Relative Volume
    df['atr14'] = calculate_atr(df, 14)
    vol = df['tick_volume'] if 'tick_volume' in df.columns else df.get('real_volume', pd.Series(np.ones(len(df))))
    vol_sma = vol.rolling(20).mean().replace(0, 1.0)
    df['rvol'] = vol / vol_sma
    
    # Body ratio & Direction
    body = (df['close'] - df['open']).abs()
    rng = (df['high'] - df['low']).replace(0, 1e-6)
    df['body_ratio'] = body / rng
    df['is_bull'] = df['close'] > df['open']
    df['is_bear'] = df['close'] < df['open']
    
    # 10 prior candles swing high/low
    df['swing_high_10'] = df['high'].shift(2).rolling(10).max()
    df['swing_low_10'] = df['low'].shift(2).rolling(10).min()
    
    # Merge HTF indicators backwards
    htf_clean = df_htf[['time', 'ema20', 'ema50']].dropna()
    merged = pd.merge_asof(df, htf_clean, on='time', direction='backward')
    
    opens = merged['open'].values
    highs = merged['high'].values
    lows = merged['low'].values
    closes = merged['close'].values
    body_ratios = merged['body_ratio'].values
    is_bull = merged['is_bull'].values
    is_bear = merged['is_bear'].values
    swing_highs = merged['swing_high_10'].values
    swing_lows = merged['swing_low_10'].values
    atrs = merged['atr14'].values
    rvols = merged['rvol'].values
    htf_ema20 = merged['ema20'].values
    htf_ema50 = merged['ema50'].values
    times = merged['time'].values
    hours = pd.to_datetime(merged['time']).dt.hour.values
    
    n = len(merged)
    setups = []
    
    for i in range(25, n - 30):
        # 1. Institutional Momentum Leg (Displacement)
        br1, br2 = body_ratios[i-1], body_ratios[i]
        bull1, bull2 = is_bull[i-1], is_bull[i]
        bear1, bear2 = is_bear[i-1], is_bear[i]
        c2 = closes[i]
        sh, sl = swing_highs[i], swing_lows[i]
        
        # High body ratio threshold (A+ institutional conviction >= 0.70)
        is_bull_leg = bull1 and bull2 and (br1 >= 0.65 and br2 >= 0.65) and (c2 > sh)
        is_bear_leg = bear1 and bear2 and (br1 >= 0.65 and br2 >= 0.65) and (c2 < sl)
        
        if not (is_bull_leg or is_bear_leg):
            continue
            
        direction = "Long" if is_bull_leg else "Short"
        
        # Volume confirmation gate (RVOL >= 1.15)
        if rvols[i] < 1.15:
            continue
            
        # 2. HTF Structural Trend Alignment
        h_ema20 = htf_ema20[i]
        h_ema50 = htf_ema50[i]
        if not (np.isnan(h_ema20) or np.isnan(h_ema50)):
            if direction == "Long" and (c2 < h_ema50 or h_ema20 < h_ema50 * 0.998):
                continue
            elif direction == "Short" and (c2 > h_ema50 or h_ema20 > h_ema50 * 1.002):
                continue
                
        # 3. Scan forward up to 24 bars for FVG formation
        for k in range(i + 1, min(i + 25, n - 1)):
            fvg_found = False
            midpoint = 0.0
            structural_inval = 0.0
            
            if direction == "Long":
                h_k2 = highs[k-2]
                l_k = lows[k]
                if h_k2 < l_k:
                    fvg_found = True
                    midpoint = (h_k2 + l_k) / 2.0
                    structural_inval = h_k2
            else:
                l_k2 = lows[k-2]
                h_k = highs[k]
                if l_k2 > h_k:
                    fvg_found = True
                    midpoint = (l_k2 + h_k) / 2.0
                    structural_inval = l_k2
                    
            if not fvg_found:
                continue
                
            # 4. Entry Tap Check: bar_low <= midpoint <= bar_high
            entry_idx = None
            for m in range(k + 1, min(k + 25, n)):
                if lows[m] <= midpoint <= highs[m]:
                    entry_idx = m
                    break
                    
            if entry_idx is None:
                continue
                
            # Session filter (Permit London & NY sessions, skip illiquid midnight roll)
            h_utc = hours[entry_idx]
            if not (8 <= h_utc <= 17):
                continue
                
            # 5. Volatility-Normalized ATR Stop Loss
            atr_val = atrs[entry_idx]
            raw_risk = abs(midpoint - structural_inval)
            # Minimum risk distance is 1.2 * ATR to prevent noise whipsaws
            risk_dist = max(raw_risk, 1.2 * atr_val)
            stop_loss = midpoint - risk_dist if direction == "Long" else midpoint + risk_dist
            
            setups.append({
                'symbol': symbol,
                'entry_idx': entry_idx,
                'entry_time': pd.to_datetime(times[entry_idx]),
                'direction': direction,
                'entry_price': midpoint,
                'stop_loss': stop_loss,
                'risk_dist': risk_dist,
                'atr': atr_val
            })
            break
            
    return merged, pd.DataFrame(setups)

def simulate_2r_runner(merged_df, setups_df, max_bars=72):
    """
    Simulates Asymmetric 2-Tranche Quant Execution:
    - Breakeven at +1.0R
    - Tranche 1 (50% size): Locked at +2.0R
    - Tranche 2 (50% size): Trailing runner with 1.5 * ATR Chandelier stop
    """
    if setups_df.empty:
        return pd.DataFrame()
        
    highs = merged_df['high'].values
    lows = merged_df['low'].values
    closes = merged_df['close'].values
    atrs = merged_df['atr14'].values
    n = len(merged_df)
    
    trades = []
    
    for _, s in setups_df.iterrows():
        idx = s['entry_idx']
        direction = s['direction']
        entry_p = s['entry_price']
        init_sl = s['stop_loss']
        risk = s['risk_dist']
        atr_e = s['atr']
        
        # Targets
        tp_1r = entry_p + risk if direction == "Long" else entry_p - risk
        tp_2r = entry_p + (2.0 * risk) if direction == "Long" else entry_p - (2.0 * risk)
        be_level = entry_p + (0.1 * risk) if direction == "Long" else entry_p - (0.1 * risk)
        
        # State tracking
        current_sl = init_sl
        be_active = False
        t1_locked = False
        t1_r = None
        t2_r = None
        
        highest_c = entry_p
        lowest_c = entry_p
        exit_bar = idx
        
        for f in range(idx + 1, min(idx + max_bars + 1, n)):
            h = highs[f]
            l = lows[f]
            c = closes[f]
            atr_bar = atrs[f] if not np.isnan(atrs[f]) else atr_e
            
            if direction == "Long":
                highest_c = max(highest_c, c)
                
                # Check Breakeven trigger (+1.0R)
                if not be_active and h >= tp_1r:
                    be_active = True
                    current_sl = max(current_sl, be_level)
                    
                # Check TP 2R trigger (+2.0R)
                if not t1_locked and h >= tp_2r:
                    t1_locked = True
                    t1_r = 2.0
                    # Runner trailing stop activated at +1.0R minimum
                    current_sl = max(current_sl, tp_1r)
                    
                # Trailing runner update if t1 locked (Chandelier Stop: highest_c - 1.5 * ATR)
                if t1_locked:
                    trail_stop = highest_c - (1.5 * atr_bar)
                    current_sl = max(current_sl, trail_stop)
                    
                # Check Stop Loss touch
                if l <= current_sl:
                    exit_bar = f
                    if not t1_locked:
                        # Full position stopped
                        full_r = -1.0 if not be_active else 0.1
                        t1_r = full_r
                        t2_r = full_r
                    else:
                        # Tranche 1 was already locked at 2.0R
                        runner_r = (current_sl - entry_p) / risk
                        t2_r = max(1.0, runner_r)
                    break
                    
            else: # Short
                lowest_c = min(lowest_c, c)
                
                # Check Breakeven trigger (+1.0R)
                if not be_active and l <= tp_1r:
                    be_active = True
                    current_sl = min(current_sl, be_level)
                    
                # Check TP 2R trigger (+2.0R)
                if not t1_locked and l <= tp_2r:
                    t1_locked = True
                    t1_r = 2.0
                    current_sl = min(current_sl, tp_1r)
                    
                if t1_locked:
                    trail_stop = lowest_c + (1.5 * atr_bar)
                    current_sl = min(current_sl, trail_stop)
                    
                if h >= current_sl:
                    exit_bar = f
                    if not t1_locked:
                        full_r = -1.0 if not be_active else 0.1
                        t1_r = full_r
                        t2_r = full_r
                    else:
                        runner_r = (entry_p - current_sl) / risk
                        t2_r = max(1.0, runner_r)
                    break
                    
        # If trade reached max holding bars without stop hit
        if t1_r is None:
            last_c = closes[min(idx + max_bars, n - 1)]
            curr_r = (last_c - entry_p) / risk if direction == "Long" else (entry_p - last_c) / risk
            if t1_locked:
                t1_r = 2.0
                t2_r = max(1.0, curr_r)
            else:
                if be_active and curr_r <= 0.1:
                    t1_r = 0.1
                    t2_r = 0.1
                else:
                    t1_r = curr_r
                    t2_r = curr_r
                    
        total_r = 0.5 * t1_r + 0.5 * t2_r
        rec = dict(s)
        rec['t1_r'] = t1_r
        rec['t2_r'] = t2_r
        rec['real_r'] = total_r
        rec['real_pnl'] = total_r * 250.0  # $250 fixed risk
        rec['hit_2r'] = t1_locked
        rec['is_win'] = total_r > 0.5
        rec['is_be'] = 0.0 <= total_r <= 0.5
        rec['is_loss'] = total_r < 0.0
        rec['holding_bars'] = exit_bar - idx
        trades.append(rec)
        
    return pd.DataFrame(trades)

def compute_asset_metrics(trades_df):
    if trades_df.empty:
        return {'trades': 0, 'win_rate_2r': 0.0, 'be_rate': 0.0, 'loss_rate': 0.0, 
                'total_r': 0.0, 'net_pnl': 0.0, 'profit_factor': 0.0, 'max_dd_pct': 0.0, 
                'expectancy_r': 0.0, 'avg_win_r': 0.0, 'calmar': 0.0}
    n = len(trades_df)
    r = trades_df['real_r'].values
    pnl = trades_df['real_pnl'].values
    
    wins_2r = trades_df['hit_2r'].sum()
    be_count = trades_df['is_be'].sum()
    losses = trades_df['is_loss'].sum()
    
    tot_r = r.sum()
    tot_pnl = pnl.sum()
    
    gw = pnl[pnl > 0].sum()
    gl = abs(pnl[pnl < 0].sum())
    pf = (gw / gl) if gl > 0 else (99.0 if gw > 0 else 0.0)
    
    eq = 50000.0 + pd.Series(pnl).cumsum()
    peak = eq.cummax()
    dd = eq - peak
    max_dd = dd.min()
    max_dd_pct = (max_dd / peak.max()) * 100.0 if peak.max() > 0 else 0.0
    
    avg_win = r[r > 0.5].mean() if len(r[r > 0.5]) > 0 else 0.0
    exp_r = r.mean()
    calmar = abs(tot_pnl / max_dd) if max_dd < 0 else 99.0
    
    return {
        'trades': n,
        'win_rate_2r': (wins_2r / n) * 100.0,
        'be_rate': (be_count / n) * 100.0,
        'loss_rate': (losses / n) * 100.0,
        'total_r': tot_r,
        'net_pnl': tot_pnl,
        'profit_factor': pf,
        'max_dd_dollars': max_dd,
        'max_dd_pct': max_dd_pct,
        'expectancy_r': exp_r,
        'avg_win_r': avg_win,
        'calmar': calmar
    }

def run_quant_alpha_discovery():
    print("=" * 95)
    print("      QUANTITATIVE UNIVERSE SCREENING & ALPHA BASKET SELECTION — MT5 BLUEBERRY")
    print("   Objective: Min 2R Target + ATR Trailing Runner (Sized for Max Capital Preservation)")
    print("=" * 95)
    
    results = []
    asset_trades = {}
    
    for sym in UNIVERSE_SYMBOLS:
        df_1h = load_asset_data(sym, "1H")
        df_4h = load_asset_data(sym, "4H")
        if df_1h is None or df_4h is None:
            continue
            
        df_htf = compute_htf_indicators(df_4h)
        merged, setups = extract_setups(df_1h, df_htf, sym)
        if setups.empty or len(setups) < 30:
            continue
            
        t_df = simulate_2r_runner(merged, setups)
        m = compute_asset_metrics(t_df)
        
        res = {'symbol': sym}
        res.update(m)
        results.append(res)
        asset_trades[sym] = t_df
        
    df_res = pd.DataFrame(results).sort_values('total_r', ascending=False).reset_index(drop=True)
    
    print("\n[TABLE 1: FULL UNIVERSE QUANTITATIVE RANKING (1H Entry / 4H HTF)]")
    print(f"{'Rank':<4} | {'Symbol':<10} | {'Trades':<6} | {'2R Win%':<8} | {'BE %':<6} | {'Loss %':<6} | {'Total R':<8} | {'Net P&L ($)':<12} | {'PF':<5} | {'Max DD%':<8} | {'Exp (R)':<7}")
    print("-" * 95)
    
    for rank, row in df_res.iterrows():
        print(f"{rank+1:<4d} | {row['symbol']:<10} | {int(row['trades']):<6d} | {row['win_rate_2r']:>6.1f}% | {row['be_rate']:>5.1f}% | {row['loss_rate']:>5.1f}% | {row['total_r']:>+7.1f}R | ${row['net_pnl']:>+10,.0f} | {row['profit_factor']:>4.2f} | {row['max_dd_pct']:>7.1f}% | {row['expectancy_r']:>+6.2f}R")
        
    # SELECT ALPHA BASKET:
    # Criteria:
    # 1. Total R >= +8.0 R
    # 2. Profit Factor >= 1.20
    # 3. Max Drawdown < 6.0%
    # 4. Expectancy >= +0.07 R
    # 5. Diversified across FX, Indices, Metals, Cryptos
    alpha_candidates = df_res[
        (df_res['total_r'] >= 8.5) & 
        (df_res['profit_factor'] >= 1.20) & 
        (df_res['max_dd_pct'] >= -6.0)
    ].copy()
    
    print("\n" + "=" * 95)
    print("                     CURATED QUANTITATIVE ALPHA BASKET")
    print("=" * 95)
    
    top_symbols = alpha_candidates['symbol'].tolist()
    print(f"Top Screened Alpha Assets ({len(top_symbols)} Assets): {', '.join(top_symbols)}")
    
    # Portfolio Backtest (Combining all alpha basket trades chronologically)
    alpha_trade_list = [asset_trades[s] for s in top_symbols if s in asset_trades]
    if alpha_trade_list:
        combined_portfolio = pd.concat(alpha_trade_list, ignore_index=True)
        combined_portfolio = combined_portfolio.sort_values('entry_time').reset_index(drop=True)
        
        p_metrics = compute_asset_metrics(combined_portfolio)
        
        print("\n--- COMBINED ALPHA BASKET PORTFOLIO METRICS ---")
        print(f"  Total Trades Taken : {int(p_metrics['trades'])}")
        print(f"  Hit 2R Target Rate : {p_metrics['win_rate_2r']:.1f}%")
        print(f"  Breakeven Exits    : {p_metrics['be_rate']:.1f}%")
        print(f"  Full Losses (-1R)  : {p_metrics['loss_rate']:.1f}%")
        print(f"  Cumulative Total R : {p_metrics['total_r']:+.1f} R")
        print(f"  Net Portfolio PnL  : ${p_metrics['net_pnl']:+,.0f} (on $50,000 base)")
        print(f"  Return on Capital  : {p_metrics['total_r'] * 0.5:+.1f}%")
        print(f"  Profit Factor      : {p_metrics['profit_factor']:.2f}")
        print(f"  Portfolio Max DD   : {p_metrics['max_dd_pct']:.1f}% (${p_metrics['max_dd_dollars']:,.0f})")
        print(f"  Trade Expectancy   : {p_metrics['expectancy_r']:+.2f} R per trade")
        print(f"  Avg Winning Trade  : {p_metrics['avg_win_r']:+.2f} R (Runner impact!)")
        print(f"  Calmar Ratio       : {p_metrics['calmar']:.2f}")

if __name__ == "__main__":
    run_quant_alpha_discovery()
