r"""
TRI-PAPER QUANTITATIVE SYNTHESIS AUDIT
Testing and combining:
1. SSRN-6725880 (Teghjot Singh Bindra): Multi-Asset Momentum FVG
2. SSRN-5002058 (Quantpedia / Beluska & Vojtko): Cross-Sectional Currency Mean Reversion
3. SSRN-6858299 (Reid Fintak): Regime-Conditioned Trend Following
Tested on local Blueberry Markets MT5 Data.
Author: Antigravity (Ponytail Minimal Code Protocol)
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = r"C:\Users\SIGMA\Documents\FVG_Strategy\Blueberry_Data"

FX_PAIRS = {
    'EUR': 'EURUSD.pi_1H.parquet',
    'GBP': 'GBPUSD.pi_1H.parquet',
    'AUD': 'AUDUSD.pi_1H.parquet',
    'JPY': 'USDJPY.pi_1H.parquet',  # Note: JPY inverted (USD/JPY)
    'CAD': 'USDCAD.pi_1H.parquet',  # CAD inverted (USD/CAD)
    'CHF': 'USDCHF.pi_1H.parquet'   # CHF inverted (USD/CHF)
}

def load_clean_data(filename):
    p = os.path.join(DATA_DIR, filename)
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    df = df[df['time'] >= '2023-03-22'].sort_values('time').reset_index(drop=True)
    return df

def build_cross_sectional_basket():
    """
    Builds the Quantpedia (Paper 2) Cross-Sectional Currency Basket:
    Computes normalized cumulative returns of each currency vs the basket average.
    """
    dfs = {}
    for c, fn in FX_PAIRS.items():
        df = load_clean_data(fn)
        if df is not None:
            # For JPY, CAD, CHF, invert price to represent Currency/USD
            if c in ['JPY', 'CAD', 'CHF']:
                df['close_norm'] = 1.0 / df['close']
            else:
                df['close_norm'] = df['close']
            df['cum_ret'] = df['close_norm'] / df['close_norm'].iloc[0]
            dfs[c] = df.set_index('time')['cum_ret']
            
    panel = pd.DataFrame(dfs).dropna()
    # Basket average anchor
    panel['basket_mean'] = panel.mean(axis=1)
    
    # Deviations from mean (Paper 2)
    devs = pd.DataFrame(index=panel.index)
    for c in FX_PAIRS.keys():
        devs[c] = (panel[c] - panel['basket_mean']) / panel['basket_mean']
        
    return panel, devs

def evaluate_paper2_standalone():
    """
    Evaluates Paper 2 (Cross-Sectional Mean Reversion alone) with Linear vs Exponential Sizing.
    """
    panel, devs = build_cross_sectional_basket()
    returns = panel.pct_change().dropna()
    
    # Linear Strategy (Weights proportional to -deviation)
    w_lin = -devs.shift(1).dropna()
    w_lin = w_lin.div(w_lin.abs().sum(axis=1).replace(0, 1.0), axis=0) # Normalize to 100% exposure
    lin_ret = (w_lin * returns).sum(axis=1)
    
    # Exponential Strategy (Weights proportional to -sign(dev) * (dev^2))
    w_exp = -np.sign(devs.shift(1).dropna()) * (devs.shift(1).dropna() ** 2)
    w_exp = w_exp.div(w_exp.abs().sum(axis=1).replace(0, 1.0), axis=0)
    exp_ret = (w_exp * returns).sum(axis=1)
    
    # Performance metrics
    def stats(s, name):
        ann_ret = s.mean() * 24 * 252 * 100.0
        ann_vol = s.std() * np.sqrt(24 * 252) * 100.0
        sharpe = (ann_ret / ann_vol) if ann_vol > 0 else 0.0
        cum = (1.0 + s).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min() * 100.0
        return {'Strategy': name, 'Ann Return %': ann_ret, 'Ann Vol %': ann_vol, 'Sharpe': sharpe, 'Max DD %': max_dd}
        
    return pd.DataFrame([stats(lin_ret, 'Paper 2 Linear Sizing'), stats(exp_ret, 'Paper 2 Exponential Sizing')])

def run_hybrid_synthesis():
    """
    Combines:
    1. Paper 1: FVG Momentum Displacement & Midpoint Entry
    2. Paper 2: Cross-Sectional Currency Basket Deviation (Macro Anchor)
    3. Paper 3: Valuation / Extension Regime Gate (Avoid buying overextended tops)
    4. Our Quant Asymmetric 2R + Trailing Runner Execution
    """
    import quant_alpha_basket as qb
    
    panel, devs = build_cross_sectional_basket()
    
    hybrid_results = []
    baseline_results = []
    
    for sym_c, fn in [('EUR', 'EURUSD.pi_1H.parquet'), ('GBP', 'GBPUSD.pi_1H.parquet'), ('AUD', 'AUDUSD.pi_1H.parquet'), ('JPY', 'USDJPY.pi_1H.parquet')]:
        df_1h = qb.load_asset_data(fn.replace('_1H.parquet', ''), "1H")
        df_4h = qb.load_asset_data(fn.replace('_1H.parquet', ''), "4H")
        if df_1h is None or df_4h is None:
            continue
            
        df_htf = qb.compute_htf_indicators(df_4h)
        merged, setups = qb.extract_setups(df_1h, df_htf, fn.replace('_1H.parquet', ''))
        if setups.empty:
            continue
            
        # 1. Baseline Performance (Paper 1 + 2R Runner)
        t_base = qb.simulate_2r_runner(merged, setups)
        m_base = qb.compute_asset_metrics(t_base)
        baseline_results.append({'Asset': sym_c, 'Type': 'Baseline (FVG alone)', **m_base})
        
        # 2. Add Paper 2 Cross-Sectional Filter + Paper 3 Regime Gate
        # Paper 2: If Currency is undervalued in basket (dev < 0), take Longs; if overvalued (dev > 0), take Shorts.
        # Paper 3: Compute 200 SMA extension on 1H: if distance > +2.0 ATR (overextended top), skip Long breakout.
        merged['sma200'] = merged['close'].rolling(200).mean()
        merged['dist_sma200'] = (merged['close'] - merged['sma200']) / merged['atr14'].replace(0, 1e-4)
        
        # Align cross-sectional deviations
        dev_series = devs[sym_c] if sym_c in devs.columns else None
        
        filtered_setups = []
        for _, s in setups.iterrows():
            idx = s['entry_idx']
            direction = s['direction']
            t = s['entry_time']
            
            # Paper 3 Regime Check: avoid buying when price is severely overextended (+2.5 ATR above 200 SMA)
            dist_sma = merged.at[idx, 'dist_sma200']
            if pd.notna(dist_sma):
                if direction == "Long" and dist_sma > 2.5:
                    continue # Skip overbought momentum whipsaw
                elif direction == "Short" and dist_sma < -2.5:
                    continue # Skip oversold momentum whipsaw
                    
            # Paper 2 Cross-Sectional Alignment
            # For JPY (USDJPY): USD is base, JPY is quote.
            # Long USDJPY means Long USD / Short JPY.
            if dev_series is not None and t in dev_series.index:
                curr_dev = dev_series.loc[t]
                if sym_c == 'JPY':
                    # Long USDJPY means Short JPY -> requires JPY to be overvalued (dev > -0.01)
                    if direction == "Long" and curr_dev < -0.02:
                        continue
                    elif direction == "Short" and curr_dev > 0.02:
                        continue
                else:
                    # Direct currency (EUR, GBP, AUD vs USD)
                    # Long means Long Currency -> Currency should not be overextended (dev < 0.03)
                    if direction == "Long" and curr_dev > 0.03:
                        continue
                    elif direction == "Short" and curr_dev < -0.03:
                        continue
                        
            filtered_setups.append(s)
            
        df_filt_setups = pd.DataFrame(filtered_setups)
        t_hybrid = qb.simulate_2r_runner(merged, df_filt_setups)
        m_hybrid = qb.compute_asset_metrics(t_hybrid)
        hybrid_results.append({'Asset': sym_c, 'Type': 'Tri-Paper Hybrid Synthesized', **m_hybrid})
        
    return pd.DataFrame(baseline_results), pd.DataFrame(hybrid_results)

def main():
    print("=" * 95)
    print("          RESEARCH PAPERS MULTI-STRATEGY COMPARATIVE AUDIT & SYNTHESIS")
    print("=" * 95)
    
    # 1. Standalone Evaluation of Paper 2 (Quantpedia Mean Reversion)
    print("\n[PART 1: STANDALONE EVALUATION OF SSRN-5002058 (QUANTPEDIA MEAN REVERSION)]")
    p2_df = evaluate_paper2_standalone()
    print(p2_df.to_string(index=False))
    
    # 2. Hybrid Synthesis Evaluation (Paper 1 + Paper 2 + Paper 3 + Quant Runner)
    print("\n" + "=" * 95)
    print("[PART 2: HEAD-TO-HEAD COMPARISON: BASELINE FVG VS TRI-PAPER SYNTHESIZED HYBRID]")
    print("=" * 95)
    df_base, df_hyb = run_hybrid_synthesis()
    
    merged = pd.concat([df_base, df_hyb]).sort_values(['Asset', 'Type']).reset_index(drop=True)
    cols = ['Asset', 'Type', 'trades', 'win_rate_2r', 'be_rate', 'loss_rate', 'total_r', 'profit_factor', 'max_dd_pct', 'expectancy_r']
    print(merged[cols].to_string(index=False))
    
    print("\n--- AGGREGATE BASKET PERFORMANCE SUMMARY ---")
    b_tot_r = df_base['total_r'].sum()
    h_tot_r = df_hyb['total_r'].sum()
    b_avg_pf = df_base['profit_factor'].mean()
    h_avg_pf = df_hyb['profit_factor'].mean()
    b_max_dd = df_base['max_dd_pct'].min()
    h_max_dd = df_hyb['max_dd_pct'].min()
    
    print(f"  Baseline Portfolio Total R : {b_tot_r:+.1f} R | Avg PF: {b_avg_pf:.2f} | Worst Asset DD: {b_max_dd:.1f}%")
    print(f"  Synthesized Hybrid Total R : {h_tot_r:+.1f} R | Avg PF: {h_avg_pf:.2f} | Worst Asset DD: {h_max_dd:.1f}%")

if __name__ == "__main__":
    main()
