"""
Audit harness: reproduce quant_alpha_basket.py results locally, then quantify
each methodological bias by re-simulating the SAME trade list under corrections.

Run from the repository root:  python quant_review/audit_repro.py
Outputs JSON to quant_review/results/audit_repro.json
"""

# ── portable path bootstrap (added for repo distribution) ──────────────────
import os, sys
from pathlib import Path
PACKAGE_DIR = Path(__file__).resolve().parent   # <repo>/quant_review
REPO_ROOT = PACKAGE_DIR.parent                  # repository root

import os, sys, json, types, importlib
import numpy as np
import pandas as pd

REPO = str(REPO_ROOT)
OUT = str(PACKAGE_DIR / "results")
os.makedirs(OUT, exist_ok=True)

# ── load repo module with patched DATA_DIR ─────────────────────────────────
sys.path.insert(0, REPO)
import quant_alpha_basket as qb
qb.DATA_DIR = os.path.join(REPO, "Blueberry_Data")

BASKET = ["NAS100.p", "BTCUSD.pi", "UK100.p", "AU200.p", "EURUSD.pi",
          "SOLUSD.p", "USDCHF.pi", "USDJPY.pi", "XAGUSD.pi", "JP225.p"]

R_DOLLAR = 10.0   # $ per 1R on the $2.5k account
INIT_EQ = 2500.0

# Blueberry raw-ECN approximations per instrument (round-turn, per side ~half)
# spread typical in points + $7/lot RT commission (3.5 each way). We express as
# R-fraction cost using typical stop distances from the user's README table.
COST_USD_PER_TRADE = {
    # sym: (avg spread pts->USD per 1 trade at the README lot sizes, commission)
    # README lot sizes @ $10 risk: EURUSD 0.05 lots @18 pip stop, etc.
    # spread cost = spread * lot * pointvalue; commission = 3.5USD/lot RT *2 sides? -> 7/lot RT
}
# Simpler, defensible: cost in R computed per trade from instrument spread & lot.

def r_dollar_cost(symbol, risk_usd=10.0):
    """Round-turn cost estimate in R for $10-risk trades at README lot sizes.
    spread+commission ≈ (spread_pts * pt_val * lots) + (7.0 * lots)."""
    table = {  # sym: (spread_pts, point_value_per_lot, lots)
        "EURUSD.pi": (0.2, 10.0, 0.05),   # 0.2 pip spread, $10/pip/lot, 0.05 lots
        "USDJPY.pi": (0.2, 6.7, 0.07),
        "USDCHF.pi": (0.3, 11.0, 0.05),
        "BTCUSD.pi": (15.0, 1.0, 0.02),
        "SOLUSD.p":  (0.02, 1.0, 0.10),
        "NAS100.p":  (1.0, 1.0, 0.03),
        "UK100.p":   (1.0, 1.3, 0.03),
        "AU200.p":   (1.0, 0.67, 0.05),
        "JP225.p":   (8.0, 0.67, 0.02),
        "XAGUSD.pi": (0.02, 50.0, 0.01),
    }
    sp, pv, lots = table[symbol]
    cost = sp * pv * lots + 7.0 * lots
    return cost / risk_usd


def portfolio_metrics(trades, r_col="total_r", order_col="exit_time"):
    t = trades.sort_values(order_col).reset_index(drop=True)
    n = len(t)
    if n == 0:
        return dict(n=0)
    pnl = t[r_col] * R_DOLLAR
    eq = INIT_EQ + pnl.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    gw = pnl[pnl > 0].sum()
    gl = abs(pnl[pnl < 0].sum())
    return {
        "n": n,
        "total_r": float(t[r_col].sum()),
        "net_usd": float(pnl.sum()),
        "pf": float(gw / gl) if gl > 0 else 99.0,
        "max_dd_usd": float(dd.min()),
        "max_dd_pct": float(dd.min() / peak.max() * 100),
        "expectancy_r": float(t[r_col].mean()),
        "ret_pct": float(pnl.sum() / INIT_EQ * 100),
    }


def bucket_stats(t):
    n = len(t)
    return {
        "hit2r_pct": float(t["hit_2r"].mean() * 100),
        "be_pct": float(((t["total_r"] >= 0.0) & (t["total_r"] <= 0.5)).mean() * 100),
        "loss_pct": float((t["total_r"] < 0.0).mean() * 100),
    }


def main():
    # ── 1. Reproduce as-is ─────────────────────────────────────────────────
    per_asset = []
    asset_frames = []
    merged_frames = []
    setups_frames = []
    for sym in BASKET:
        df_1h = qb.load_asset_data(sym, "1H")
        df_4h = qb.load_asset_data(sym, "4H")
        if df_1h is None or df_4h is None:
            print(f"!! missing data for {sym}")
            continue
        df_htf = qb.compute_htf_indicators(df_4h)
        # honest_htf_merge=False: reproduce the README's published baseline
        # (pre-fix merge). The fixed engine defaults to True.
        merged, setups = qb.extract_setups(df_1h, df_htf, sym, honest_htf_merge=False)
        if setups.empty:
            continue
        t_df = qb.simulate_2r_runner(merged, setups)
        per_asset.append({"symbol": sym, "n": len(t_df)})
        asset_frames.append(t_df)
        merged_frames.append(merged)
        setups_frames.append(setups)

    basket = pd.concat(asset_frames, ignore_index=True).sort_values("entry_time").reset_index(drop=True)
    # repo's exit-bar assignment: simulate_2r_runner does not record exit_time; reconstruct
    # We reconstruct exit times by merging on entry_idx+holding_bars per symbol.
    # Simpler: recompute exit time using holding_bars offset on the merged frame.
    exit_times = []
    for sym in BASKET:
        pass  # filled below via lookup

    # Build per-symbol merged time arrays for exit-time reconstruction
    sym_time = {}
    for sym, mg in zip([s for s in BASKET if any(f['symbol'].iloc[0] == s for f in asset_frames)], merged_frames):
        sym_time[sym] = mg["time"].reset_index(drop=True)

    def exit_time_row(row):
        tt = sym_time[row["symbol"]]
        j = int(row["entry_idx"]) + int(row["holding_bars"])
        return tt.iloc[min(j, len(tt) - 1)]

    basket["exit_time"] = basket.apply(exit_time_row, axis=1)

    m_asis = portfolio_metrics(basket, "real_r", "entry_time")
    b_asis = bucket_stats(basket.assign(total_r=basket["real_r"]))

    # ── 2. Corrections, applied on the same trade list ─────────────────────
    # 2a. Costs only
    basket_cost = basket.copy()
    basket_cost["total_r"] = basket["real_r"] - basket["symbol"].map(r_dollar_cost)
    m_cost = portfolio_metrics(basket_cost, "total_r", "entry_time")

    # 2b. Exit-time ordering (equity marked at exit, not entry)
    m_exit_order = portfolio_metrics(basket, "real_r", "exit_time")

    # 2c. Costs + exit ordering
    m_cost_exit = portfolio_metrics(basket_cost, "total_r", "exit_time")

    # ── 3. Uncorrected bucket shares (for README comparison) ──────────────
    res = {
        "per_asset_trades": per_asset,
        "asis": {**m_asis, **b_asis},
        "cost_only": m_cost,
        "exit_order_only": m_exit_order,
        "cost_and_exit_order": m_cost_exit,
        "cost_r_by_symbol": {s: r_dollar_cost(s) for s in BASKET},
    }
    with open(os.path.join(OUT, "audit_repro.json"), "w") as f:
        json.dump(res, f, indent=2)

    print("== AS-IS (repo engine, entry-time ordering) ==")
    for k, v in m_asis.items():
        print(f"  {k:16s} {v}")
    print("  buckets:", b_asis)
    print("\n== + COSTS ==")
    for k, v in m_cost.items():
        print(f"  {k:16s} {v}")
    print("\n== exit-time ordering ==")
    for k, v in m_exit_order.items():
        print(f"  {k:16s} {v}")
    print("\n== costs + exit ordering ==")
    for k, v in m_cost_exit.items():
        print(f"  {k:16s} {v}")

    # Save trade lists for later modules
    basket.to_parquet(os.path.join(OUT, "basket_trades_asis.parquet"), index=False)
    # save merged frames + setups for exit-engine work
    for sym, mg, st in zip([f["symbol"].iloc[0] for f in asset_frames], merged_frames, setups_frames):
        mg.to_parquet(os.path.join(OUT, f"merged_{sym.replace('.', '_')}.parquet"), index=False)
        st.to_parquet(os.path.join(OUT, "setups_" + sym.replace(".", "_") + ".parquet"), index=False)
    print("\nSaved basket trades + merged frames to", OUT)


if __name__ == "__main__":
    main()
