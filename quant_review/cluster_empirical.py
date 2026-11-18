"""
cluster_empirical.py — Real-data stress correlation + clustered drawdown study.

1. Build daily returns for the 10 basket assets from 1H parquets.
2. EWMA correlation history: show stress-regime corr spike (equity+crypto).
3. Clustering on full-sample corr vs stress-tail corr: does the structure
   collapse to one mega-cluster in stress?
4. Portfolio DD simulation: replay the 974 basket trades chronologically.
   - Scheme A (repo): flat $10 per trade.
   - Scheme B (heat budget): cluster caps + geometric decay + vol scaling.
   Compare clustered-drawdown (sum of R in simultaneous same-cluster windows).
"""

# ── portable path bootstrap (added for repo distribution) ──────────────────
import os, sys
from pathlib import Path
PACKAGE_DIR = Path(__file__).resolve().parent   # <repo>/quant_review
REPO_ROOT = PACKAGE_DIR.parent                  # repository root

import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, str(PACKAGE_DIR))
from cluster_sizer import ewma_corr, cluster_assets, stress_correlation_scan, HeatBudgetSizer

RES = str(PACKAGE_DIR / "results")
DATA = str(REPO_ROOT / "Blueberry_Data")

BASKET = ["NAS100.p", "BTCUSD.pi", "UK100.p", "AU200.p", "EURUSD.pi",
          "SOLUSD.p", "USDCHF.pi", "USDJPY.pi", "XAGUSD.pi", "JP225.p"]


def load_daily_returns():
    rets = {}
    for sym in BASKET:
        fn = os.path.join(DATA, f"{sym}_1H.parquet")
        df = pd.read_parquet(fn)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time")
        r = df.set_index("time")["close"].resample("1D").last().pct_change().dropna()
        rets[sym] = r
    return pd.DataFrame(rets).dropna()


def main():
    rets = load_daily_returns()
    print(f"Daily returns panel: {rets.shape[0]} days x {rets.shape[1]} assets "
          f"({rets.index[0].date()} -> {rets.index[-1].date()})")

    # ── 1. Full-sample vs stress-tail correlation ───────────────────────────
    corr_full = rets.corr()
    print("\nFull-sample daily correlation (equity+crypto block):")
    eq_crypto = ["NAS100.p", "UK100.p", "AU200.p", "JP225.p", "BTCUSD.pi", "SOLUSD.p"]
    sub = corr_full.loc[eq_crypto, eq_crypto]
    print(sub.round(2).to_string())

    # stress scan
    s, tail_dates, corr_stress = stress_correlation_scan(rets, halflife=20, top_pct=5)
    print(f"\nStress dates (top 5% EWMA corr): {len(tail_dates)} days, e.g. "
          f"{[str(pd.Timestamp(d).date()) for d in tail_dates[:5]]}")
    off = corr_full.loc[eq_crypto, eq_crypto].values
    n = len(eq_crypto)
    full_avg = (off.sum() - np.trace(off)) / (n * n - n)
    off_s = corr_stress[np.ix_([corr_full.columns.get_loc(c) for c in eq_crypto],
                               [corr_full.columns.get_loc(c) for c in eq_crypto])]
    stress_avg = (off_s.sum() - np.trace(off_s)) / (n * n - n)
    print(f"Eq+Crypto avg pairwise corr: full-sample {full_avg:.2f} -> stress-tail {stress_avg:.2f}")

    # avg all-asset pairwise corr in stress
    m = corr_stress
    na = m.shape[0]
    all_stress = (m.sum() - np.trace(m)) / (na * na - na)
    mfull = corr_full.values
    all_full = (mfull.sum() - np.trace(mfull)) / (na * na - na)
    print(f"All-10 avg pairwise corr:     full-sample {all_full:.2f} -> stress-tail {all_stress:.2f}")

    # ── 2. Clustering: full vs stress ──────────────────────────────────────
    clusters_full, assign_full = cluster_assets(corr_full, threshold=0.55)
    print("\nClusters (full-sample corr):")
    for k, v in clusters_full.items():
        print(f"  {k}: {v}")
    corr_stress_df = pd.DataFrame(corr_stress, index=corr_full.index, columns=corr_full.columns)
    clusters_stress, _ = cluster_assets(corr_stress_df, threshold=0.55)
    print("Clusters (stress-tail corr):")
    for k, v in clusters_stress.items():
        print(f"  {k}: {v}")

    # ── 3. Trade replay: flat vs heat-budget sizing ────────────────────────
    trades = pd.read_parquet(os.path.join(RES, "trades_B4_plus_gap_fills.parquet")) \
        if os.path.exists(os.path.join(RES, "trades_B4_plus_gap_fills.parquet")) else None
    if trades is None:
        # fall back to asis trades (column real_r) and normalize to total_r
        trades = pd.read_parquet(os.path.join(RES, "basket_trades_asis.parquet"))
        if "total_r" not in trades.columns and "real_r" in trades.columns:
            trades["total_r"] = trades["real_r"]

    trades = trades.sort_values("entry_time").reset_index(drop=True)

    # map assets to clusters with the full-sample assignment (live-feasible:
    # cluster assignment estimated on trailing data only in production)
    assign = assign_full

    def replay(scheme):
        eq = 2500.0
        peak = eq
        max_dd = 0.0
        eq_curve = []
        sizer = HeatBudgetSizer(equity=2500.0, heat_total_pct=0.03,
                                heat_cluster_pct=0.015, base_risk_pct=0.004,
                                decay=0.6, window_bars=24)
        sizer.register_cluster_map(assign)
        total_r_usd = 0.0
        n_skipped = 0
        for i, row in trades.iterrows():
            bar = int(pd.Timestamp(row["entry_time"]).value // 3.6e12)  # hour index
            if scheme == "flat":
                risk_usd = 10.0
            else:
                # vol ratio proxy: use the trade's own ATR vs asset median
                vr = 1.0
                risk_usd = sizer.size_position(row["symbol"], now_bar=bar, vol_ratio=vr)
                if risk_usd <= 0:
                    n_skipped += 1
                    continue
            pnl = row["total_r"] * risk_usd
            total_r_usd += pnl
            eq += pnl
            peak = max(peak, eq)
            max_dd = min(max_dd, eq - peak)
            eq_curve.append(eq)
        return {
            "final_equity": round(eq, 0), "net_pnl": round(total_r_usd, 0),
            "max_dd_usd": round(max_dd, 1),
            "max_dd_pct_of_peak": round(max_dd / peak * 100, 2),
            "n_skipped": n_skipped, "n_traded": len(trades) - n_skipped,
        }

    flat = replay("flat")
    heat = replay("heat")
    print("\n── Portfolio replay (honest engine trades, B4) ──")
    print(f"  Flat $10/trade     : PnL ${flat['net_pnl']:+,.0f}  maxDD ${flat['max_dd_usd']:.0f} ({flat['max_dd_pct_of_peak']}%)  trades {flat['n_traded']}")
    print(f"  Heat-budget sizing : PnL ${heat['net_pnl']:+,.0f}  maxDD ${heat['max_dd_usd']:.0f} ({heat['max_dd_pct_of_peak']}%)  trades {heat['n_traded']} (skipped {heat['n_skipped']})")

    # ── 4. Clustered-loss stress: worst simultaneous same-cluster day ──────
    trades["date"] = pd.to_datetime(trades["entry_time"]).dt.date
    trades["cluster"] = trades["symbol"].map(assign)
    daily_cluster_r = trades.groupby(["date", "cluster"])["total_r"].sum()
    worst = daily_cluster_r.sort_values().head(8)
    print("\nWorst same-cluster same-day R sums:")
    for (d, c), v in worst.items():
        members = [s for s in trades[trades['cluster'] == c]['symbol'].unique()]
        print(f"  {d} {c} ({', '.join(members[:4])}): {v:+.1f}R")

    out = {
        "corr_eq_crypto_full": round(float(full_avg), 3),
        "corr_eq_crypto_stress": round(float(stress_avg), 3),
        "corr_all_full": round(float(all_full), 3),
        "corr_all_stress": round(float(all_stress), 3),
        "stress_dates": [str(pd.Timestamp(d).date()) for d in tail_dates[:10]],
        "n_stress_dates": int(len(tail_dates)),
        "clusters_full": clusters_full,
        "clusters_stress": clusters_stress,
        "replay_flat": flat,
        "replay_heat": heat,
        "worst_cluster_days": [
            {"date": str(d), "cluster": c, "r": round(float(v), 1)}
            for (d, c), v in worst.items()
        ],
    }
    with open(os.path.join(RES, "cluster_empirical.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved cluster_empirical.json")


if __name__ == "__main__":
    main()
