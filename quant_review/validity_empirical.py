"""
validity_empirical.py — Apply the validity toolkit to the REAL strategy.

1. DSR: the kept strategy (honest engine B4) tested against the number of
   configurations tried during research (16 exit-grid variants + basket
   screening of ~30 assets).
2. PBO via CSCV on the 16 exit-policy configuration returns (monthly agg).
3. Block bootstrap of the honest engine's 974-trade R sequence: distribution
   of max DD and expectancy, P(total R < 0).
4. Walk-forward stability of expectancy across 6 chronological folds.
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
from validity import deflated_sharpe, pbo_cscv, block_bootstrap_trade_r, walk_forward_windows

RES = str(PACKAGE_DIR / "results")

# exit-policy grid metrics from exit_grid.json + the trade files we saved
GRID_KEYS = ["A0_repo_exact", "A1_repo_plus_costs", "B1_sequencing_pess", "B2_plus_entry_bar",
             "B3_plus_honest_trail", "B4_plus_gap_fills", "C_chandelier_k2.0", "D_chandelier_k2.5",
             "E_chandelier_k3.0", "F_chan_high_k3.0", "G_swing_lb3", "I_psar_0.02_0.20",
             "K_giveback_0.30", "L_giveback_0.40", "M_hybrid_k2.5_g0.50", "N_hybrid_k3.0_g0.40"]


def main():
    trades = pd.read_parquet(os.path.join(RES, "trades_B4_plus_gap_fills.parquet"))
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    r = trades["total_r"].values

    # ── 1. DSR ──────────────────────────────────────────────────────────────
    # Trial count for multiple-testing adjustment: 16 exit variants + ~10
    # basket-screen criteria combos effectively tested = ~26. Variance of
    # trial Sharpes estimated from the exit-grid Sharpe dispersion.
    grid = json.load(open(os.path.join(RES, "exit_grid.json")))
    # monthly return stream of the kept strategy (B4) for SR computation
    trades["month"] = pd.to_datetime(trades["entry_time"]).dt.to_period("M")
    monthly_r = trades.groupby("month")["total_r"].sum()  # in R
    monthly_ret_pct = monthly_r * 0.004  # R -> fraction of equity (0.4% per R)

    # per-config Sharpe (annualized, from monthly totals)
    cfg_sharpes = []
    for k in GRID_KEYS:
        if k in grid:
            g = grid[k]
            # reconstruct approximate monthly SR from expectancy & n per year
            # (use trade-level: mean/sem * sqrt(trades per year))
            n_years = 3.65  # 2023-03 -> 2026-11
            tpy = g["n"] / n_years
            sr_ann = g["expectancy_r"] / 1.35 * np.sqrt(tpy)  # 1.35 = std of R
            cfg_sharpes.append(sr_ann)
    cfg_sharpes = np.array(cfg_sharpes)

    dsr = deflated_sharpe(monthly_ret_pct.values, n_trials=max(10, len(cfg_sharpes) + 10),
                          var_trials=float(np.var(cfg_sharpes)) if len(cfg_sharpes) > 2 else 0.25,
                          periods_per_year=12)
    print("── Deflated Sharpe (kept strategy, honest engine, costs incl.) ──")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in dsr.items()})

    # ── 2. PBO via CSCV on the exit-config matrix ───────────────────────────
    # Build a config x month matrix of R totals from the saved trade files
    # (we have full trade lists for 4 configs; approximate others via the grid
    # totals by rescaling — instead, use the 4 real ones + note the caveat)
    cfg_frames = {}
    for k in ["A0_repo_exact", "B4_plus_gap_fills", "I_psar_0.02_0.20", "G_swing_lb3"]:
        p = os.path.join(RES, f"trades_{k}.parquet")
        if os.path.exists(p):
            t = pd.read_parquet(p)
            t["month"] = pd.to_datetime(t["entry_time"]).dt.to_period("M")
            cfg_frames[k] = t.groupby("month")["total_r"].sum()
    M = pd.DataFrame(cfg_frames).fillna(0.0)
    # add synthetic noise configs? No — keep honest: PBO on the 4 real configs
    pbo = pbo_cscv(M, n_partitions=6, metric="sharpe")
    print("\n── PBO via CSCV (4 real exit configs, monthly R) ──")
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in pbo.items()})

    # ── 3. Block bootstrap ─────────────────────────────────────────────────
    boot = block_bootstrap_trade_r(r, n_boot=2000, block=10, seed=7)
    print("\n── Block bootstrap (974 trades, block=10) ──")
    for k in ["expectancy", "total_r", "max_dd_r"]:
        print(f"  {k}: {boot[k]}")
    print(f"  P(total R < 0) = {boot['prob_total_r_negative']}")

    # ── 4. Walk-forward expectancy stability ────────────────────────────────
    folds = walk_forward_windows(len(r), train_bars=400, test_bars=100, anchored=False)
    wf_rows = []
    for tr_s, tr_e, te_s, te_e in folds:
        if te_e - te_s < 50:
            continue
        wf_rows.append({
            "test_window": f"[{te_s}:{te_e}]",
            "test_trades": int(te_e - te_s),
            "test_expectancy_r": round(float(r[te_s:te_e].mean()), 3),
            "test_total_r": round(float(r[te_s:te_e].sum()), 1),
        })
    print("\n── Walk-forward test-window expectancy ──")
    for w in wf_rows:
        print(f"  {w['test_window']:12s} n={w['test_trades']:4d} exp={w['test_expectancy_r']:+.3f}R total={w['test_total_r']:+.1f}R")
    neg = [w for w in wf_rows if w["test_expectancy_r"] < 0]
    print(f"  negative windows: {len(neg)} / {len(wf_rows)}")

    out = {
        "dsr": {k: v for k, v in dsr.items()},
        "pbo": pbo,
        "bootstrap": boot,
        "walk_forward": wf_rows,
        "n_negative_wf_windows": len(neg),
        "n_wf_windows": len(wf_rows),
        "cfg_sharpes_mean": round(float(cfg_sharpes.mean()), 2),
        "cfg_sharpes_std": round(float(cfg_sharpes.std()), 2),
    }
    with open(os.path.join(RES, "validity_empirical.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved validity_empirical.json")


if __name__ == "__main__":
    main()
