"""
exit_grid.py — run the exit-policy comparison grid on the REAL basket setups
(merged frames + setups saved by audit_repro.py), with realistic costs and
pessimistic intrabar resolution. Also produces MFE/MAE-based diagnostics.

Outputs: results/exit_grid.json (+ trades parquet per policy for charting)
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
from exit_engine import ExitPolicy, run_policy, wilder_atr

RES = str(PACKAGE_DIR / "results")

BASKET = ["NAS100.p", "BTCUSD.pi", "UK100.p", "AU200.p", "EURUSD.pi",
          "SOLUSD.p", "USDCHF.pi", "USDJPY.pi", "XAGUSD.pi", "JP225.p"]

def sym_key(s):
    return s.replace(".", "_")

# cost in R per trade (from audit_repro table)
COST_R = {
    "EURUSD.pi": 0.017, "USDJPY.pi": 0.019, "USDCHF.pi": 0.025,
    "BTCUSD.pi": 0.044, "SOLUSD.p": 0.072, "NAS100.p": 0.031,
    "UK100.p": 0.034, "AU200.p": 0.037, "JP225.p": 0.078, "XAGUSD.pi": 0.017,
}

R_DOLLAR = 10.0
INIT_EQ = 2500.0

def portfolio(trades):
    t = trades.sort_values("entry_time").reset_index(drop=True)
    pnl = t["total_r"] * R_DOLLAR
    eq = INIT_EQ + pnl.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    gw = pnl[pnl > 0].sum(); gl = abs(pnl[pnl < 0].sum())
    return {
        "n": int(len(t)),
        "total_r": round(float(t["total_r"].sum()), 1),
        "net_usd": round(float(pnl.sum()), 0),
        "pf": round(float(gw / gl), 3) if gl > 0 else 99.0,
        "max_dd_usd": round(float(dd.min()), 1),
        "max_dd_pct": round(float(dd.min() / peak.max() * 100), 2),
        "expectancy_r": round(float(t["total_r"].mean()), 3),
        "hit2r_pct": round(float(t["hit_2r"].mean() * 100), 1),
        "be_pct": round(float(((t["total_r"] >= 0) & (t["total_r"] <= 0.5)).mean() * 100), 1),
        "loss_pct": round(float((t["total_r"] < 0).mean() * 100), 1),
        "avg_win_r": round(float(t.loc[t["total_r"] > 0.5, "total_r"].mean()), 2) if (t["total_r"] > 0.5).any() else 0.0,
        "avg_runner_r": round(float(t.loc[t["hit_2r"], "t2_r"].mean()), 2) if t["hit_2r"].any() else 0.0,
        "timeout_pct": round(float((t["exit_reason"] == "timeout").mean() * 100), 1),
    }

POLICIES = [
    # ── decomposition ladder: A0 replicates the repo engine exactly ────────
    ("A0_repo_exact",           ExitPolicy("chandelier", k=1.5, intrabar="optimistic",
                                           entry_bar_adverse=False, trail_same_bar=True, gap_fill=False)),
    ("A1_repo_plus_costs",      ExitPolicy("chandelier", k=1.5, intrabar="optimistic",
                                           entry_bar_adverse=False, trail_same_bar=True, gap_fill=False)),
    # B-ladder: add one realism correction at a time (all with costs)
    ("B1_sequencing_pess",      ExitPolicy("chandelier", k=1.5, intrabar="pessimistic",
                                           entry_bar_adverse=False, trail_same_bar=True, gap_fill=False)),
    ("B2_plus_entry_bar",       ExitPolicy("chandelier", k=1.5, intrabar="pessimistic",
                                           entry_bar_adverse=True, trail_same_bar=True, gap_fill=False)),
    ("B3_plus_honest_trail",    ExitPolicy("chandelier", k=1.5, intrabar="pessimistic",
                                           entry_bar_adverse=True, trail_same_bar=False, gap_fill=False)),
    ("B4_plus_gap_fills",       ExitPolicy("chandelier", k=1.5, intrabar="pessimistic",
                                           entry_bar_adverse=True, trail_same_bar=False, gap_fill=True)),
    # ── policy variants on the fully honest engine (B4 setting) ────────────
    ("C_chandelier_k2.0",       ExitPolicy("chandelier", k=2.0, intrabar="pessimistic",
                                           entry_bar_adverse=True, trail_same_bar=False, gap_fill=True)),
    ("D_chandelier_k2.5",       ExitPolicy("chandelier", k=2.5, intrabar="pessimistic",
                                           entry_bar_adverse=True, trail_same_bar=False, gap_fill=True)),
    ("E_chandelier_k3.0",       ExitPolicy("chandelier", k=3.0, intrabar="pessimistic",
                                           entry_bar_adverse=True, trail_same_bar=False, gap_fill=True)),
    ("F_chan_high_k3.0",        ExitPolicy("chandelier_high", k=3.0, intrabar="pessimistic",
                                           entry_bar_adverse=True, trail_same_bar=False, gap_fill=True)),
    ("G_swing_lb3",             ExitPolicy("swing", swing_lb=3, intrabar="pessimistic",
                                           entry_bar_adverse=True, trail_same_bar=False, gap_fill=True)),
    ("I_psar_0.02_0.20",        ExitPolicy("psar", psar_af0=0.02, psar_af_max=0.20,
                                           intrabar="pessimistic",
                                           entry_bar_adverse=True, trail_same_bar=False, gap_fill=True)),
    ("K_giveback_0.30",         ExitPolicy("giveback", giveback=0.30, intrabar="pessimistic",
                                           entry_bar_adverse=True, trail_same_bar=False, gap_fill=True)),
    ("L_giveback_0.40",         ExitPolicy("giveback", giveback=0.40, intrabar="pessimistic",
                                           entry_bar_adverse=True, trail_same_bar=False, gap_fill=True)),
    ("M_hybrid_k2.5_g0.50",     ExitPolicy("hybrid", k=2.5, giveback=0.50, intrabar="pessimistic",
                                           entry_bar_adverse=True, trail_same_bar=False, gap_fill=True)),
    ("N_hybrid_k3.0_g0.40",     ExitPolicy("hybrid", k=3.0, giveback=0.40, intrabar="pessimistic",
                                           entry_bar_adverse=True, trail_same_bar=False, gap_fill=True)),
]
NO_COST = {"A0_repo_exact"}

def main():
    # load saved merged frames + setups
    frames = {}
    for sym in BASKET:
        k = sym_key(sym)
        mp = os.path.join(RES, f"merged_{k}.parquet")
        sp = os.path.join(RES, f"setups_{k}.parquet")
        if os.path.exists(mp) and os.path.exists(sp):
            frames[sym] = (pd.read_parquet(mp), pd.read_parquet(sp))
    print(f"Loaded {len(frames)} assets: {[s for s in frames]}")

    results = {}
    trades_by_policy = {}
    for name, pol in POLICIES:
        all_t = []
        for sym, (merged, setups) in frames.items():
            if name in NO_COST:
                cost_fn = None
            else:
                cost_fn = lambda s: COST_R.get(s, 0.03)
            t = run_policy(merged, setups, pol, cost_r_fn=cost_fn)
            if len(t):
                all_t.append(t)
        if not all_t:
            continue
        trades = pd.concat(all_t, ignore_index=True)
        trades_by_policy[name] = trades
        results[name] = portfolio(trades)
        r = results[name]
        print(f"{name:32s} n={r['n']:4d} R={r['total_r']:+7.1f} PF={r['pf']:5.2f} "
              f"DD%={r['max_dd_pct']:5.2f} exp={r['expectancy_r']:+.3f} "
              f"2R%={r['hit2r_pct']:4.1f} runner={r['avg_runner_r']:4.2f} to%={r['timeout_pct']:4.1f}")

    with open(os.path.join(RES, "exit_grid.json"), "w") as f:
        json.dump(results, f, indent=2)

    # save a compact trade table for key policies (charting / equity curves)
    for key in ["A0_repo_exact", "B4_plus_gap_fills", "I_psar_0.02_0.20", "G_swing_lb3"]:
        if key in trades_by_policy:
            trades_by_policy[key].to_parquet(os.path.join(RES, f"trades_{key}.parquet"), index=False)

    # MFE/MAE diagnostics on pessimistic baseline: how much open profit is given
    # back by runners, and what MFE distribution says about tail capture
    tb = trades_by_policy.get("B_repo_k1.5_pessimistic")
    if tb is not None:
        hit = tb[tb["hit_2r"]]
        diag = {
            "n_hit2r": int(len(hit)),
            "mfe_r_mean": round(float(hit["mfe_r"].mean()), 2),
            "mfe_r_p50": round(float(hit["mfe_r"].median()), 2),
            "mfe_r_p90": round(float(hit["mfe_r"].quantile(0.90)), 2),
            "runner_r_mean": round(float(hit["t2_r"].mean()), 2),
            "capture_ratio_mean": round(float((hit["t2_r"] / hit["mfe_r"].clip(lower=2.0)).mean()), 3),
            "capture_ratio_p50": round(float((hit["t2_r"] / hit["mfe_r"].clip(lower=2.0)).median()), 3),
        }
        print("\nMFE diagnostics (2R-hit trades):", diag)
        with open(os.path.join(RES, "mfe_diag.json"), "w") as f:
            json.dump(diag, f, indent=2)

    print("\nSaved exit_grid.json")

if __name__ == "__main__":
    main()
