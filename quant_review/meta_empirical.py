"""
meta_empirical.py — Run the meta-labeler on the REAL 974 basket trades.

Univariate feature ICs vs realized total_r (B4 honest engine), purged-CV
logistic model AUC / rank-IC / lift, and the skip-bottom-quintile policy.
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
from meta_label import build_features, make_labels, evaluate_meta_labeler, single_feature_ics

RES = str(PACKAGE_DIR / "results")

BASKET = ["NAS100.p", "BTCUSD.pi", "UK100.p", "AU200.p", "EURUSD.pi",
          "SOLUSD.p", "USDCHF.pi", "USDJPY.pi", "XAGUSD.pi", "JP225.p"]


def main():
    X_parts, t_parts = [], []
    for sym in BASKET:
        k = sym.replace(".", "_")
        mp = os.path.join(RES, f"merged_{k}.parquet")
        sp = os.path.join(RES, f"setups_{k}.parquet")
        tp = os.path.join(RES, f"trades_B4_plus_gap_fills.parquet")
        if not (os.path.exists(mp) and os.path.exists(sp)):
            continue
        merged = pd.read_parquet(mp)
        setups = pd.read_parquet(sp)
        X = build_features(merged, setups)
        X_parts.append(X)
    X = pd.concat(X_parts, ignore_index=True)
    trades = pd.read_parquet(os.path.join(RES, "trades_B4_plus_gap_fills.parquet"))
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    assert len(X) == len(trades), f"feature rows {len(X)} != trades {len(trades)}"

    r = trades["total_r"].values
    y_win = (r > 0.25).astype(int)
    y_notloss = (r > -0.05).astype(int)
    times = trades["entry_time"].values

    print(f"Real meta-labeling: {len(r)} trades, win-rate(label>0.25R)={y_win.mean()*100:.1f}%")
    print("\n── Univariate feature ICs vs realized R (B4 honest engine) ──")
    ics = single_feature_ics(X, r, y_win)
    print(ics.to_string(index=False))

    print("\n── Purged-CV logistic model (label = total_r > 0.25) ──")
    res_win, preds_win = evaluate_meta_labeler(X, y_win, r, times, n_splits=5, embargo_bars=24)
    print(f"folds={res_win['n_folds']} scored={res_win['n_scored']} AUC={res_win['auc']:.3f} rank_IC={res_win['rank_ic']:.3f}")
    print("Lift by predicted-probability quintile:")
    print(pd.DataFrame(res_win["lift"]).to_string(index=False))
    print(f"baseline expectancy {res_win['baseline_expectancy_r']:+.4f} R -> skip bottom quintile {res_win['skip_bottom_quintile_expectancy_r']:+.4f} R (skips {res_win['skip_trades_pct']}% of trades)")

    print("\n── Purged-CV logistic model (label = total_r > -0.05, i.e. avoid losses) ──")
    res_nl, preds_nl = evaluate_meta_labeler(X, y_notloss, r, times, n_splits=5, embargo_bars=24)
    print(f"folds={res_nl['n_folds']} scored={res_nl['n_scored']} AUC={res_nl['auc']:.3f} rank_IC={res_nl['rank_ic']:.3f}")

    out = {
        "univariate_ics": ics.to_dict(orient="records"),
        "model_win_label": {k: v for k, v in res_win.items() if k != "feature_names"},
        "model_notloss_label": {k: v for k, v in res_nl.items() if k != "feature_names"},
        "feature_names": res_win["feature_names"],
    }
    with open(os.path.join(RES, "meta_empirical.json"), "w") as f:
        json.dump(out, f, indent=2)
    # save preds for charting
    pd.DataFrame({"entry_time": trades["entry_time"], "pred_win": preds_win,
                  "total_r": r, "symbol": trades["symbol"]}).to_parquet(
        os.path.join(RES, "meta_preds.parquet"), index=False)
    print("\nSaved meta_empirical.json")


if __name__ == "__main__":
    main()
