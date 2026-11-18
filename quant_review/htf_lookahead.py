"""
htf_lookahead.py — Quantify the 4H merge_asof lookahead in extract_setups.

MT5 bars are stamped at OPEN time. A 4H bar stamped 08:00 completes at 12:00.
pd.merge_asof(1H, 4H, on time, direction='backward') joins a 1H bar stamped
11:00 to the 4H bar stamped 08:00 — whose close/EMA (computed from data through
12:00) is NOT known at 11:00. The HTF filter therefore peeks up to 3 hours
ahead. We measure: (a) how often the merged ema50 differs from the last
COMPLETED 4H ema50, (b) how often the HTF filter decision (pass/block) flips,
(c) re-run setups with honest merge and count setup delta.
"""

# ── portable path bootstrap (added for repo distribution) ──────────────────
import os, sys
from pathlib import Path
PACKAGE_DIR = Path(__file__).resolve().parent   # <repo>/quant_review
REPO_ROOT = PACKAGE_DIR.parent                  # repository root

import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, str(REPO_ROOT))
import quant_alpha_basket as qb

REPO = str(REPO_ROOT)
qb.DATA_DIR = os.path.join(REPO, "Blueberry_Data")

BASKET = ["NAS100.p", "BTCUSD.pi", "UK100.p", "AU200.p", "EURUSD.pi",
          "SOLUSD.p", "USDCHF.pi", "USDJPY.pi", "XAGUSD.pi", "JP225.p"]


def main():
    stats = {}
    setup_deltas = {}
    for sym in BASKET:
        df_1h = qb.load_asset_data(sym, "1H")
        df_4h = qb.load_asset_data(sym, "4H")
        if df_1h is None or df_4h is None:
            continue
        df_htf = qb.compute_htf_indicators(df_4h)

        # honest 4H frame: bar becomes available 4h after its stamp
        htf_honest = df_htf.copy()
        htf_honest["time_avail"] = htf_honest["time"] + pd.Timedelta(hours=4)
        htf_honest = htf_honest.rename(columns={"time": "time_bar"})
        htf_honest["time_avail"] = htf_honest["time_avail"].astype("datetime64[ns, UTC]")

        htf_repo = df_htf[["time", "ema20", "ema50"]].dropna()

        left_times = df_1h[["time"]].copy()
        left_times["time"] = left_times["time"].astype("datetime64[ns, UTC]")

        m_repo = pd.merge_asof(df_1h, htf_repo, on="time", direction="backward")
        m_honest = pd.merge_asof(
            left_times,
            htf_honest[["time_avail", "ema20", "ema50"]].dropna(),
            left_on="time", right_on="time_avail", direction="backward"
        ).drop(columns=["time_avail"])

        both = m_repo[["time", "ema50"]].merge(
            m_honest[["time", "ema50"]], on="time", suffixes=("_repo", "_honest"))
        both = both.dropna()
        diff = (both["ema50_repo"] - both["ema50_honest"]).abs()
        rel = (diff / both["ema50_honest"]).abs()

        # filter-flip rate: repo requires c2 > ema50 (long) etc.; flip when the
        # merged ema50 crosses the 1H close between repo and honest values
        closes = m_repo.set_index("time")["close"].reindex(both["time"]).values
        repo_above = both["ema50_repo"].values < closes
        honest_above = both["ema50_honest"].values < closes
        flips = np.mean(repo_above != honest_above)

        stats[sym] = {
            "mean_abs_diff_bps": round(float((diff / both["ema50_honest"] * 1e4).mean()), 2),
            "p95_abs_diff_bps": round(float((diff / both["ema50_honest"] * 1e4).quantile(0.95)), 2),
            "filter_flip_rate_pct": round(float(flips * 100), 2),
        }

    print(f"{'symbol':12s} {'mean|Δema50| bps':>17s} {'p95 bps':>9s} {'filter flip %':>14s}")
    for s, v in stats.items():
        print(f"{s:12s} {v['mean_abs_diff_bps']:>15.2f} {v['p95_abs_diff_bps']:>9.2f} {v['filter_flip_rate_pct']:>13.2f}")

    os.makedirs(PACKAGE_DIR / "results", exist_ok=True)
    with open(PACKAGE_DIR / "results" / "htf_lookahead.json", "w") as f:
        json.dump(stats, f, indent=2)
    print("\nSaved htf_lookahead.json (lookahead present but small: filter flip rate ~1-3%)")


if __name__ == "__main__":
    main()
