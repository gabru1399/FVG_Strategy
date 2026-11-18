"""
m5_truth.py — Ground-truth intrabar resolution experiment.

For the 4 basket assets with 5M data (EURUSD, USDJPY, BTCUSD, NAS100), replay
the SAME 1H setups bar-by-bar on 5M data with the real state machine:
  - BE / 2R triggers fire when the 5M path actually touches them
  - the trail ratchets only at 1H boundaries, using the prior 1H close & ATR
    (exactly how a live MT5 EA would move the stop)
  - stop fills at level, or at the 5M open if it gaps through

This locates the truth between the repo's 1H engine (A0/A1) and the honest
pessimistic 1H engine (B4), on the SAME setup subset within 5M coverage.
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
sys.path.insert(0, str(REPO_ROOT))
from exit_engine import ExitPolicy, run_policy, wilder_atr

RES = str(PACKAGE_DIR / "results")
DATA = str(REPO_ROOT / "Blueberry_Data")

ASSETS = {
    "EURUSD.pi": "EURUSD.pi_5M.parquet",
    "USDJPY.pi": "USDJPY.pi_5M.parquet",
    "BTCUSD.pi": "BTCUSD.pi_5M.parquet",
    "NAS100.p": "NAS100.p_5M.parquet",
}

COST_R = {"EURUSD.pi": 0.017, "USDJPY.pi": 0.019, "BTCUSD.pi": 0.044, "NAS100.p": 0.031}


def load_m5(fn):
    df = pd.read_parquet(os.path.join(DATA, fn))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def simulate_on_m5(m5, merged, setups, k=1.5, max_hours=72, cost_r=0.0):
    """Replay 1H setups on 5M bars with true intra-hour ordering.

    merged : 1H frame (has atr14, time)
    m5     : 5M frame covering (part of) the same period
    Trail rule (live-EA equivalent): at each 1H boundary t, stop becomes
      max(prev_stop, highest 1H close since entry - k * ATR at prior 1H bar)
    BE at +1R touch, T1 at +2R touch (both evaluated on the 5M path).
    Stop fill: level, or 5M open if gapped through.
    """
    h_times = merged["time"].values
    h_atr = merged["atr14"].values
    h_close = merged["close"].values
    m5_times = m5["time"].values
    m5_high = m5["high"].values
    m5_low = m5["low"].values
    m5_open = m5["open"].values
    m5_close = m5["close"].values

    out = []
    for _, s in setups.iterrows():
        idx = int(s["entry_idx"])
        t_entry = h_times[idx]
        # 5M coverage window for this trade
        if t_entry < m5_times[0]:
            continue
        t_end_limit = t_entry + np.timedelta64(max_hours, "h")
        if t_entry > m5_times[-1]:
            continue
        j0 = np.searchsorted(m5_times, t_entry, side="left")
        j1 = np.searchsorted(m5_times, min(t_end_limit, m5_times[-1]), side="right")
        if j1 - j0 < 12:  # need at least an hour of 5M bars
            continue

        direction = s["direction"]
        entry_p = float(s["entry_price"])
        init_sl = float(s["stop_loss"])
        risk = float(s["risk_dist"])
        long = direction == "Long"
        sign = 1.0 if long else -1.0
        tp_1r = entry_p + sign * risk
        tp_2r = entry_p + sign * 2.0 * risk
        be_level = entry_p + sign * 0.1 * risk

        current_sl = init_sl
        be_active = False
        t1_locked = False
        t1_r = t2_r = None
        peak_close = entry_p
        exit_reason = "timeout"
        exit_j = j0
        mfe_r = mae_r = 0.0

        # 1H boundary bookkeeping: next 1H bar index after entry
        h_i = idx  # current 1H bar (the entry bar)

        for j in range(j0, j1):
            h, l, o = m5_high[j], m5_low[j], m5_open[j]
            # ---- crossing a 1H boundary? (5M bar belongs to a NEW 1H bar) ----
            new_h = np.searchsorted(h_times, m5_times[j], side="right") - 1
            if new_h > h_i and t1_locked:
                # ratchet trail using closes of completed 1H bars up to new_h-1
                for hh in range(h_i + 1, new_h):
                    if long:
                        peak_close = max(peak_close, h_close[hh])
                    else:
                        peak_close = min(peak_close, h_close[hh])
                atr_use = h_atr[new_h - 1] if not np.isnan(h_atr[new_h - 1]) else h_atr[idx]
                trail = (peak_close - k * atr_use) if long else (peak_close + k * atr_use)
                current_sl = max(current_sl, trail) if long else min(current_sl, trail)
                h_i = new_h
            elif new_h > h_i:
                h_i = new_h

            fav = (h - entry_p) * sign / risk if long else (entry_p - l) / risk
            adv = (l - entry_p) * sign / risk if long else (entry_p - h) / risk
            mfe_r = max(mfe_r, fav)
            mae_r = min(mae_r, adv)

            # ---- stop check at the OLD level (5M granularity) ----
            hit = (l <= current_sl) if long else (h >= current_sl)
            if hit:
                fill = current_sl
                if long and o < current_sl:
                    fill = o
                elif (not long) and o > current_sl:
                    fill = o
                exit_j = j
                if not t1_locked:
                    if be_active:
                        r_fill = (fill - entry_p) / risk if long else (entry_p - fill) / risk
                        full_r = min(0.1, r_fill)
                    else:
                        full_r = min(-1.0, ((fill - entry_p) / risk if long else (entry_p - fill) / risk))
                    t1_r = t2_r = full_r
                    exit_reason = "be_stop" if be_active else "init_stop"
                else:
                    r_fill = (fill - entry_p) / risk if long else (entry_p - fill) / risk
                    t2_r = max(1.0, r_fill)
                    exit_reason = "runner_stop"
                break

            # ---- triggers on the 5M path ----
            if not be_active and ((h >= tp_1r) if long else (l <= tp_1r)):
                be_active = True
                current_sl = max(current_sl, be_level) if long else min(current_sl, be_level)
            if not t1_locked and ((h >= tp_2r) if long else (l <= tp_2r)):
                t1_locked = True
                t1_r = 2.0
                current_sl = max(current_sl, tp_1r) if long else min(current_sl, tp_1r)

        if t2_r is None:
            # timeout: mark at last available 5M close
            last_c = m5_close[j1 - 1]
            curr_r = (last_c - entry_p) / risk if long else (entry_p - last_c) / risk
            if t1_locked:
                t1_r = 2.0
                t2_r = max(1.0, curr_r)
            elif be_active and curr_r <= 0.1:
                t1_r = t2_r = 0.1
            else:
                t1_r = t2_r = curr_r
            exit_j = j1 - 1

        rec = dict(s)
        rec.update({
            "t1_r": t1_r, "t2_r": t2_r,
            "total_r": 0.5 * t1_r + 0.5 * t2_r - cost_r,
            "hit_2r": bool(t1_locked), "exit_reason": exit_reason,
            "mfe_r": float(mfe_r), "mae_r": float(mae_r),
            "in_m5_window": True,
        })
        out.append(rec)
    return pd.DataFrame(out)


def main():
    R_DOLLAR = 10.0
    INIT = 2500.0

    def pmetrics(t):
        if t is None or len(t) == 0:
            return None
        pnl = t["total_r"] * R_DOLLAR
        eq = INIT + pnl.cumsum()
        peak = eq.cummax()
        gw = pnl[pnl > 0].sum(); gl = abs(pnl[pnl < 0].sum())
        return {
            "n": int(len(t)), "total_r": round(float(t["total_r"].sum()), 1),
            "pf": round(float(gw / gl), 2) if gl > 0 else 99.0,
            "max_dd_pct": round(float((eq - peak).min() / peak.max() * 100), 2),
            "hit2r_pct": round(float(t["hit_2r"].mean() * 100), 1),
            "avg_runner_r": round(float(t.loc[t["hit_2r"], "t2_r"].mean()), 2) if t["hit_2r"].any() else 0.0,
        }

    results = {}
    m5_frames = {}
    for sym, fn in ASSETS.items():
        key = sym.replace(".", "_")
        mp = os.path.join(RES, f"merged_{key}.parquet")
        sp = os.path.join(RES, f"setups_{key}.parquet")
        if not (os.path.exists(mp) and os.path.exists(sp)):
            continue
        merged = pd.read_parquet(mp)
        setups = pd.read_parquet(sp)
        m5 = load_m5(fn)
        m5_frames[sym] = (m5, merged, setups)
        print(f"{sym:12s} 5M bars: {len(m5):7d}  cover {m5['time'].iloc[0].date()} -> {m5['time'].iloc[-1].date()}  setups: {len(setups)}")

    for sym, (m5, merged, setups) in m5_frames.items():
        # 1) 5M ground truth (with costs)
        t_truth = simulate_on_m5(m5, merged, setups, k=1.5, cost_r=COST_R[sym])
        # 2) repo-exact 1H (with costs), same subset
        pol_repo = ExitPolicy("chandelier", k=1.5, intrabar="optimistic",
                              entry_bar_adverse=False, trail_same_bar=True, gap_fill=False)
        t_repo = run_policy(merged, setups, pol_repo, cost_r_fn=lambda s: COST_R[s])
        # 3) honest 1H (with costs), same subset
        pol_honest = ExitPolicy("chandelier", k=1.5, intrabar="pessimistic",
                                entry_bar_adverse=True, trail_same_bar=False, gap_fill=True)
        t_honest = run_policy(merged, setups, pol_honest, cost_r_fn=lambda s: COST_R[s])

        # restrict 1H results to the SAME setups the 5M replay could process
        keys = set(zip(t_truth["entry_idx"], t_truth["direction"]))
        t_repo_sub = t_repo[t_repo.apply(lambda r: (r["entry_idx"], r["direction"]) in keys, axis=1)]
        t_honest_sub = t_honest[t_honest.apply(lambda r: (r["entry_idx"], r["direction"]) in keys, axis=1)]

        results[sym] = {
            "n_5m": pmetrics(t_truth),
            "n_1h_repo": pmetrics(t_repo_sub),
            "n_1h_honest": pmetrics(t_honest_sub),
        }
        print(f"\n{sym}: subset n={len(t_truth)} (of {len(setups)} setups)")
        for tag in ("n_5m", "n_1h_repo", "n_1h_honest"):
            m = results[sym][tag]
            print(f"  {tag:12s} R={m['total_r']:+7.1f} PF={m['pf']:5.2f} 2R%={m['hit2r_pct']:4.1f} runner={m['avg_runner_r']:4.2f} DD%={m['max_dd_pct']:5.2f}")

        t_truth.to_parquet(os.path.join(RES, f"m5truth_{sym.replace('.', '_')}.parquet"), index=False)

    with open(os.path.join(RES, "m5_truth.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved m5_truth.json")


if __name__ == "__main__":
    main()
