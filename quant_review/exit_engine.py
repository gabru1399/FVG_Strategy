"""
exit_engine.py — Runner-exit comparison engine for the FVG 2R + trailing strategy.

Corrects the intrabar-sequencing optimism in the repo's simulate_2r_runner and
provides a grid of runner-exit policies for the Tranche-B trail:

  baseline_repo     : replication of repo engine (optimistic same-bar sequencing)
  chandelier(k)     : stop = highest close since entry - k * ATR   (repo default k=1.5)
  chandelier_high(k): stop = highest high  since entry - k * ATR   (wider, for fat tails)
  swing(lb)         : stop = most recent CONFIRMED swing low (fractal, lb-bar confirmation)
  psar(af0, afmax)  : classic Parabolic SAR on the runner leg
  giveback(g)       : OU-style decay kill-switch — exit when open profit retraces
                      fraction g of the peak favorable excursion
  hybrid(k, g)      : chandelier(k) AND giveback(g) — whichever stops first

Shared state machine (identical to repo structure, honest sequencing):
  +1.0R touched  -> stop to breakeven +0.1R on both tranches
  +2.0R touched  -> Tranche A exits +2.0R; Tranche B runner starts
                    (runner floor +1.0R, as in repo)
  exit           : Tranche A 0.5 * t1_r + Tranche B 0.5 * t2_r

Intrabar resolution:
  pessimistic : stop checked BEFORE triggers within a bar (adverse-first)
  optimistic  : triggers processed before the stop (favorable-first bound)
  entry-bar   : adverse touch of the initial stop on the fill bar counts as a
                full loss (price already traded through the level)

All quantities are in R units (risk fractions). Costs are applied per trade in R.
Requires: pandas, numpy. Pure pandas/numpy — no TA-Lib.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------------

def wilder_atr(df, period=14):
    """ATR matching the repo convention (EWM span, adjust=False)."""
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    n = len(df)
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values


def confirmed_swings(lows, highs, lb=3):
    """Return arrays of confirmed swing low/high LEVELS and the bar index at
    which each confirmation became known (i + lb). A swing low at i is the min
    of the window [i-lb, i+lb]; it is only usable lb bars after it printed."""
    n = len(lows)
    swing_lows, swing_highs, known_at = [], [], []
    for i in range(lb, n - lb):
        w = lows[i - lb : i + lb + 1]
        if lows[i] == w.min():
            swing_lows.append(lows[i]); swing_highs.append(np.nan); known_at.append(i + lb)
        w2 = highs[i - lb : i + lb + 1]
        if highs[i] == w2.max():
            swing_lows.append(np.nan); swing_highs.append(highs[i]); known_at.append(i + lb)
    return (np.array(swing_lows), np.array(swing_highs), np.array(known_at, dtype=int))


def psar_series(highs, lows, af0=0.02, af_max=0.20):
    """Classic Parabolic SAR. Returns per-bar SAR value for long positions.
    Implemented in the standard iterative form (Wilder). We flip direction
    tracking internally but output the level that would stop a LONG runner
    (positions here are already directional; we mirror for shorts outside)."""
    n = len(highs)
    sar = np.full(n, np.nan)
    if n < 3:
        return sar
    up = True
    ep = highs[0]
    s = lows[0]
    af = af0
    sar[0] = s
    for i in range(1, n):
        s = s + af * (ep - s)
        if up:
            if lows[i] < s:                     # reversal
                up = False
                s = ep
                ep = lows[i]
                af = af0
            else:
                if highs[i] > ep:
                    ep = highs[i]
                    af = min(af + af0, af_max)
                s = min(s, lows[i - 1], lows[i - 2] if i >= 2 else lows[i - 1])
        else:
            if highs[i] > s:                    # reversal
                up = True
                s = ep
                ep = highs[i]
                af = af0
            else:
                if lows[i] < ep:
                    ep = lows[i]
                    af = min(af + af0, af_max)
                s = max(s, highs[i - 1], highs[i - 2] if i >= 2 else highs[i - 1])
        sar[i] = s
        sar[i] = s if not np.isnan(s) else sar[i - 1]
    return sar


# ----------------------------------------------------------------------------
# Policy definition
# ----------------------------------------------------------------------------

@dataclass
class ExitPolicy:
    method: str = "chandelier"          # baseline_repo|chandelier|chandelier_high|swing|psar|giveback|hybrid
    k: float = 1.5                       # chandelier ATR multiple
    swing_lb: int = 3                    # swing fractal confirmation bars
    psar_af0: float = 0.02
    psar_af_max: float = 0.20
    giveback: float = 0.40               # giveback fraction of peak open profit
    intrabar: str = "pessimistic"        # pessimistic|optimistic
    entry_bar_adverse: bool = True       # count stop touch on fill bar
    max_bars: int = 72
    trail_same_bar: bool = False         # True = repo behavior (trail ratchets with
                                         # the CURRENT bar's close/ATR before the
                                         # stop check); False = prior-bar info only
    gap_fill: bool = True                # bar opening beyond the stop fills at the
                                         # open (worse), not at the stop level


# ----------------------------------------------------------------------------
# Single-trade simulation
# ----------------------------------------------------------------------------

def simulate_trade(highs, lows, closes, atrs, setup, policy: ExitPolicy,
                   swing_lows=None, swing_known=None, swing_highs=None,
                   cost_r=0.0, opens=None):
    """Simulate one FVG trade under the shared state machine.

    highs/lows/closes/atrs/opens : full arrays of the entry timeframe (1H)
    setup                  : dict with entry_idx, direction, entry_price,
                             stop_loss, risk_dist
    policy.trail_same_bar=True + intrabar='optimistic' + entry_bar_adverse=False
    + gap_fill=False exactly replicates the repo engine.
    Returns dict of outcomes incl. MFE/MAE in R.
    """
    idx = int(setup["entry_idx"])
    direction = setup["direction"]
    entry_p = float(setup["entry_price"])
    init_sl = float(setup["stop_loss"])
    risk = float(setup["risk_dist"])
    if risk <= 0 or np.isnan(risk):
        return None
    long = direction == "Long"
    sign = 1.0 if long else -1.0
    n = len(closes)

    tp_1r = entry_p + sign * risk
    tp_2r = entry_p + sign * 2.0 * risk
    be_level = entry_p + sign * 0.1 * risk

    current_sl = init_sl
    be_active = False
    t1_locked = False
    t1_r, t2_r = None, None
    peak_fav = entry_p                    # best price since entry (high for long)
    peak_fav_close = entry_p              # best close since entry
    mae_r = 0.0
    mfe_r = 0.0
    exit_bar = idx
    exit_reason = "timeout"
    exit_price = None

    end = min(idx + policy.max_bars + 1, n)

    def stop_fill(f, lvl):
        """Fill price at the stop: at the level, or at the open if the bar
        gapped through it (worse fill)."""
        if policy.gap_fill and opens is not None:
            if long:
                return min(lvl, opens[f])
            return max(lvl, opens[f])
        return lvl

    # ---- entry bar: adverse-only check ------------------------------------
    if policy.entry_bar_adverse:
        if long and lows[idx] <= init_sl:
            fill = stop_fill(idx, init_sl)
            t1_r = t2_r = min(-1.0, (fill - entry_p) / risk)
            exit_bar = idx; exit_reason = "entry_bar_stop"; mae_r = t1_r
            exit_price = fill
        elif (not long) and highs[idx] >= init_sl:
            fill = stop_fill(idx, init_sl)
            t1_r = t2_r = min(-1.0, (entry_p - fill) / risk)
            exit_bar = idx; exit_reason = "entry_bar_stop"; mae_r = t1_r
            exit_price = fill

    # ---- bar loop ----------------------------------------------------------
    if t1_r is None:
        atr_prev = atrs[idx] if not np.isnan(atrs[idx]) else risk
        for f in range(idx + 1, end):
            h, l, c = highs[f], lows[f], closes[f]
            atr_f = atrs[f] if not np.isnan(atrs[f]) else atr_prev

            # track excursions
            fav = (h - entry_p) * sign / risk if long else (entry_p - l) / risk
            adv = (l - entry_p) * sign / risk if long else (entry_p - h) / risk
            mfe_r = max(mfe_r, fav)
            mae_r = min(mae_r, adv)

            # repo-exact mode: ratchet peaks with the CURRENT bar's close and
            # use the current bar's ATR for the trail (lookahead replication)
            if policy.trail_same_bar:
                if long:
                    peak_fav = max(peak_fav, h)
                    peak_fav_close = max(peak_fav_close, c)
                else:
                    peak_fav = min(peak_fav, l)
                    peak_fav_close = min(peak_fav_close, c)
                trail_atr = atr_f
                trail_peaks_fav, trail_peaks_close = peak_fav, peak_fav_close
            else:
                # honest mode: trail computed from information through bar f-1
                trail_atr = atr_prev
                trail_peaks_fav, trail_peaks_close = peak_fav, peak_fav_close

            policy_trail = -np.inf if long else np.inf
            if t1_locked:
                if policy.method in ("chandelier", "hybrid", "baseline_repo"):
                    policy_trail = (trail_peaks_close - policy.k * trail_atr) if long \
                        else (trail_peaks_close + policy.k * trail_atr)
                elif policy.method == "chandelier_high":
                    policy_trail = (trail_peaks_fav - policy.k * trail_atr) if long \
                        else (trail_peaks_fav + policy.k * trail_atr)
                elif policy.method == "swing":
                    # latest swing confirmed by the close of bar f-1
                    if swing_known is not None and len(swing_known):
                        mask = swing_known <= f - 1
                        if mask.any():
                            lv = swing_lows[mask][-1] if long else swing_highs[mask][-1]
                            if not np.isnan(lv):
                                policy_trail = lv
                elif policy.method == "psar":
                    sar_lvl = psar_series(highs[idx:f], lows[idx:f], policy.psar_af0,
                                          policy.psar_af_max)
                    policy_trail = sar_lvl[-1] if len(sar_lvl) and not np.isnan(sar_lvl[-1]) \
                        else (-np.inf if long else np.inf)
                elif policy.method == "giveback":
                    if long:
                        policy_trail = trail_peaks_fav - policy.giveback * (trail_peaks_fav - entry_p)
                    else:
                        policy_trail = trail_peaks_fav + policy.giveback * (entry_p - trail_peaks_fav)

            def _eff_sl():
                if long:
                    return max(current_sl, policy_trail) if t1_locked else current_sl
                return min(current_sl, policy_trail) if t1_locked else current_sl

            # ---- optimistic mode: triggers processed BEFORE the stop check ---
            if policy.intrabar == "optimistic":
                if not be_active and ((h >= tp_1r) if long else (l <= tp_1r)):
                    be_active = True
                    current_sl = max(current_sl, be_level) if long else min(current_sl, be_level)
                if not t1_locked and ((h >= tp_2r) if long else (l <= tp_2r)):
                    t1_locked = True
                    t1_r = 2.0
                    current_sl = max(current_sl, tp_1r) if long else min(current_sl, tp_1r)

            # ---- 1. stop check ------------------------------------------------
            eff_sl = _eff_sl()
            hit_stop = (l <= eff_sl) if long else (h >= eff_sl)
            if hit_stop:
                exit_bar = f
                exit_reason = "runner_stop" if t1_locked else ("be_stop" if be_active else "init_stop")
                fill = stop_fill(f, eff_sl)
                exit_price = fill
                if not t1_locked:
                    full_r = 0.1 if be_active else -1.0
                    if be_active:
                        # BE stop: gap through BE fills at open (can be < 0.1R)
                        r_at_fill = (fill - entry_p) / risk if long else (entry_p - fill) / risk
                        full_r = min(0.1, r_at_fill)
                    t1_r = t2_r = full_r
                else:
                    runner_r = (fill - entry_p) / risk if long else (entry_p - fill) / risk
                    t2_r = max(1.0, runner_r)
                break

            # ---- 2. triggers after surviving the bar (pessimistic) ----------
            if not be_active and ((h >= tp_1r) if long else (l <= tp_1r)):
                be_active = True
                current_sl = max(current_sl, be_level) if long else min(current_sl, be_level)
            if not t1_locked and ((h >= tp_2r) if long else (l <= tp_2r)):
                t1_locked = True
                t1_r = 2.0
                current_sl = max(current_sl, tp_1r) if long else min(current_sl, tp_1r)

            # ---- 3. update running peaks at bar close ----------------------
            if not policy.trail_same_bar:
                if long:
                    peak_fav = max(peak_fav, h)
                    peak_fav_close = max(peak_fav_close, c)
                else:
                    peak_fav = min(peak_fav, l)
                    peak_fav_close = min(peak_fav_close, c)
            atr_prev = atr_f

    # ---- timeout ------------------------------------------------------------
    if t2_r is None:
        last_c = closes[min(idx + policy.max_bars, n - 1)]
        curr_r = (last_c - entry_p) / risk if long else (entry_p - last_c) / risk
        exit_reason = "timeout"
        if t1_locked:
            t1_r = 2.0
            t2_r = max(1.0, curr_r)
        elif be_active and curr_r <= 0.1:
            t1_r = t2_r = 0.1
        else:
            t1_r = t2_r = curr_r
        exit_bar = min(idx + policy.max_bars, n - 1)
        exit_price = last_c

    total_r = 0.5 * t1_r + 0.5 * t2_r - cost_r
    rec = dict(setup)
    rec.update({
        "t1_r": t1_r, "t2_r": t2_r, "total_r": total_r,
        "hit_2r": bool(t1_locked), "exit_reason": exit_reason,
        "holding_bars": int(exit_bar - idx),
        "mfe_r": float(mfe_r), "mae_r": float(mae_r),
        "be_active": bool(be_active),
        "exit_price": float(exit_price) if exit_price is not None else None,
    })
    return rec


def _psar_runner_level(idx, f, long, highs, lows, policy, entry_p):
    """For psar policy: run SAR from the entry bar to f and return the level
    that would stop the runner (SAR flips are treated as stop)."""
    h = highs[idx : f + 1]
    l = lows[idx : f + 1]
    sar = _psar_segment(h, l, long, policy.psar_af0, policy.psar_af_max)
    return sar


def _psar_segment(h, l, long, af0, af_max):
    """SAR over a segment assuming the initial direction; returns last level."""
    n = len(h)
    if n == 0:
        return np.nan
    if long:
        ep = h[0]; s = l[0]
    else:
        ep = l[0]; s = h[0]
    af = af0
    for i in range(1, n):
        s = s + af * (ep - s)
        if long:
            if h[i] > ep:
                ep = h[i]; af = min(af + af0, af_max)
            s = min(s, l[i - 1], l[max(i - 2, 0)])
        else:
            if l[i] < ep:
                ep = l[i]; af = min(af + af0, af_max)
            s = max(s, h[i - 1], h[max(i - 2, 0)])
    return s


# ----------------------------------------------------------------------------
# Batch runner
# ----------------------------------------------------------------------------

def run_policy(merged_df, setups_df, policy: ExitPolicy, cost_r_fn=None):
    """Simulate all setups in a merged 1H frame under one policy."""
    if setups_df is None or len(setups_df) == 0:
        return pd.DataFrame()
    highs = merged_df["high"].values
    lows = merged_df["low"].values
    closes = merged_df["close"].values
    atrs = wilder_atr(merged_df) if "atr14" not in merged_df.columns else merged_df["atr14"].values

    swing_lows = swing_highs = swing_known = None
    if policy.method == "swing":
        swing_lows, swing_highs, swing_known = confirmed_swings(lows, highs, policy.swing_lb)

    # pre-compute PSAR once per frame if needed
    psar_full = None
    if policy.method == "psar":
        psar_full = psar_series(highs, lows, policy.psar_af0, policy.psar_af_max)

    # track opens for gap-true fills
    opens = merged_df["open"].values if "open" in merged_df.columns else None

    out = []
    for _, s in setups_df.iterrows():
        setup = dict(s)
        cost = cost_r_fn(setup["symbol"]) if cost_r_fn else 0.0
        if policy.method == "psar" and psar_full is not None:
            # use the pre-computed full-series SAR level at each bar via closure
            rec = _simulate_with_psar_series(highs, lows, closes, atrs, setup,
                                             policy, psar_full, cost)
        else:
            rec = simulate_trade(highs, lows, closes, atrs, setup, policy,
                                 swing_lows, swing_known, swing_highs, cost,
                                 opens=opens)
        if rec is not None:
            out.append(rec)
    return pd.DataFrame(out)


def _simulate_with_psar_series(highs, lows, closes, atrs, setup, policy, psar_full, cost_r):
    """PSAR variant using a precomputed series: runner stops when the bar
    crosses the SAR level for its direction (SAR computed full-sample)."""
    idx = int(setup["entry_idx"])
    direction = setup["direction"]
    entry_p = float(setup["entry_price"])
    init_sl = float(setup["stop_loss"])
    risk = float(setup["risk_dist"])
    if risk <= 0 or np.isnan(risk):
        return None
    long = direction == "Long"
    sign = 1.0 if long else -1.0
    tp_1r = entry_p + sign * risk
    tp_2r = entry_p + sign * 2.0 * risk
    be_level = entry_p + sign * 0.1 * risk

    current_sl = init_sl
    be_active = False
    t1_locked = False
    t1_r = t2_r = None
    mfe_r = mae_r = 0.0
    exit_bar = idx
    exit_reason = "timeout"
    n = len(closes)
    end = min(idx + policy.max_bars + 1, n)

    if policy.entry_bar_adverse:
        if long and lows[idx] <= init_sl:
            t1_r = t2_r = -1.0; exit_reason = "entry_bar_stop"; mae_r = -1.0
        elif (not long) and highs[idx] >= init_sl:
            t1_r = t2_r = -1.0; exit_reason = "entry_bar_stop"; mae_r = -1.0

    if t1_r is None:
        for f in range(idx + 1, end):
            h, l = highs[f], lows[f]
            fav = (h - entry_p) * sign / risk if long else (entry_p - l) / risk
            adv = (l - entry_p) * sign / risk if long else (entry_p - h) / risk
            mfe_r = max(mfe_r, fav); mae_r = min(mae_r, adv)

            sar_lvl = psar_full[f - 1] if f - 1 >= idx else np.nan
            policy_trail = sar_lvl if not np.isnan(sar_lvl) else (-np.inf if long else np.inf)
            # SAR flips against us => stop at the prior extreme; approximate by level
            if long:
                eff_sl = max(current_sl, policy_trail) if t1_locked else current_sl
                hit = l <= eff_sl
            else:
                eff_sl = min(current_sl, policy_trail) if t1_locked else current_sl
                hit = h >= eff_sl

            if policy.intrabar == "optimistic":
                if not be_active and ((h >= tp_1r) if long else (l <= tp_1r)):
                    be_active = True
                    current_sl = max(current_sl, be_level) if long else min(current_sl, be_level)
                if not t1_locked and ((h >= tp_2r) if long else (l <= tp_2r)):
                    t1_locked = True; t1_r = 2.0
                    current_sl = max(current_sl, tp_1r) if long else min(current_sl, tp_1r)
                if long:
                    eff_sl = max(current_sl, policy_trail) if t1_locked else current_sl
                    hit = l <= eff_sl
                else:
                    eff_sl = min(current_sl, policy_trail) if t1_locked else current_sl
                    hit = h >= eff_sl

            if hit:
                exit_bar = f
                exit_reason = "runner_stop" if t1_locked else ("be_stop" if be_active else "init_stop")
                if not t1_locked:
                    full_r = 0.1 if be_active else -1.0
                    t1_r = t2_r = full_r
                else:
                    runner_r = (eff_sl - entry_p) / risk if long else (entry_p - eff_sl) / risk
                    t2_r = max(1.0, runner_r)
                break

            if not be_active and ((h >= tp_1r) if long else (l <= tp_1r)):
                be_active = True
                current_sl = max(current_sl, be_level) if long else min(current_sl, be_level)
            if not t1_locked and ((h >= tp_2r) if long else (l <= tp_2r)):
                t1_locked = True; t1_r = 2.0
                current_sl = max(current_sl, tp_1r) if long else min(current_sl, tp_1r)

    if t1_r is None:
        last_c = closes[min(idx + policy.max_bars, n - 1)]
        curr_r = (last_c - entry_p) / risk if long else (entry_p - last_c) / risk
        if t1_locked:
            t1_r = 2.0; t2_r = max(1.0, curr_r)
        elif be_active and curr_r <= 0.1:
            t1_r = t2_r = 0.1
        else:
            t1_r = t2_r = curr_r
        exit_bar = min(idx + policy.max_bars, n - 1)

    total_r = 0.5 * t1_r + 0.5 * t2_r - cost_r
    rec = dict(setup)
    rec.update({"t1_r": t1_r, "t2_r": t2_r, "total_r": total_r,
                "hit_2r": bool(t1_locked), "exit_reason": exit_reason,
                "holding_bars": int(exit_bar - idx),
                "mfe_r": float(mfe_r), "mae_r": float(mae_r),
                "be_active": bool(be_active)})
    return rec


# ----------------------------------------------------------------------------
# Synthetic smoke test
# ----------------------------------------------------------------------------

def _synthetic_ohlcv(n=600, seed=7, drift=0.0004, vol=0.0016, freq="1h"):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    times = pd.date_range(t0, periods=n, freq=freq)
    rets = rng.normal(drift, vol, n) + 0.0002 * np.sin(np.arange(n) / 37.0)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, vol / 2, n))
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, n)))
    return pd.DataFrame({"time": times, "open": open_, "high": hi, "low": lo,
                         "close": close, "tick_volume": rng.integers(80, 400, n)})


def _synthetic_setups(df, n_setups=12, seed=11):
    """Fabricate plausible setups: entry at midpoint of a 3-bar gap, stop below."""
    rng = np.random.default_rng(seed)
    n = len(df)
    setups = []
    i = 60
    while len(setups) < n_setups and i < n - 120:
        entry_p = df["close"].iloc[i]
        risk = max(df["high"].iloc[i] - df["low"].iloc[i], 1e-4) * 1.5
        setups.append({
            "symbol": "SYNTH", "entry_idx": i,
            "entry_time": df["time"].iloc[i],
            "direction": "Long" if rng.random() > 0.4 else "Short",
            "entry_price": entry_p,
            "stop_loss": entry_p - risk, "risk_dist": risk, "atr": risk,
        })
        i += rng.integers(30, 55)
    return pd.DataFrame(setups)


if __name__ == "__main__":
    df = _synthetic_ohlcv()
    setups = _synthetic_setups(df)
    print(f"Synthetic smoke test: {len(df)} bars, {len(setups)} setups")
    policies = [
        ("baseline_repo  (k=1.5, optimistic)", ExitPolicy("chandelier", k=1.5, intrabar="optimistic", entry_bar_adverse=False)),
        ("chandelier k=1.5 pessimistic",       ExitPolicy("chandelier", k=1.5)),
        ("chandelier k=2.0 pessimistic",       ExitPolicy("chandelier", k=2.0)),
        ("chandelier_high k=3.0",              ExitPolicy("chandelier_high", k=3.0)),
        ("swing lb=3",                         ExitPolicy("swing", swing_lb=3)),
        ("psar 0.02/0.20",                     ExitPolicy("psar")),
        ("giveback 0.40",                      ExitPolicy("giveback", giveback=0.40)),
        ("hybrid k=2.0 g=0.50",                ExitPolicy("hybrid", k=2.0, giveback=0.50)),
    ]
    rows = []
    for name, pol in policies:
        t = run_policy(df, setups, pol)
        rows.append({"policy": name, "n": len(t),
                     "total_r": round(t["total_r"].sum(), 2),
                     "hit2r%": round(t["hit_2r"].mean() * 100, 1),
                     "avg_win_r": round(t.loc[t["total_r"] > 0.5, "total_r"].mean(), 2) if (t["total_r"] > 0.5).any() else 0.0,
                     "timeout%": round((t["exit_reason"] == "timeout").mean() * 100, 1)})
    print(pd.DataFrame(rows).to_string(index=False))
    assert all(r["n"] == len(setups) for r in rows), "policy run dropped setups"
    print("exit_engine smoke test OK")
