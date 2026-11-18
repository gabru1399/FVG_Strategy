"""
meta_label.py — Meta-labeling filter for FVG entries (Lopez de Prado style).

Question 3 of the review: which lightweight feature best filters low-probability
FVG fills BEFORE entry?

Feature library (computable from the repo's OHLCV + tick_volume parquets):
  Geometry / context:
    gap_atr        : FVG gap size / ATR14
    risk_atr       : stop distance / ATR14
    displacement   : body ratio of the momentum-leg candles (mean of the two)
    rvol           : relative volume of trigger candle vs 20-bar SMA
    rvol_leg       : relative volume of the second displacement candle
    swing_dist_atr : breakout distance beyond 10-bar swing, in ATR
    htf_dist_atr   : distance from 4H EMA50 at entry, in ATR
    atr_regime     : ATR14 / ATR100 (volatility expansion regime)
    range_pos      : entry price position in the last 50-bar range [0,1]
  Persistence / microstructure (bar-based proxies):
    er20           : Kaufman efficiency ratio, 20 bars (trendiness)
    hurst_rs       : rescaled-range (R/S) Hurst estimate over 64 bars
    vol_trend      : tick-volume 5-bar MA / 50-bar MA
    ret_skew5      : skew of last 5 bar returns (impulse asymmetry)
    session_hour   : UTC hour bucketed (London / overlap / NY / off)
    day_of_week    : weekday one-hot reduced to sin/cos

  NOT available in this dataset (flagged for future collection):
    bid/ask imbalance, true order-flow delta, tick entropy at the touch.
    The framework accepts them as extra columns when present.

Pipeline:
  1. build_features(merged, setups) -> feature frame aligned to setups
  2. make_labels(trades, mode) -> binary label (default: total_r > 0.25)
  3. purged_kfold_split(times, n_splits, embargo_bars) -> train/test indices
     (time-ordered folds with purge gap + embargo, no shuffling)
  4. evaluate: LogisticRegression (L2) per fold; metrics = AUC, rank-IC
     (Spearman between predicted prob and realized R), and a lift table:
     expected R by predicted-probability decile.
  5. Simple threshold policy: skip trades below prob quantile q; report
     resulting expectancy / trade count / total R.

Pure pandas/numpy/sklearn. Deterministic.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ----------------------------------------------------------------------------
# 1. Features
# ----------------------------------------------------------------------------

def kaufman_er(closes, period=20):
    """Efficiency ratio |net change| / sum|diff| over period."""
    c = pd.Series(closes)
    change = (c - c.shift(period)).abs()
    vol = c.diff().abs().rolling(period).sum()
    return (change / vol.replace(0, np.nan)).values


def hurst_rs(prices, window=64):
    """Sliding rescaled-range Hurst exponent.

    For each window: H = log(R/S) / log(n) on the aggregated series
    (simplified single-scale estimate — fast, adequate as a ranking feature).
    """
    p = pd.Series(prices, dtype=float)
    out = np.full(len(p), np.nan)
    logp = np.log(p.values)
    for i in range(window, len(p)):
        w = logp[i - window:i]
        # detrend by subtracting linear fit endpoints (classic R/S on increments)
        inc = np.diff(w)
        if np.allclose(inc, 0):
            continue
        mean_cum = np.cumsum(inc - inc.mean())
        R = mean_cum.max() - mean_cum.min()
        S = inc.std()
        if S > 0 and R > 0:
            out[i] = np.log(R / S) / np.log(len(inc))
    return out


def build_features(merged: pd.DataFrame, setups: pd.DataFrame) -> pd.DataFrame:
    """Feature frame indexed like setups (one row per trade)."""
    f = pd.DataFrame(index=setups.index)

    highs = merged["high"].values
    lows = merged["low"].values
    closes = merged["close"].values
    opens = merged["open"].values
    atrs = merged["atr14"].values if "atr14" in merged.columns else None
    times = pd.to_datetime(merged["time"])
    vol = merged["tick_volume"].values.astype(float) if "tick_volume" in merged.columns \
        else np.ones(len(merged))

    # rolling volume SMA20
    vol_sma = pd.Series(vol).rolling(20).mean().values
    vol_sma5 = pd.Series(vol).rolling(5).mean().values
    vol_sma50 = pd.Series(vol).rolling(50).mean().values

    er20 = kaufman_er(closes, 20)
    hurst = hurst_rs(closes, 64)

    # ATR regime & 50-bar range position
    atr100 = pd.Series(atrs).rolling(100).median().values if atrs is not None else None
    roll50_hi = pd.Series(highs).rolling(50).max().values
    roll50_lo = pd.Series(lows).rolling(50).min().values

    # HTF distance (4H EMA50 was merged as ema50 in extract_setups)
    htf_ema50 = merged["ema50"].values if "ema50" in merged.columns else None

    # body ratio arrays
    body = np.abs(closes - opens)
    rng = np.maximum(highs - lows, 1e-12)
    body_ratio = body / rng

    # swing distances
    swing_hi = pd.Series(highs).shift(2).rolling(10).max().values
    swing_lo = pd.Series(lows).shift(2).rolling(10).min().values

    hours = times.dt.hour.values
    dow = times.dt.dayofweek.values

    for i, s in setups.iterrows():
        idx = int(s["entry_idx"])
        atr = atrs[idx] if atrs is not None and not np.isnan(atrs[idx]) else np.nanmax(rng[idx-20:idx]) if idx >= 20 else np.nan
        direction = s["direction"]
        long = direction == "Long"

        # gap size in ATR: |entry - structural boundary| and the FVG width
        risk = float(s["risk_dist"])
        raw = abs(float(s["entry_price"]) - float(s["stop_loss"]))

        feats = {
            "gap_atr": raw / atr if atr and not np.isnan(atr) else np.nan,
            "risk_atr": risk / atr if atr and not np.isnan(atr) else np.nan,
            "displacement": float(np.mean([body_ratio[idx - 1], body_ratio[idx]])),
            "rvol": vol[idx] / vol_sma[idx - 1] if vol_sma[idx - 1] and vol_sma[idx - 1] > 0 else np.nan,
            "rvol_leg": vol[idx - 1] / vol_sma[idx - 1] if vol_sma[idx - 1] and vol_sma[idx - 1] > 0 else np.nan,
            "swing_dist_atr": ((closes[idx] - swing_hi[idx]) / atr) if long else ((swing_lo[idx] - closes[idx]) / atr)
                               if not np.isnan(swing_hi[idx]) and not np.isnan(swing_lo[idx]) else np.nan,
            "er20": er20[idx],
            "hurst_rs": hurst[idx],
            "vol_trend": vol_sma5[idx] / vol_sma50[idx] if vol_sma50[idx] and vol_sma50[idx] > 0 else np.nan,
            "atr_regime": atrs[idx] / atr100[idx] if atr100 is not None and not np.isnan(atr100[idx]) else np.nan,
            "range_pos": (closes[idx] - roll50_lo[idx]) / (roll50_hi[idx] - roll50_lo[idx])
                         if not np.isnan(roll50_hi[idx]) and roll50_hi[idx] > roll50_lo[idx] else np.nan,
            "hour_sin": np.sin(2 * np.pi * hours[idx] / 24),
            "hour_cos": np.cos(2 * np.pi * hours[idx] / 24),
            "dow_sin": np.sin(2 * np.pi * dow[idx] / 5),
            "dow_cos": np.cos(2 * np.pi * dow[idx] / 5),
        }
        if htf_ema50 is not None and not np.isnan(htf_ema50[idx]):
            feats["htf_dist_atr"] = (closes[idx] - htf_ema50[idx]) / atr if (long and atr) else \
                                    (htf_ema50[idx] - closes[idx]) / atr
        else:
            feats["htf_dist_atr"] = np.nan
        for k, v in feats.items():
            f.loc[i, k] = v

    # recent return skew (last 5 bars, signed by direction)
    rets = pd.Series(closes).pct_change()
    for i, s in setups.iterrows():
        idx = int(s["entry_idx"])
        r5 = rets.iloc[idx - 5:idx]
        sign = 1.0 if s["direction"] == "Long" else -1.0
        f.loc[i, "ret_skew5"] = sign * float(r5.skew()) if len(r5) == 5 and not np.isnan(r5.skew()) else np.nan

    f["direction_long"] = (setups["direction"] == "Long").astype(float).values
    return f


# ----------------------------------------------------------------------------
# 2. Labels
# ----------------------------------------------------------------------------

def make_labels(trades: pd.DataFrame, mode="win", threshold=0.25):
    """Binary labels from simulated outcomes (meta-labels).

    mode='win'       : 1 if total_r > threshold (a 'good' trade)
    mode='not_loss'  : 1 if total_r > -0.05 (avoid full losses)
    """
    r = trades["total_r"].values
    if mode == "win":
        return (r > threshold).astype(int)
    elif mode == "not_loss":
        return (r > -0.05).astype(int)
    raise ValueError(mode)


# ----------------------------------------------------------------------------
# 3. Purged, embargoed time-series CV
# ----------------------------------------------------------------------------

def purged_kfold_splits(times, n_splits=5, embargo_bars=24):
    """Time-ordered K-fold with purge + embargo.

    Returns list of (train_idx, test_idx) arrays (positional).
    Purge: drop training samples within `embargo_bars` bars (by timestamp
    distance) of any test sample, BEFORE and AFTER the test block.
    """
    times = pd.to_datetime(pd.Series(times).reset_index(drop=True))
    n = len(times)
    order = np.argsort(times.values)  # should already be sorted; be safe
    folds = []
    bounds = np.linspace(0, n, n_splits + 1, dtype=int)
    emb = pd.Timedelta(hours=embargo_bars)  # bars are 1H
    for k in range(n_splits):
        test_pos = np.arange(bounds[k], bounds[k + 1])
        if len(test_pos) == 0:
            continue
        t_lo = times.iloc[order[bounds[k]]]
        t_hi = times.iloc[order[min(bounds[k + 1] - 1, n - 1)]]
        train_mask = (times < (t_lo - emb)) | (times > (t_hi + emb))
        train_pos = np.where(train_mask.values)[0]
        if len(train_pos) < 30 or len(test_pos) < 10:
            continue
        folds.append((train_pos, test_pos))
    return folds


# ----------------------------------------------------------------------------
# 4. Evaluation
# ----------------------------------------------------------------------------

def evaluate_meta_labeler(X: pd.DataFrame, y: np.ndarray, r: np.ndarray,
                          times, n_splits=5, embargo_bars=24, seed=42):
    """Purged-CV logistic regression with AUC + rank-IC + lift table.

    X     : feature frame (rows = trades)
    y     : binary labels
    r     : realized total_r per trade (for IC and lift)
    times : entry times per trade (for purging)
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import roc_auc_score
    from scipy.stats import spearmanr

    cols = [c for c in X.columns if X[c].notna().mean() > 0.8]
    Xc = X[cols].fillna(X[cols].median())
    Xc = Xc.astype(float)

    preds = np.full(len(Xc), np.nan)
    coefs = []
    folds = purged_kfold_splits(times, n_splits=n_splits, embargo_bars=embargo_bars)
    for train_pos, test_pos in folds:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
        ])
        model.fit(Xc.iloc[train_pos], y[train_pos])
        preds[test_pos] = model.predict_proba(Xc.iloc[test_pos])[:, 1]
        coefs.append(model.named_steps["lr"].coef_[0])

    mask = ~np.isnan(preds)
    out = {
        "n_folds": len(folds),
        "n_scored": int(mask.sum()),
        "auc": float(roc_auc_score(y[mask], preds[mask])) if len(np.unique(y[mask])) > 1 else np.nan,
        "rank_ic": float(spearmanr(preds[mask], r[mask]).statistic),
        "feature_names": cols,
        "mean_coef": dict(zip(cols, np.mean(coefs, axis=0).round(3).tolist())),
    }

    # lift table: expected R by predicted-probability quintile
    q = pd.qcut(pd.Series(preds[mask]), 5, labels=False, duplicates="drop")
    lift = []
    for i in sorted(pd.Series(q).unique()):
        sel = q == i
        lift.append({
            "quintile": int(i) + 1,
            "mean_prob": round(float(np.mean(preds[mask][sel.values])), 3),
            "mean_r": round(float(np.mean(r[mask][sel.values])), 3),
            "win_rate": round(float(np.mean(y[mask][sel.values] == 1)), 3),
            "n": int(sel.sum()),
        })
    out["lift"] = lift

    # threshold policy: skip lowest-q quintile trades
    base_exp = float(np.mean(r[mask]))
    skip_exp = float(np.mean(r[mask][q.values > 0])) if (q.values > 0).any() else base_exp
    out["baseline_expectancy_r"] = round(base_exp, 4)
    out["skip_bottom_quintile_expectancy_r"] = round(skip_exp, 4)
    out["skip_trades_pct"] = round(float(np.mean(q.values == 0)) * 100, 1)
    return out, preds


def single_feature_ics(X: pd.DataFrame, r: np.ndarray, y=None):
    """Univariate rank-IC of each feature vs realized R (no CV needed for a
    first-pass ranking) + AUC vs the binary label."""
    from sklearn.metrics import roc_auc_score
    rows = []
    yy = y if y is not None else (r > 0.25).astype(int)
    for c in X.columns:
        s = X[c]
        m = s.notna().values
        if m.sum() < 50:
            continue
        ic = spearmanr(s.values[m], r[m]).statistic
        try:
            auc = roc_auc_score(yy[m], s.values[m])
        except ValueError:
            auc = np.nan
        rows.append({"feature": c, "rank_ic": round(float(ic), 4),
                     "auc": round(float(auc), 4) if not np.isnan(auc) else None})
    return pd.DataFrame(rows).sort_values("rank_ic", key=lambda s: s.abs(), ascending=False)


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(5)
    n = 1500
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    times = pd.date_range(t0, periods=n, freq="1h")
    rets = rng.normal(0.0001, 0.0015, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.roll(close, 1) * (1 + rng.normal(0, 0.0005, n)); open_[0] = 100
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.0006, n)))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.0006, n)))
    vol = rng.integers(90, 400, n).astype(float)
    df = pd.DataFrame({"time": times, "open": open_, "high": hi, "low": lo,
                       "close": close, "tick_volume": vol})
    from exit_engine import wilder_atr
    df["atr14"] = wilder_atr(df)
    df["ema50"] = df["close"].ewm(span=50).mean()

    # synthetic setups
    setups = pd.DataFrame([
        {"entry_idx": i, "direction": "Long" if rng.random() > 0.5 else "Short",
         "entry_price": close[i], "stop_loss": close[i] - 0.5, "risk_dist": 0.5}
        for i in range(100, n - 100, 7)
    ])
    setups["entry_time"] = [times[i] for i in setups["entry_idx"]]
    setups["symbol"] = "SYNTH"

    X = build_features(df, setups)
    print("Feature frame:", X.shape)
    print(X.head(3).round(3).to_string())

    # synthetic outcome correlated with rvol (so IC should find it)
    r = 0.3 * X["rvol"].fillna(1).values * rng.normal(1, 0.3, len(X)) + rng.normal(0, 0.4, len(X))
    y = (r > 0.25).astype(int)
    res, preds = evaluate_meta_labeler(X, y, r, setups["entry_time"].values)
    print(f"\nPurged-CV: folds={res['n_folds']} AUC={res['auc']:.3f} rank_IC={res['rank_ic']:.3f}")
    print("Lift table:")
    print(pd.DataFrame(res["lift"]).to_string(index=False))
    ics = single_feature_ics(X, r)
    print("\nUnivariate ICs (top 5):")
    print(ics.head(5).to_string(index=False))
    print("meta_label smoke test OK")
