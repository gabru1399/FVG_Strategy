"""
validity.py — Statistical validity toolkit for the FVG strategy review.

1. Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014): adjusts the observed
   Sharpe for multiple testing (number of trials N, trial variance) and for
   non-normality (skew, kurtosis) of returns.
2. PBO (probability of backtest overfitting) via CSCV (combinatorially
   symmetric cross-validation) on a matrix of strategy configurations.
3. Walk-forward protocol generator: anchored/rolling windows with the exact
   index boundaries, so parameter choices inside each fold use only past data.
4. Trade-level bootstrap: drawdown and expectancy confidence intervals via
   stationary-ish block bootstrap on the trade sequence.

Pure numpy/pandas. Deterministic (seeded).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis


# ----------------------------------------------------------------------------
# 1. Deflated Sharpe Ratio
# ----------------------------------------------------------------------------

def sharpe_stats(returns, periods_per_year=252):
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    mu, sd = r.mean(), r.std(ddof=1)
    if sd == 0:
        return dict(sharpe=0.0, skew=0.0, kurt=0.0, n=len(r))
    ann = np.sqrt(periods_per_year)
    return dict(sharpe=float(mu / sd * ann), skew=float(skew(r)),
                kurt=float(kurtosis(r, fisher=False)), n=len(r))


def expected_max_sharpe(n_trials, var_trials, sharpe_benchmark=0.0):
    """E[max Z] over n_trials zero-mean trials with variance var_trials
    (the BLdP 'deflation' term)."""
    if n_trials <= 1:
        return 0.0
    emc = 0.5772156649015329  # Euler-Mascheroni
    gamma = emc - np.pi**2 / 6.0 * (1.0 / np.log(n_trials))  # not used; approx below
    # BLdP Eq: E[max] ≈ (1-γ)Φ^{-1}(1-1/N) + γΦ^{-1}(1-1/(N·e))
    if n_trials < 5:
        return 0.0
    z1 = norm.ppf(1.0 - 1.0 / n_trials)
    z2 = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_trials) * ((1.0 - emc) * z1 + emc * z2))


def deflated_sharpe(returns, n_trials, var_trials, periods_per_year=252,
                    skew_override=None, kurt_override=None):
    """Return dict with observed SR (annualized), DSR p-value, and DSR-adjusted SR.

    n_trials, var_trials: number and cross-sectional variance of the Sharpe
    ratios of ALL strategy variants tried during research (not just the kept
    one) — the multiple-testing bookkeeping.
    """
    st = sharpe_stats(returns, periods_per_year)
    n = st["n"]
    s = st["sharpe"]
    if n < 20 or s == 0.0:
        return {**st, "dsr_p": np.nan, "sr_star": np.nan, "verdict": "insufficient_data"}

    g1 = skew_override if skew_override is not None else st["skew"]
    g2 = kurt_override if kurt_override is not None else st["kurt"]

    # PSR (probabilistic Sharpe ratio) vs the deflated benchmark SR*
    e_max = expected_max_sharpe(n_trials, var_trials)
    # benchmark in ANNUALIZED space, then convert to per-period for the test
    sr_star_ann = e_max  # trials assumed annualized-comparable
    sr_star_per = sr_star_ann / np.sqrt(periods_per_year)
    s_per = s / np.sqrt(periods_per_year)

    # SE of SR with non-normality correction (Mertens/Lo):
    se = np.sqrt(max(1e-12, (1.0 - g1 * s_per + (g2 - 1.0) / 4.0 * s_per**2) / (n - 1)))
    psr = norm.cdf((s_per - sr_star_per) / se)
    return {**st, "dsr_p": float(psr), "sr_star_ann": float(sr_star_ann),
            "verdict": "PASS" if psr > 0.95 else ("WEAK" if psr > 0.50 else "FAIL")}


# ----------------------------------------------------------------------------
# 2. PBO via CSCV
# ----------------------------------------------------------------------------

def pbo_cscv(strategy_matrix: pd.DataFrame, n_partitions=8, metric="sharpe"):
    """Combinatorially symmetric cross-validation PBO.

    strategy_matrix : DataFrame, rows = time periods (chronological),
                      columns = strategy configurations, values = period returns.
    Returns dict: pbo (fraction of CSCV combinations where the IS-best config
    underperforms the median OOS), and per-config OOS rank distribution.
    """
    M = strategy_matrix.dropna().values
    T, N = M.shape
    if T < n_partitions * 4 or N < 2:
        return {"pbo": np.nan, "reason": "matrix too small"}

    parts = np.array_split(np.arange(T), n_partitions)
    combs = list(_combinations(n_partitions, n_partitions // 2))
    omega = []  # OOS ranks of IS-best configs (relative performance)
    for train_parts in combs:
        test_parts = [p for p in range(n_partitions) if p not in train_parts]
        tr_idx = np.concatenate([parts[p] for p in train_parts])
        te_idx = np.concatenate([parts[p] for p in test_parts])
        if len(tr_idx) == 0 or len(te_idx) == 0:
            continue
        if metric == "sharpe":
            is_score = M[tr_idx].mean(axis=0) / (M[tr_idx].std(axis=0) + 1e-12)
            oos_score = M[te_idx].mean(axis=0) / (M[te_idx].std(axis=0) + 1e-12)
        else:  # mean return
            is_score = M[tr_idx].mean(axis=0)
            oos_score = M[te_idx].mean(axis=0)
        best_is = np.argmax(is_score)
        # relative rank of IS-best in the OOS ranking (0=best)
        oos_rank = np.argsort(np.argsort(-oos_score))
        rel = oos_rank[best_is] / (N - 1)  # in (0,1)
        omega.append(rel)
    omega = np.array(omega)
    pbo = float(np.mean(omega > 0.5))  # IS-best lands in bottom half OOS
    return {
        "pbo": pbo,
        "n_combinations": len(omega),
        "median_oos_rel_rank_of_is_best": float(np.median(omega)),
        "interpretation": "pbo>0.5 => selection is overfit more often than not",
    }


def _combinations(n, k):
    from itertools import combinations
    return combinations(range(n), k)


# ----------------------------------------------------------------------------
# 3. Walk-forward protocol
# ----------------------------------------------------------------------------

def walk_forward_windows(n, train_bars, test_bars, anchored=False):
    """Yield (train_start, train_end, test_start, test_end) index tuples.

    n         : total number of bars (or trades) in chronological order
    anchored  : True = expanding window (train always starts at 0)
    """
    starts = range(0, n - train_bars, test_bars)
    out = []
    for s in starts:
        tr_s = 0 if anchored else s
        tr_e = s + train_bars
        te_e = min(tr_e + test_bars, n)
        if te_e <= tr_e:
            break
        out.append((tr_s, tr_e, tr_e, te_e))
    return out


# ----------------------------------------------------------------------------
# 4. Block bootstrap on trade sequences
# ----------------------------------------------------------------------------

def block_bootstrap_trade_r(r_values, n_boot=2000, block=10, seed=7,
                            stat="expectancy", dd_units="R"):
    """Circular block bootstrap over the trade R sequence.

    Returns dict with bootstrap distribution of the statistic:
      expectancy, total_r, max_drawdown (in R or $ via dd_units_per_r).
    """
    rng = np.random.default_rng(seed)
    r = np.asarray(r_values, dtype=float)
    r = r[~np.isnan(r)]
    n = len(r)
    if n < block * 3:
        return {"error": "too few trades"}

    n_blocks = int(np.ceil(n / block))
    stats = {"expectancy": [], "total_r": [], "max_dd_r": []}
    for _ in range(n_boot):
        idx = (rng.integers(0, n, n_blocks)[:, None] + np.arange(block)[None, :]).ravel()[:n]
        seq = r[idx % n]
        stats["expectancy"].append(seq.mean())
        stats["total_r"].append(seq.sum())
        eq = np.cumsum(seq)
        peak = np.maximum.accumulate(eq)
        stats["max_dd_r"].append(float((eq - peak).min()))
    out = {}
    for k, v in stats.items():
        v = np.array(v)
        out[k] = {
            "mean": round(float(v.mean()), 3),
            "p5": round(float(np.percentile(v, 5)), 3),
            "p50": round(float(np.percentile(v, 50)), 3),
            "p95": round(float(np.percentile(v, 95)), 3),
        }
    out["prob_total_r_negative"] = round(float(np.mean(np.array(stats["total_r"]) < 0)), 4)
    return out


# ----------------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(11)

    # 1) DSR: 300 daily returns, SR ~1.2, tested among 20 trials
    rets = rng.normal(0.0008, 0.01, 300) + 0.002 * np.sin(np.arange(300) / 40)
    dsr = deflated_sharpe(rets, n_trials=20, var_trials=0.25)
    print("DSR smoke:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in dsr.items()})

    # 2) PBO: 6 configs, one has true edge, others noise
    T = 480
    true_edge = rng.normal(0.0010, 0.008, T)
    M = pd.DataFrame({f"cfg{i}": rng.normal(0.0002, 0.008, T) for i in range(5)})
    M["cfg_true"] = true_edge + rng.normal(0, 0.002, T)
    pbo = pbo_cscv(M, n_partitions=8)
    print("\nPBO smoke:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in pbo.items()})

    # 3) walk-forward
    wf = walk_forward_windows(1000, train_bars=600, test_bars=100)
    print(f"\nWalk-forward smoke: {len(wf)} windows, first={wf[0]}, last={wf[-1]}")

    # 4) bootstrap
    rs = np.concatenate([rng.normal(-1, 0.2, 300), rng.normal(0.9, 1.6, 700)])
    boot = block_bootstrap_trade_r(rs, n_boot=1000)
    print("\nBootstrap smoke:")
    for k, v in boot.items():
        print(f"  {k}: {v}")
    print("validity smoke test OK")
