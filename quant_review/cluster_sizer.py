"""
cluster_sizer.py — Correlation-clustered risk budgeting for a small account.

Problem: the 10-asset basket mixes 4 equity indices + 2 cryptos that correlate
to ~1.0 in liquidity shocks. Fixed $10/trade per asset means a 2020/2022-style
regime hits all six simultaneously -> clustered drawdown.

Solution stack (in increasing sophistication):
  1. EWMA correlation matrix (halflife ~ 20 days) on daily returns
  2. Correlation clustering: single-linkage on distance d = sqrt(0.5*(1-rho))
     (a lightweight HRP front-end; full recursive bisection optional)
  3. Budget: cluster heat caps. Total portfolio heat H_total (sum of open
     position risks) capped; each cluster capped at H_cluster; within a
     cluster, capital split inverse-volatility.
  4. Optional correlation floor in stress: when average pairwise corr within
     a cluster exceeds rho_stress, the cluster budget is scaled by
     sqrt(target_cluster_heat / realized_cluster_heat).

Usage on a $2,500 account:
  heat_total   = 0.02 .. 0.03  (2-3% aggregate open risk; $50-75)
  heat_cluster = 0.01 .. 0.015 (1-1.5% per cluster; $25-37.5)
  per-trade risk = heat_slot; with 1 slot per asset the base unit is
  $10 = 0.40%, and simultaneous entries in one cluster shrink geometrically:
  entry n in the same cluster within the co-movement window trades at
  $10 * rho_decay^n (rho_decay ~ 0.6).

Pure pandas/numpy. Works on any wide DataFrame of returns (assets x dates).
"""

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# 1. EWMA correlation matrix
# ----------------------------------------------------------------------------

def ewma_corr(returns: pd.DataFrame, halflife=20, min_periods=30):
    """Exponentially-weighted correlation matrix.

    returns : DataFrame, columns = assets, index = dates, values = simple returns.
    halflife: decay halflife in observations (days).
    """
    spans = {c: returns[c].ewm(halflife=halflife, min_periods=min_periods).corr()
             for c in returns.columns}
    # ewm(...).corr() pairwise on the full frame:
    ew = returns.ewm(halflife=halflife, min_periods=min_periods).corr()
    # take the last cross-section
    last_date = ew.index.get_level_values(0)[-1]
    corr = ew.loc[last_date]
    if isinstance(corr, pd.Series):
        corr = corr.unstack()
    return corr


def corr_history(returns: pd.DataFrame, halflife=20, min_periods=30, freq="ME"):
    """Series of EWMA correlation snapshots (dates -> corr matrices), monthly."""
    ew = returns.ewm(halflife=halflife, min_periods=min_periods).corr()
    dates = ew.index.get_level_values(0)
    snaps = {}
    for d in pd.DatetimeIndex(dates.unique()).to_period(freq).unique():
        last_in_period = dates[dates.to_period(freq) == d][-1]
        c = ew.loc[last_in_period]
        snaps[str(d)] = c.unstack() if isinstance(c, pd.Series) else c
    return snaps


# ----------------------------------------------------------------------------
# 2. Correlation clustering (single linkage on sqrt-distance)
# ----------------------------------------------------------------------------

def cluster_assets(corr: pd.DataFrame, threshold=0.55):
    """Single-linkage agglomerative clustering on d = sqrt(0.5*(1 - rho)).

    threshold: max distance within a cluster (0.47 ~ rho 0.55).
    Returns dict cluster_name -> list of assets, plus assignment Series.
    """
    assets = list(corr.columns)
    n = len(assets)
    d = np.sqrt(0.5 * (1.0 - corr.values.clip(-1, 1)))
    np.fill_diagonal(d, 0.0)

    # union-find single linkage below threshold
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if d[i, j] <= threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(assets[i])

    # order clusters by size, name them C1, C2, ...
    ordered = sorted(groups.values(), key=lambda g: (-len(g), g[0]))
    clusters = {f"C{ i+1 }": sorted(g) for i, g in enumerate(ordered)}
    assign = {a: name for name, lst in clusters.items() for a in lst}
    return clusters, pd.Series(assign, name="cluster")


def hrp_weights(corr: pd.DataFrame, cov: pd.DataFrame = None):
    """Full Hierarchical Risk Parity (Lopez de Prado 2016):
    scipy-free single-linkage ordering + recursive bisection with
    inverse-variance allocation. Returns weight Series.

    cov defaults to the correlation-implied unit diagonal if not supplied.
    """
    assets = list(corr.columns)
    n = len(assets)
    if cov is None:
        cov = pd.DataFrame(np.eye(n), index=assets, columns=assets)
    d = np.sqrt(0.5 * (1.0 - corr.values.clip(-1, 1)))
    np.fill_diagonal(d, 0.0)

    # single-linkage order via successive nearest-neighbor merging
    # (equivalent to scipy linkage 'single' on this distance)
    order = _single_linkage_order(d)

    # recursive bisection on the ordered list
    w = pd.Series(1.0, index=range(n))
    items = [list(range(n))]
    while items:
        new_items = []
        for cluster_idx in items:
            if len(cluster_idx) <= 1:
                continue
            split = len(cluster_idx) // 2
            left, right = cluster_idx[:split], cluster_idx[split:]
            var_l = _cluster_var(left, cov.values, order)
            var_r = _cluster_var(right, cov.values, order)
            alpha = 1.0 - var_l / (var_l + var_r)
            for i in left:
                w[order[i]] *= alpha
            for i in right:
                w[order[i]] *= 1.0 - alpha
            new_items.extend([left, right])
        items = new_items

    out = pd.Series(0.0, index=assets)
    for i, a in enumerate(order):
        out[a] = w[i]
    return out / out.sum()


def _single_linkage_order(d):
    """Return asset indices in quasi-diagonal single-linkage order."""
    n = d.shape[0]
    clusters = [[i] for i in range(n)]
    d = d.copy()
    np.fill_diagonal(d, np.inf)
    while len(clusters) > 1:
        # find the two closest clusters (min pairwise distance)
        best = (np.inf, 0, 1)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                dd = d[np.ix_(clusters[a], clusters[b])].min()
                if dd < best[0]:
                    best = (dd, a, b)
        _, a, b = best
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    return clusters[0]


def _cluster_var(idx_list, cov, order):
    """Inverse-variance portfolio variance of the sub-cluster."""
    cols = [order[i] for i in idx_list]
    sub = cov[np.ix_(cols, cols)]
    iv = 1.0 / np.diag(sub)
    iv /= iv.sum()
    return float(iv @ sub @ iv)


# ----------------------------------------------------------------------------
# 3. Heat-budget position sizer
# ----------------------------------------------------------------------------

class HeatBudgetSizer:
    """Slot-based heat budget with cluster caps and geometric entry decay.

    Parameters
    ----------
    equity            : current account equity (dollars)
    heat_total_pct    : aggregate open-risk budget, fraction of equity
    heat_cluster_pct  : per-cluster open-risk budget
    base_risk_pct     : risk of the FIRST slot in a fresh cluster
    decay             : multiplier applied per additional concurrent entry in
                        the same cluster within the co-movement window
    window_bars       : bars defining "concurrent" (default 24 x 1H bars = 1 day)
    """

    def __init__(self, equity=2500.0, heat_total_pct=0.03, heat_cluster_pct=0.015,
                 base_risk_pct=0.004, decay=0.6, window_bars=24):
        self.equity = equity
        self.heat_total = equity * heat_total_pct
        self.heat_cluster = equity * heat_cluster_pct
        self.base_risk = equity * base_risk_pct
        self.decay = decay
        self.window = window_bars
        # state: cluster -> list of (entry_bar, risk_usd)
        self.open_slots = {}

    def register_cluster_map(self, assignment: pd.Series):
        self.asset_cluster = assignment.to_dict()

    def cluster_heat(self, now_bar):
        """Realized open risk per cluster, expiring entries older than window."""
        heat = {}
        for c, slots in self.open_slots.items():
            live = [r for (b, r) in slots if now_bar - b <= self.window]
            self.open_slots[c] = [(b, r) for (b, r) in slots if now_bar - b <= self.window]
            heat[c] = sum(live)
        return heat

    def size_position(self, asset, now_bar, vol_ratio=1.0):
        """Return risk (dollars) for a new entry, or 0.0 if budget exhausted.

        vol_ratio : current ATR / median ATR of the asset — scales risk down in
                    high-vol regimes (inverse-vol flavor without a full cov).
        """
        c = self.asset_cluster.get(asset, "C1")
        heat = self.cluster_heat(now_bar)
        open_c = heat.get(c, 0.0)
        open_all = sum(heat.values())

        n_concurrent = len([1 for (b, r) in self.open_slots.get(c, [])])
        risk = self.base_risk * (self.decay ** n_concurrent) / max(vol_ratio, 1.0)

        # cluster cap
        if open_c + risk > self.heat_cluster:
            risk = self.heat_cluster - open_c
        # global cap
        if open_all + risk > self.heat_total:
            risk = self.heat_total - open_all
        if risk < self.base_risk * 0.25:
            return 0.0  # too small to bother — skip the trade
        self.open_slots.setdefault(c, []).append((now_bar, risk))
        return risk


def stress_correlation_scan(returns: pd.DataFrame, halflife=20, top_pct=5):
    """Find the worst correlation regime: for each date, average pairwise EWMA
    correlation of the OFF-DIAGONAL. Returns the top_pct most-correlated dates
    and the average corr matrix in that tail."""
    ew = returns.ewm(halflife=halflife, min_periods=30).corr()
    dates = ew.index.get_level_values(0).unique()
    avg = {}
    for d in dates:
        c = ew.loc[d]
        if isinstance(c, pd.Series):
            c = c.unstack()
        m = c.values
        n = m.shape[0]
        if n < 2 or np.isnan(m).any():
            continue
        off = (m.sum() - np.trace(m)) / (n * n - n)
        avg[d] = off
    s = pd.Series(avg).sort_values(ascending=False)
    k = max(1, int(len(s) * top_pct / 100))
    tail_dates = s.index[:k]
    # average corr matrix over tail dates
    acc = None
    cnt = 0
    for d in tail_dates:
        c = ew.loc[d]
        if isinstance(c, pd.Series):
            c = c.unstack()
        acc = c.values if acc is None else acc + c.values
        cnt += 1
    return s, tail_dates, (acc / cnt if cnt else None)


# ----------------------------------------------------------------------------
# Smoke test on synthetic correlated returns
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(3)
    n = 500
    # 3 latent factors: equity, crypto, fx
    f1 = rng.normal(0, 0.01, n)          # equity factor
    f2 = rng.normal(0, 0.02, n)          # crypto factor
    f3 = rng.normal(0, 0.004, n)         # fx factor
    data = {
        "NAS100": f1 * 1.0 + rng.normal(0, 0.004, n),
        "UK100":  f1 * 0.9 + rng.normal(0, 0.004, n),
        "AU200":  f1 * 0.8 + rng.normal(0, 0.004, n),
        "JP225":  f1 * 0.7 + rng.normal(0, 0.004, n),
        "BTCUSD": f2 * 1.0 + f1 * 0.3 + rng.normal(0, 0.008, n),
        "SOLUSD": f2 * 1.2 + f1 * 0.3 + rng.normal(0, 0.010, n),
        "EURUSD": f3 * 1.0 + rng.normal(0, 0.002, n),
        "USDJPY": f3 * -0.8 + rng.normal(0, 0.002, n),
        "USDCHF": f3 * -0.6 + rng.normal(0, 0.002, n),
        "XAGUSD": f1 * 0.2 + rng.normal(0, 0.006, n),
    }
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rets = pd.DataFrame(data, index=idx)

    corr = ewma_corr(rets)
    print("EWMA corr (last snapshot), rounded:")
    print(corr.round(2).to_string())

    clusters, assign = cluster_assets(corr, threshold=0.55)
    print("\nClusters:")
    for name, lst in clusters.items():
        print(f"  {name}: {lst}")

    w = hrp_weights(corr)
    print("\nHRP weights:")
    print((w * 100).round(1).to_string())

    sizer = HeatBudgetSizer(equity=2500.0)
    sizer.register_cluster_map(assign)
    # simulate 6 simultaneous entries in the equity cluster
    print("\nHeat-budget sizing, 6 same-day equity-cluster entries:")
    for i, a in enumerate(["NAS100", "UK100", "AU200", "JP225", "XAGUSD", "BTCUSD"]):
        r = sizer.size_position(a, now_bar=10)
        print(f"  entry {i+1} {a:7s} -> risk ${r:.2f}")
    r = sizer.size_position("EURUSD", now_bar=10)
    print(f"  entry 7 EURUSD  -> risk ${r:.2f} (fresh FX cluster)")

    s, tail, m = stress_correlation_scan(rets)
    print(f"\nStress scan: top corr dates {list(str(d.date()) for d in tail[:3])}, "
          f"avg pairwise corr in tail = {np.nanmean(m):.2f}" if m is not None else "no tail")
    print("cluster_sizer smoke test OK")
