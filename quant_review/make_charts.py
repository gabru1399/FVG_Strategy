"""
make_charts.py — Generate all report figures from the empirical results.
Follows typesetting/charts.md: no top/right spines, dashed grid 20% opacity,
legends outside plot area, first/last/max/min labels only, palette colors.
"""

# ── portable path bootstrap (added for repo distribution) ──────────────────
import os, sys
from pathlib import Path
PACKAGE_DIR = Path(__file__).resolve().parent   # <repo>/quant_review
REPO_ROOT = PACKAGE_DIR.parent                  # repository root

import os, sys, json
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.font_manager as fm
for _f in ('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf',
           'C:/Windows/Fonts/msyh.ttc'):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
for _f in ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
           'C:/Windows/Fonts/arial.ttf'):
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'Microsoft YaHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, str(PACKAGE_DIR))
from validity import block_bootstrap_trade_r

RES = str(PACKAGE_DIR / "results")
OUT = str(PACKAGE_DIR / "charts")
os.makedirs(OUT, exist_ok=True)

# palette (from palette.cascade)
ACCENT   = "#88702a"
ACCENT_2 = "#529cb5"
HEADER   = "#786b44"
ICON     = "#907e49"
BORDER   = "#d3ccb8"
MUTED    = "#8d8a83"
ERR      = "#8e443e"
OK       = "#3f7c54"
TXT      = "#22211f"

def style_ax(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(BORDER)
    ax.spines['bottom'].set_color(BORDER)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.25, color=MUTED)
    ax.set_axisbelow(True)

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=170, facecolor='white')
    plt.close(fig)
    print("saved", name)

# ── 1. Equity curves ────────────────────────────────────────────────────────
asis = pd.read_parquet(os.path.join(RES, "basket_trades_asis.parquet")).sort_values("entry_time")
a0 = pd.read_parquet(os.path.join(RES, "trades_A0_repo_exact.parquet")).sort_values("entry_time")
b4 = pd.read_parquet(os.path.join(RES, "trades_B4_plus_gap_fills.parquet")).sort_values("entry_time")
psar = pd.read_parquet(os.path.join(RES, "trades_I_psar_0.02_0.20.parquet")).sort_values("entry_time")

fig, ax = plt.subplots(figsize=(7.4, 3.6), constrained_layout=True)
for df, label, color, lw in [
    (asis, "Reported engine (no costs)", MUTED, 1.6),
    (a0, "Repo replication + costs", ACCENT_2, 1.6),
    (b4, "Corrected engine (B4) + costs", ACCENT, 2.4),
    (psar, "PSAR runner + costs", OK, 1.8),
]:
    ax.plot(np.arange(1, len(df) + 1), df["total_r"].cumsum() if "total_r" in df else df["real_r"].cumsum(),
            label=label, color=color, linewidth=lw, solid_capstyle='round')
style_ax(ax)
ax.set_xlabel("Trade number (chronological)", fontsize=9, color=TXT)
ax.set_ylabel("Cumulative R", fontsize=9, color=TXT)
ax.legend(loc='upper left', frameon=False, fontsize=8.5)
save(fig, "equity_curves.png")

# ── 2. Decomposition waterfall ─────────────────────────────────────────────
grid = json.load(open(os.path.join(RES, "exit_grid.json")))
steps = [
    ("Repo engine\n(as reported)", grid["A0_repo_exact"]["total_r"], 'base'),
    ("ECN costs", grid["A1_repo_plus_costs"]["total_r"] - grid["A0_repo_exact"]["total_r"], 'delta'),
    ("Sequencing\n(pessimistic)", grid["B1_sequencing_pess"]["total_r"] - grid["A1_repo_plus_costs"]["total_r"], 'delta'),
    ("Entry-bar\nadverse check", grid["B2_plus_entry_bar"]["total_r"] - grid["B1_sequencing_pess"]["total_r"], 'delta'),
    ("Honest trail\ntiming", grid["B3_plus_honest_trail"]["total_r"] - grid["B2_plus_entry_bar"]["total_r"], 'delta'),
    ("Gap-through\nstop fills", grid["B4_plus_gap_fills"]["total_r"] - grid["B3_plus_honest_trail"]["total_r"], 'delta'),
]
fig, ax = plt.subplots(figsize=(7.4, 3.8), constrained_layout=True)
cum = 0.0
xs = np.arange(len(steps))
for i, (label, val, kind) in enumerate(steps):
    if kind == 'base':
        ax.bar(i, val, color=HEADER, width=0.62)
        ax.text(i, val + 8, f"{val:+.0f}R", ha='center', fontsize=8.5, color=TXT)
        cum = val
    else:
        color = OK if val >= 0 else ERR
        ax.bar(i, val, bottom=cum, color=color, width=0.62, alpha=0.85)
        ax.text(i, cum + val + (8 if val >= 0 else -16), f"{val:+.0f}R", ha='center', fontsize=8.5, color=TXT)
        cum += val
        ax.plot([i - 0.31, i + 0.31], [cum, cum], color=MUTED, lw=0.7, ls=':')
ax.bar(len(steps), cum, color=ACCENT, width=0.62)
ax.text(len(steps), cum + 8, f"{cum:+.0f}R", ha='center', fontsize=8.5, color=TXT, fontweight='bold')
ax.set_xticks(list(xs) + [len(steps)])
ax.set_xticklabels([s[0] for s in steps] + ["Corrected\nengine (B4)"], fontsize=7.6, color=TXT)
style_ax(ax)
ax.set_ylabel("Cumulative R (974 trades)", fontsize=9, color=TXT)
ax.set_ylim(0, 360)
save(fig, "decomposition.png")

# ── 3. Exit-policy grid (horizontal bars) ───────────────────────────────────
pol_names = {
    "A0_repo_exact": "A0 repo engine (no costs)",
    "A1_repo_plus_costs": "A1 repo engine + costs",
    "B1_sequencing_pess": "B1 + pessimistic sequencing",
    "B2_plus_entry_bar": "B2 + entry-bar adverse check",
    "B3_plus_honest_trail": "B3 + honest trail timing",
    "B4_plus_gap_fills": "B4 + gap-true fills (= corrected)",
    "C_chandelier_k2.0": "Chandelier k=2.0",
    "D_chandelier_k2.5": "Chandelier k=2.5",
    "E_chandelier_k3.0": "Chandelier k=3.0",
    "F_chan_high_k3.0": "Chandelier-high k=3.0",
    "G_swing_lb3": "Swing structure lb=3",
    "I_psar_0.02_0.20": "Parabolic SAR 0.02/0.20",
    "K_giveback_0.30": "Giveback 30%",
    "L_giveback_0.40": "Giveback 40%",
    "M_hybrid_k2.5_g0.50": "Hybrid k=2.5, g=0.50",
    "N_hybrid_k3.0_g0.40": "Hybrid k=3.0, g=0.40",
}
items = [(pol_names[k], v["total_r"], v["pf"]) for k, v in grid.items() if k in pol_names]
items.sort(key=lambda t: t[1])
fig, ax = plt.subplots(figsize=(7.4, 4.6), constrained_layout=True)
names = [t[0] for t in items]
vals = [t[1] for t in items]
pfs = [t[2] for t in items]
colors_bar = [ACCENT_2 if "repo" in n or "B1" in n or "B2" in n else (OK if n.startswith("Parabolic") else HEADER) for n in names]
bars = ax.barh(names, vals, color=colors_bar, height=0.62, alpha=0.9)
for bar, v, p in zip(bars, vals, pfs):
    ax.text(bar.get_width() + 4, bar.get_y() + bar.get_height() / 2, f"{v:+.0f}R (PF {p:.2f})",
            va='center', fontsize=8, color=TXT)
style_ax(ax)
ax.set_xlabel("Total R over 974 trades (costs included unless noted)", fontsize=9, color=TXT)
ax.set_xlim(0, 445)
ax.tick_params(axis='y', labelsize=8.3)
save(fig, "exit_grid.png")

# ── 4. Correlation heatmaps: full vs stress ────────────────────────────────
BASKET = ["NAS100.p", "BTCUSD.pi", "UK100.p", "AU200.p", "EURUSD.pi",
          "SOLUSD.p", "USDCHF.pi", "USDJPY.pi", "XAGUSD.pi", "JP225.p"]
DATA = str(REPO_ROOT / "Blueberry_Data")
rets = {}
for sym in BASKET:
    df = pd.read_parquet(os.path.join(DATA, f"{sym}_1H.parquet"))
    df["time"] = pd.to_datetime(df["time"], utc=True)
    r = df.sort_values("time").set_index("time")["close"].resample("1D").last().pct_change().dropna()
    rets[sym] = r
rets = pd.DataFrame(rets).dropna()
corr_full = rets.corr()

from cluster_sizer import stress_correlation_scan
_, tail_dates, corr_stress = stress_correlation_scan(rets, halflife=20, top_pct=5)
corr_stress = pd.DataFrame(corr_stress, index=corr_full.index, columns=corr_full.columns)

short = {"NAS100.p": "NAS100", "BTCUSD.pi": "BTC", "UK100.p": "UK100", "AU200.p": "AU200",
         "EURUSD.pi": "EURUSD", "SOLUSD.p": "SOL", "USDCHF.pi": "USDCHF", "USDJPY.pi": "USDJPY",
         "XAGUSD.pi": "XAG", "JP225.p": "JP225"}

fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.7), constrained_layout=True)
for ax, mat, title in [(axes[0], corr_full, "Full-sample daily correlation"),
                       (axes[1], corr_stress, "Stress-tail EWMA correlation (top 5% days)")]:
    m = mat.loc[BASKET, BASKET]
    im = ax.imshow(m.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels([short[s] for s in BASKET], rotation=45, ha='right', fontsize=7.5)
    ax.set_yticklabels([short[s] for s in BASKET], fontsize=7.5)
    for i in range(10):
        for j in range(10):
            v = m.values[i, j]
            if i != j and abs(v) >= 0.55:
                ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=6.2,
                        color='white' if abs(v) > 0.75 else '#333333')
    ax.set_title(title, fontsize=9, color=TXT, pad=8)
    ax.tick_params(colors=MUTED)
cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.015)
cbar.ax.tick_params(labelsize=7.5, colors=MUTED)
save(fig, "corr_stress.png")

# ── 5. M5 ground truth ──────────────────────────────────────────────────────
m5 = json.load(open(os.path.join(RES, "m5_truth.json")))
assets = list(m5.keys())
labels = [short[a] for a in assets]
truth = [m5[a]["n_5m"]["total_r"] for a in assets]
repo = [m5[a]["n_1h_repo"]["total_r"] for a in assets]
honest = [m5[a]["n_1h_honest"]["total_r"] for a in assets]
x = np.arange(len(assets))
w = 0.26
fig, ax = plt.subplots(figsize=(7.2, 3.4), constrained_layout=True)
ax.bar(x - w, repo, w, label="1H repo engine", color=MUTED, alpha=0.85)
ax.bar(x, honest, w, label="1H corrected engine", color=ACCENT_2, alpha=0.9)
ax.bar(x + w, truth, w, label="5M ground truth", color=OK, alpha=0.95)
for xi, t in zip(x, truth):
    ax.text(xi + w, t + 0.8, f"{t:+.0f}", ha='center', fontsize=8, color=TXT)
style_ax(ax)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Total R (same-setup subset)", fontsize=9, color=TXT)
ax.axhline(0, color=BORDER, lw=0.8)
ax.legend(loc='upper left', frameon=False, fontsize=8.5, ncol=3)
save(fig, "m5_truth.png")

# ── 6. Meta-label lift ─────────────────────────────────────────────────────
meta = json.load(open(os.path.join(RES, "meta_empirical.json")))
lift = meta["model_win_label"]["lift"]
q = [l["quintile"] for l in lift]
mr = [l["mean_r"] for l in lift]
wr = [l["win_rate"] * 100 for l in lift]
fig, ax = plt.subplots(figsize=(6.8, 3.3), constrained_layout=True)
bars = ax.bar(q, mr, width=0.58, color=[ERR if v < 0.25 else ACCENT for v in mr], alpha=0.9)
for b, v, w_ in zip(bars, mr, wr):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}R\n{w_:.0f}% wr",
            ha='center', fontsize=8, color=TXT)
style_ax(ax)
ax.set_xticks(q); ax.set_xticklabels([f"Q{i}" for i in q], fontsize=9)
ax.set_xlabel("Predicted-probability quintile (purged CV, out-of-fold)", fontsize=9, color=TXT)
ax.set_ylabel("Mean realized R", fontsize=9, color=TXT)
ax.set_ylim(0, 0.62)
save(fig, "meta_lift.png")

# ── 7. Bootstrap drawdown distribution ──────────────────────────────────────
trades = b4.sort_values("entry_time").reset_index(drop=True)
rng_seed = 7
r = trades["total_r"].values
# re-run bootstrap to get raw draws (same seed as validity_empirical)
from validity import block_bootstrap_trade_r as bbt
# reproduce draws manually for the histogram
rng = np.random.default_rng(7)
n = len(r); block = 10; n_boot = 2000
n_blocks = int(np.ceil(n / block))
dds = []
for _ in range(n_boot):
    idx = (rng.integers(0, n, n_blocks)[:, None] + np.arange(block)[None, :]).ravel()[:n]
    seq = r[idx % n]
    eq = np.cumsum(seq)
    peak = np.maximum.accumulate(eq)
    dds.append((eq - peak).min())
dds = np.array(dds)
fig, ax = plt.subplots(figsize=(6.8, 3.2), constrained_layout=True)
ax.hist(dds, bins=48, color=ACCENT_2, alpha=0.85, edgecolor='white', linewidth=0.4)
ax.axvline(np.percentile(dds, 5), color=ERR, lw=1.6, ls='--')
ax.axvline(np.percentile(dds, 50), color=TXT, lw=1.6, ls='--')
ax.text(np.percentile(dds, 5) - 0.3, ax.get_ylim()[1] * 0.92, f"p5: {np.percentile(dds,5):.0f}R",
        ha='right', fontsize=8.5, color=ERR)
ax.text(np.percentile(dds, 50) + 0.3, ax.get_ylim()[1] * 0.92, f"median: {np.percentile(dds,50):.0f}R",
        ha='left', fontsize=8.5, color=TXT)
style_ax(ax)
ax.set_xlabel("Maximum drawdown in R (block bootstrap, 2,000 resamples, block=10)", fontsize=9, color=TXT)
ax.set_ylabel("Frequency", fontsize=9, color=TXT)
save(fig, "bootstrap_dd.png")

print("All charts done ->", OUT)
