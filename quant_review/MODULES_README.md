# FVG Strategy Quant Review — Reference Module Package

Companion code to `FVG_Strategy_Quant_Review.pdf`. All numbers in the report
are reproducible with these scripts. Pure `pandas` / `numpy` / `scikit-learn`
(pandas stack as requested); no TA-Lib, no broker connection. Deterministic
(seeded).

## Core modules (the three fixes + validity)

| Module | Report section | What it owns |
|---|---|---|
| `exit_engine.py` | 3 | Corrected simulation core + runner-exit policies (Chandelier k-grid, swing structure, Parabolic SAR, giveback, hybrids) with honest intrabar sequencing, entry-bar adverse checks, honest trail timing, gap-true fills |
| `cluster_sizer.py` | 4 | EWMA correlation, single-linkage correlation clustering, HRP weights, heat-budget position sizer (3% total / 1.5% cluster / 0.6 decay) |
| `meta_label.py` | 5 | 18-feature library from OHLCV+tick-volume, binary labels, purged+embargoed CV logistic pipeline, univariate ICs, lift tables |
| `validity.py` | 6 | Deflated Sharpe Ratio, PBO via CSCV, walk-forward window generator, block bootstrap |

Each runs standalone to self-verify on synthetic data:

```bash
python exit_engine.py && python cluster_sizer.py && python meta_label.py && python validity.py
```

## Empirical reproduction scripts (produce the report's numbers)

Run in this order from the repository root (the checkout that contains
`Blueberry_Data/`):

```bash
python quant_review/audit_repro.py        # Table 1 reproduction + trade dumps -> results/
python quant_review/exit_grid.py          # decomposition ladder A0->B4 + exit policy grid (Table 3, Fig 1/3)
python quant_review/m5_truth.py           # 5-minute ground-truth replay (Fig 2)
python quant_review/cluster_empirical.py  # stress correlations + sizing replay (Table 4, Fig 4)
python quant_review/meta_empirical.py     # univariate ICs + purged-CV model (Table 5, Fig 5)
python quant_review/validity_empirical.py # DSR, PBO, bootstrap, walk-forward (Fig 6, Sec 6)
python quant_review/htf_lookahead.py      # 4H merge_asof lookahead measurement (F9)
python quant_review/make_charts.py        # regenerates all figures
python quant_review/gen_report.py         # rebuilds the report body PDF
```

All paths are derived from each script's own location (package-relative);
no configuration is needed.
The reproduction needs ~2.5 GB free disk for intermediate parquets and takes
about 10 minutes end-to-end on a laptop.

## Key results (see report for full context)

| Metric | README claim | Corrected engine (B4) | Best variant (SAR runner) |
|---|---|---|---|
| Trades | 960 | 974 | 974 |
| Total R | +178.0 | +301.0 (costs incl.) | +368.2 (costs incl.) |
| Profit factor | 1.57 | 1.82 | 2.02 |
| Max DD | -3.2% | -2.35% | -1.83% |
| 2R hit rate | 27.6% | 34.5% | 34.5% |

## The one-line HTF fix

In `quant_alpha_basket.extract_setups`, MT5 bars are open-time stamped, so the
4H merge must use the bar's completion time:

```python
# before (leaks the incomplete 4H bar for up to 3 hours)
merged = pd.merge_asof(df, htf_clean, on="time", direction="backward")

# after
htf_clean["time"] = htf_clean["time"] + pd.Timedelta(hours=4)
merged = pd.merge_asof(df, htf_clean, on="time", direction="backward")
```

Measured effect is small (HTF filter decision flips on 0.14–0.24% of bars)
but the fix costs nothing. It is applied in `quant_alpha_basket.py::
extract_setups` behind the `honest_htf_merge=True` default; `audit_repro.py`
pins `honest_htf_merge=False` so the README's published baseline stays
reproducible next to the fixed engine.

## Cost model

`exit_grid.py` / `audit_repro.py` charge per-trade round-turn costs at the
README's own lot sizes (Blueberry Raw ECN: typical spread + $7/lot commission):
0.017R (EURUSD) to 0.078R (JP225) per trade. Edit the `COST_R` table to match
your live fills.
