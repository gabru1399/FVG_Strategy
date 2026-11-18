# Quant Review Package

Independent quantitative review of the FVG Momentum & Alpha Basket strategy
(1H entry / 4H HTF), shipped alongside the full report
[`FVG_Strategy_Quant_Review.pdf`](FVG_Strategy_Quant_Review.pdf).

The package contains three groups of files:

1. **Reference modules** — the corrected engine components proposed by the
   review: `exit_engine.py` (honest intrabar sequencing + exit-policy grid),
   `cluster_sizer.py` (EWMA correlation, HRP weights, heat-budget sizing),
   `meta_label.py` (18-feature library, purged+embargoed CV pipeline),
   `validity.py` (Deflated Sharpe, PBO/CSCV, block bootstrap, walk-forward).
2. **Empirical scripts** — reproduce every number and figure in the report
   from `Blueberry_Data/`. They are path-portable: run from the repository
   root, e.g. `python quant_review/audit_repro.py`. No configuration needed.
3. **Outputs** — `results/` (8 JSONs + intermediate parquets the pipeline
   stages produce), `charts/` (7 report figures), and the final PDF.

## Quick start

```bash
# from the repository root (Blueberry_Data/ must be present)
python quant_review/audit_repro.py
python quant_review/exit_grid.py
python quant_review/m5_truth.py
python quant_review/cluster_empirical.py
python quant_review/meta_empirical.py
python quant_review/validity_empirical.py
python quant_review/htf_lookahead.py
python quant_review/make_charts.py
python quant_review/gen_report.py && python quant_review/merge_pdf.py
```

Requirements: Python 3.9+, `numpy`, `pandas`, `pyarrow`, `scikit-learn`,
`matplotlib`, `reportlab`, `pypdf`. The reproduction needs ~2.5 GB free disk
for intermediate parquets and takes about 10 minutes end-to-end.

## What the review changed in the engine

Exactly one line of `quant_alpha_basket.py` was changed, behind the new
`honest_htf_merge=True` default in `extract_setups`:

```python
htf_clean['time'] = htf_clean['time'] + pd.Timedelta(hours=4)
merged = pd.merge_asof(df, htf_clean, on='time', direction='backward')
```

MT5 bars are open-time stamped, so the 4H bar stamped 08:00 only completes at
12:00. Merging backward on raw stamps let the still-forming bar's EMA50 leak
into 1H decisions for up to three hours (finding F9). The fix costs nothing:
the HTF filter decision flips on only 0.14–0.24% of bars across the basket
(see `results/htf_lookahead.json`). `audit_repro.py` pins
`honest_htf_merge=False` so the README's published baseline numbers stay
exactly reproducible next to the fixed engine.

## Key results

| Metric | README claim | Corrected engine (B4) | Best variant (SAR runner) |
|---|---|---|---|
| Trades | 960 | 974 | 974 |
| Total R | +178.0 | +301.0 (costs incl.) | +368.2 (costs incl.) |
| Profit factor | 1.57 | 1.82 | 2.02 |
| Max DD | -3.2% | -2.35% | -1.83% |
| 2R hit rate | 27.6% | 34.5% | 34.5% |

Corrected = honest intrabar sequencing (stop-before-target pessimism, entry-bar
adverse check, honest trail timing, gap-true fills) plus per-trade costs at the
README's own lot sizes. Full method and per-finding detail in the PDF.

## Layout

```
quant_review/
├── FVG_Strategy_Quant_Review.pdf   # the report (16 pp, 8 sections, 7 figures)
├── MODULES_README.md               # per-module docs + run order
├── README.md                       # this file
├── audit_repro.py                  # step 1: reproduce published baseline
├── exit_grid.py ... gen_report.py  # steps 2-9 of the pipeline (see MODULES_README.md)
├── cluster_sizer.py  exit_engine.py  meta_label.py  validity.py   # modules
├── cover.pdf  cover.html  merge_pdf.py                          # report cover
├── charts/                          # 7 report figures (PNG)
└── results/                         # 8 result JSONs shipped with the repo
```
