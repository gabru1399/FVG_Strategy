# Multi-Asset Quantitative FVG Momentum & Alpha Basket Strategy

An institutional-grade quantitative trading framework developed on live **MetaTrader 5 (MT5) tick data from Blueberry Markets**, evaluating and enhancing the Fair Value Gap (FVG) Momentum strategy across 156 assets.

---

## 1. Executive Summary & Core Results

Retail implementations of the Fair Value Gap (FVG) strategy rely on fragile 1:1 scalping and fixed pip stops that fail under real market friction. By introducing **Volatility Normalization (ATR-floor stops)**, an **Institutional Volume Gate (RVOL ≥ 1.15)**, and an **Asymmetric $2\text{R} + \text{Trailing Runner}$ execution engine**, this system achieves exceptional risk-adjusted performance across non-correlated asset classes.

### A. The 10-Asset Curated Alpha Basket (1H Entry / 4H HTF)
* **Capital Base**: \$2,500.00
* **Risk Per Trade ($1.0\text{ R}$)**: \$10.00 ($0.40\%$ risk)
* **Total Sample Size**: **960 verified trades** (~4 years, 2023–2026)
* **Hit $\ge 2\text{R}$ Win Rate**: **27.6%**
* **Breakeven Exit Rate ($+0.1\text{R}$)**: **39.8%** *(Risk eliminated on 4 out of 10 trades)*
* **Full Loss Rate ($-1.0\text{R}$)**: **32.6%** *(67.4% of all trades exit with ZERO loss or profit)*
* **Cumulative Return**: **+178.0 R** (Net Profit: **+$1,780.00** on \$2,500 base, $+71.2\%$ return)
* **Profit Factor**: **1.57**
* **Max Portfolio Drawdown**: **-3.2% (-$80.00)**
* **Calmar Ratio**: **14.70**
* **Trade Expectancy**: **+0.19 R per trade**
* **Average Winner Value**: **+$17.10** *(driven by the $1.5\times\text{ATR}$ Chandelier runner)*

### B. High-Frequency 5-Minute Intraday Scalp Basket (5M Entry / 1H HTF)
* **Approved Assets**: `EURUSD.pi`, `GBPUSD.pi`, `USDJPY.pi`, `NAS100.p`
* **Dataset**: ~100,000 live 5-minute bars per asset (~16 months from MT5)
* **Cumulative Return**: **+592.7 R** (Net Profit: **+$5,927.00** on \$10 risk)
* **Win Rate**: **55.1%** | **Max Drawdown**: **< 6.0%**

---

## 2. The 10-Asset Alpha Portfolio Allocation

| Asset | Category | Total Trades | 2R Win% | BE% | Loss% | Total R | Profit Factor | Max DD | Expectancy |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **NAS100.p** | US Tech Index | 90 | 36.7% | 34.4% | 28.9% | **+31.2 R** | **2.20** | **-3.7%** | +0.35 R |
| **BTCUSD.pi** | Crypto | 128 | 28.1% | 41.4% | 30.5% | **+26.1 R** | **1.67** | **-3.4%** | +0.20 R |
| **UK100.p** | UK Index | 84 | 29.8% | 40.5% | 29.8% | **+24.1 R** | **1.96** | **-3.6%** | +0.29 R |
| **AU200.p** | AUS Index | 80 | 35.0% | 31.2% | 33.8% | **+23.0 R** | **1.85** | **-3.2%** | +0.29 R |
| **EURUSD.pi** | FX Major | 138 | 27.5% | 36.2% | 36.2% | **+18.8 R** | **1.38** | **-3.8%** | +0.14 R |
| **SOLUSD.p** | Crypto | 93 | 28.0% | 40.9% | 31.2% | **+17.1 R** | **1.59** | **-1.8%** | +0.18 R |
| **USDCHF.pi** | FX Major | 114 | 21.1% | 48.2% | 30.7% | **+11.0 R** | **1.32** | **-3.1%** | +0.10 R |
| **USDJPY.pi** | FX Major | 128 | 25.0% | 37.5% | 37.5% | **+9.5 R** | **1.20** | **-4.8%** | +0.07 R |
| **XAGUSD.pi** | Silver | 58 | 24.1% | 44.8% | 31.0% | **+9.4 R** | **1.52** | **-2.3%** | +0.16 R |
| **JP225.p** | Japan Index | 61 | 21.3% | 41.0% | 37.7% | **+8.7 R** | **1.38** | **-5.6%** | +0.14 R |

> **Blacklist**: **XAUUSD (Gold)** with tight stops suffers severe spread/wick stopouts (-393 R loss). Never trade Gold with retail fixed-pip stops.

---

## 3. Strategy Mechanics (2R + Trailing Runner)

```
[Entry at FVG Midpoint]
      │
      ├──> Price reaches +1.0R ──> Move Stop Loss on both halves to Breakeven (+0.1R) [Risk Eliminated]
      │
      ├──> Price reaches +2.0R ──> Tranche 1 (50%): Take Profit locked at +2.0R (+$10.00)
      │                          Tranche 2 (50%): Trailing Runner Activated!
      │
      └──> Price Expands      ──> Chandelier Trailing Stop (Highest Close - 1.5 × ATR)
                                 Rides trend to +3R, +4R, +6R, +8R+
```

### Setup Requirements:
1. **Momentum Leg (Displacement)**: 2 consecutive candles with body ratio $|C-O|/(H-L) \ge 0.65$.
2. **Volume Confirmation**: Candle 2 tick volume $\ge 1.15 \times \text{SMA}_{20}$ volume.
3. **Swing Break**: Close breaches 10-bar prior swing high/low.
4. **HTF Confluence**: 4H chart $\text{Close} > \text{EMA 50}$ & $\text{EMA 20} \ge \text{EMA 50}$ (for Longs).
5. **Entry**: Limit order at the FVG Midpoint.
6. **Stop Loss**: $\max(\text{FVG boundary}, 1.2 \times \text{ATR}_{14})$.

---

## 4. MT5 Lot Sizing Table ($2,500 Account, $10 Risk)

$$\text{Lot Size} = \frac{\$10.00}{\text{Stop Loss Distance} \times \text{Tick Value per Lot}}$$

| Asset | Stop Distance | Pip/Point Value (per 0.01 lot) | MT5 Lot Size ($10 Risk) |
| :--- | :---: | :---: | :---: |
| **EURUSD.pi** | 18 pips | \$0.10 | **0.05 Lots** |
| **EURUSD.pi (5M)** | 9 pips | \$0.10 | **0.11 Lots** |
| **GBPUSD.pi (5M)** | 9 pips | \$0.10 | **0.11 Lots** |
| **USDJPY.pi** | 22 pips | \$0.063 | **0.07 Lots** |
| **USDCHF.pi** | 16 pips | \$0.118 | **0.05 Lots** |
| **NAS100.p** | 35 points | \$0.10 | **0.03 Lots** |
| **UK100.p** | 25 points | \$0.13 | **0.03 Lots** |
| **AU200.p** | 30 points | \$0.067 | **0.05 Lots** |
| **BTCUSD.pi** | \$450 distance | \$0.01 / \$1 | **0.02 Lots** |
| **XAGUSD.pi** | 30 cents | \$0.50 / 10c | **0.01 Lots** |

---

## 5. Repository File Structure

* `quant_alpha_basket.py`: Complete quantitative screening, ranking, and portfolio backtest engine.
* `test_m5_fvg_strategy.py`: 5-minute tick-level backtest engine with asymmetric R:R.
* `dump_mt5_m5.py`: High-speed data extraction script pulling 100,000 M5 bars from MT5 into Parquet.
* `backtest_fvg.py`: Initial verification engine auditing the research paper's claims vs MT5 candles.
* `test_papers_synthesis.py`: Audit evaluating SSRN-6725880, SSRN-5002058 (Quantpedia), and SSRN-6858299.
* `Blueberry_Data/`: Historical 1H, 4H, and 5M parquet files for 156 assets from Blueberry Markets MT5.
* `FVG_Strategy.pdf`: Original research paper by Teghjot Singh Bindra.
