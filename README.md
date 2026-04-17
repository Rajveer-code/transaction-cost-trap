# When the Gate Stays Closed

**Empirical Evidence of Near-Zero Cross-Sectional Predictability in Large-Cap NASDAQ Equities Using an IC-Gated Machine Learning Framework**

*Rajveer Singh Pall — Independent Researcher*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Under Review](https://img.shields.io/badge/status-under%20review-orange.svg)]()

---

## Overview

This repository accompanies the paper **"When the Gate Stays Closed: Empirical Evidence of Near-Zero Cross-Sectional Predictability in Large-Cap NASDAQ Equities Using an IC-Gated Machine Learning Framework"**.

We implement a walk-forward cross-sectional conviction ranking framework that integrates isotonic-calibrated ensemble learning with an **Information Coefficient (IC) gate** — a statistical mechanism that prevents position-taking when no reliable cross-sectional signal is detected. Applied to a survivorship-bias-controlled universe of **30 large-cap NASDAQ-100 stocks** over a **6-year out-of-sample window** (October 2018 – October 2024, 1,512 trading days), the framework finds no exploitable predictive signal using 49 technical indicators.

The IC gate correctly stays closed throughout the evaluation window. We argue this constitutes a correct decision, not a failure — making this a principled empirical study of signal absence in efficient markets.

---

## Key Results

| Strategy | Ann. Return | Sharpe Ratio | Max Drawdown |
|---|---|---|---|
| **Equal-Weight Benchmark** | +25.0% | **0.96** | −32.4% |
| SPY Buy-and-Hold | +14.9% | 0.74 | −33.7% |
| TopK1 (ML Conviction) | −5.9% | −0.16 | −67.0% |
| Random Top-1 | −4.6% | −0.12 | −65.6% |

**IC Signal Test** — Mean IC = −0.0005 &nbsp;|&nbsp; ICIR = −0.0023 &nbsp;|&nbsp; t = −0.090 &nbsp;|&nbsp; p = 0.464 (not significant)

**Permutation Test** — Observed TopK1 Sharpe = −0.16 vs null 95th pct = +0.44 &nbsp;|&nbsp; p = 0.742

**Calibration** — ECE < 0.025 across all 12 folds (excellent, despite zero discrimination)

> **Core finding:** The ensemble achieves well-calibrated probability estimates (ECE < 0.025) while providing zero cross-sectional discrimination (IC ≈ 0). This orthogonality — calibration quality independent of predictive content — is a key methodological finding.

---

## The IC Gate Mechanism

```
[ Daily OHLCV — 30 NASDAQ-100 Stocks, Jan 2015 – Dec 2024 ]
        │
        ▼
[ Feature Engineering ] ──── 49 Strictly Causal Technical Indicators
        │
        ▼
[ Walk-Forward Engine ] ──── 12-Fold Expanding Window (no lookahead)
        │
        ▼
[ Ensemble Model      ] ──── CatBoost + Random Forest + MLP (equal weight)
        │
        ▼
[ Isotonic Calibration] ──── Per-fold calibration on held-out window
        │
        ▼
[ IC Gate             ] ──── HAC t-test (Newey-West, lag=9) + Permutation test
        │                    CLOSED if p ≥ 0.05  →  no positions taken
        ▼
[ Vectorized Backtest ] ──── TopK Conviction Ranking + Transaction Cost Model
```

---

## Installation

```bash
git clone https://github.com/Rajveer-code/transaction-cost-trap.git
cd transaction-cost-trap
pip install -r requirements.txt
```

> **PyTorch (optional):** The MLP component requires PyTorch. The ensemble degrades gracefully to CatBoost + Random Forest if PyTorch is unavailable. Install PyTorch separately from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Quickstart

**Recommended — reproduce the paper results exactly (30-stock NASDAQ-100 dataset included):**

```bash
python scripts/run_experiments.py --data-path data/nasdaq30_prices.parquet --use-cache
```

**For rapid testing with a small download (7 stocks, not paper-matched):**

```bash
python scripts/run_experiments.py --use-cache
```

**Expected runtime:** 30–90 minutes on modern hardware. Fold-level checkpoints are written to `results/predictions/` — interrupted runs resume automatically from the last completed fold.

**Fama-French factor regression (supplementary analysis):**

```bash
python scripts/factor_regression.py
```

**Permutation test (parallelised, 1,000 shuffles):**

```bash
python scripts/parallel_permutation.py
```

---

## Repository Structure

```
transaction-cost-trap/
├── README.md
├── requirements.txt
│
├── data/
│   └── nasdaq30_prices.parquet       # 30-stock NASDAQ-100 OHLCV, Jan 2015 – Dec 2024
│                                     # Survivorship-bias controlled; SPY included as benchmark
│
├── src/
│   ├── data/
│   │   └── data_loader.py            # OHLCV ingestion + 49-feature causal matrix
│   ├── training/
│   │   ├── models.py                 # CatBoost, RF, MLP, and Ensemble classes
│   │   ├── walk_forward.py           # Expanding-window generator with temporal embargoes
│   │   └── calibration.py            # Isotonic regression, ECE, IC computation
│   └── backtesting/
│       └── backtester.py             # Vectorized strategy engine + transaction cost model
│
├── scripts/
│   ├── run_experiments.py            # End-to-end pipeline with checkpointing
│   ├── factor_regression.py          # Fama-French 6-factor regression
│   └── parallel_permutation.py       # Parallelised permutation null distribution
│
├── paper/
│   └── when_the_gate_stays_closed.docx   # Manuscript (under review)
│
└── results/
    ├── metrics/                      # CSV outputs: IC test, strategy comparison, etc.
    │   ├── ic_test_results.csv
    │   ├── strategy_comparison.csv
    │   ├── k_sensitivity.csv
    │   ├── subperiod_analysis.csv
    │   ├── cost_sensitivity_topk1.csv
    │   └── factor_regression_topk1_specs.csv
    ├── permutation/                  # Permutation null distribution
    │   ├── permutation_topk1.csv
    │   └── permutation_topk1_summary.csv
    ├── figures/                      # Publication figures (PNG + PDF)
    │   ├── fig01_framework.{png,pdf}
    │   ├── fig02_cumulative_returns.{png,pdf}
    │   ├── fig03_permutation.{png,pdf}
    │   ├── fig04_subperiod_heatmap.{png,pdf}
    │   ├── fig05_strategy_comparison.{png,pdf}
    │   ├── fig06_k_sensitivity.{png,pdf}
    │   ├── fig07_cost_sensitivity.{png,pdf}
    │   ├── fig08_ic_analysis.{png,pdf}
    │   ├── fig09_reliability_grid.{png,pdf}
    │   ├── fig10_ff_alpha_table.{png,pdf}
    │   └── fig11_model_ablation.{png,pdf}
    └── plots/
        └── reliability_diagrams/     # Per-fold ECE reliability diagrams (all 3 models)
```

---

## Model Configuration

| Component | Parameter | Value |
|---|---|---|
| **CatBoost** | iterations | 500 |
| | learning_rate | 0.05 |
| | depth | 6 |
| | l2_leaf_reg | 3.0 |
| | early_stopping_rounds | 50 |
| | random_seed / thread_count | 42 / 1 (reproducibility) |
| **Random Forest** | n_estimators | 500 |
| | max_depth | 10 |
| | min_samples_leaf | 20 |
| | random_state | 42 |
| **MLP** | architecture | [256, 128, 64] |
| | dropout | 0.3 |
| | learning_rate | 1e-3 |
| | early_stopping | patience=10 |
| **Calibration** | method | Isotonic regression (per fold) |
| **IC Gate** | parametric test | HAC t-test (Newey-West lag=9) |
| | threshold | p < 0.05 (one-tailed) |
| | non-parametric | Permutation test (1,000 shuffles) |

All random seeds are fixed to 42. `n_jobs=1` is set throughout for full determinism.

---

## Feature Set (49 Causal Technical Indicators)

All features are strictly backward-looking. No `center=True`, `min_periods` equal to the full window length.

| Category | Count | Features |
|---|---|---|
| Momentum | 8 | RSI-14/21, MACD line/signal/histogram, ROC-10/21, Williams %R |
| Bollinger Bands | 5 | Upper, lower, mid, bandwidth, %B position, CCI-20 |
| Volatility | 6 | ATR-14/21, rolling log-return vol 5/21/63d, HL range |
| Trend / MA | 8 | EMA-9/21/50/200, SMA-50/200, price-to-SMA200, price-to-SMA50 |
| Returns | 6 | 1d through 21d log returns |
| Volume | 5 | OBV, OBV-EMA, volume z-score 5d/21d, MFI-14 |
| Candle Structure | 4 | OC body, upper/lower shadow, DPO-20 |
| Directional | 6 | Stochastic %K/%D, ADX-14, DI+/DI−, VWAP deviation |

**Target variable:** `y_t = 1{Close(t+2) > Close(t+1)}` — execution assumed at Close(t+1), return measured over the subsequent holding period. This 2-day shift eliminates execution-at-signal lookahead bias.

---

## Interpreting the Results

| Output file | What it tells you |
|---|---|
| `results/metrics/ic_test_results.csv` | `significant == False` across all folds confirms no exploitable cross-sectional signal |
| `results/metrics/strategy_comparison.csv` | TopK1 Sharpe < Equal-Weight Sharpe confirms the ranking is uninformative |
| `results/permutation/permutation_topk1_summary.csv` | `p_value > 0.05` confirms observed performance is indistinguishable from random |
| `results/figures/fig09_reliability_grid.png` | ECE < 0.025 per fold confirms calibration works; IC ≈ 0 confirms no discrimination |
| `results/metrics/k_sensitivity.csv` | Monotonic Sharpe improvement with K = "benchmark convergence signature" of an uninformative ranker |

---

## Walk-Forward Design

| Split | Training window | Calibration window | Test window |
|---|---|---|---|
| Fold 1 | Jan 2015 – Dec 2016 | Jan 2017 – Jun 2017 | Jul 2017 – Dec 2017 |
| … | expanding | fixed 6-month held-out | fixed 6-month |
| Fold 12 | Jan 2015 – Mar 2024 | Apr 2024 – Jun 2024 | Jul 2024 – Dec 2024 |

The calibration window is a separate held-out subset of the training period. The calibrator never observes the test distribution before inference.

---

## Reproducing the Figures

All 11 publication figures were generated with matplotlib (300 DPI, Times New Roman, Ocean Dusk palette). To regenerate:

```bash
python scripts/run_experiments.py --data-path data/nasdaq30_prices.parquet --plot
```

Individual figures can also be regenerated from the saved CSV outputs in `results/metrics/` and `results/permutation/`.

---

## Limitations

- The 30-stock NASDAQ-100 universe covers only mega-cap technology stocks — results may not generalise to broader or more diverse universes.
- Features are restricted to OHLCV-derived technical indicators. Fundamental data, NLP signals, and cross-asset momentum were not tested.
- Daily rebalancing with a 2-day execution lag may not match all investor profiles.
- The sample period (2015–2024) is dominated by a prolonged NASDAQ bull market.

---

## Citation

If you use this code or build on this framework, please cite:

```bibtex
@article{pall2025gate,
  title   = {When the Gate Stays Closed: Empirical Evidence of Near-Zero
             Cross-Sectional Predictability in Large-Cap NASDAQ Equities
             Using an IC-Gated Machine Learning Framework},
  author  = {Pall, Rajveer Singh},
  journal = {Under review},
  year    = {2025},
  url     = {https://github.com/Rajveer-code/transaction-cost-trap}
}
```

---

## References

- Bailey et al. (2014). Pseudo-mathematics and financial charlatanism. *Notices of the AMS*, 61(5).
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Chordia et al. (2014). Have capital market anomalies attenuated? *Journal of Financial Economics*, 114(2).
- Fama, E. F. (1991). Efficient capital markets: II. *Journal of Finance*, 46(5).
- Grinold & Kahn (1999). *Active Portfolio Management*. McGraw-Hill.
- Gu, Kelly & Xiu (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5).
- Harvey, Liu & Zhu (2016). …and the cross-section of expected returns. *Review of Financial Studies*, 29(1).
- Niculescu-Mizil & Caruana (2005). Predicting good probabilities with supervised learning. *ICML*.
- Prokhorenkova et al. (2018). CatBoost. *NeurIPS*, 6638–6648.

---

## License

MIT License — Copyright (c) 2025 Rajveer Singh Pall
