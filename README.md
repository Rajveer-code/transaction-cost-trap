# When the Gate Stays Closed

**Empirical Evidence of Near-Zero Cross-Sectional Predictability in Large-Cap NASDAQ Equities Using an IC-Gated Machine Learning Framework**

*Rajveer Singh Pall — Independent Researcher*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Under Review](https://img.shields.io/badge/status-under%20review-orange.svg)]()

---

## Overview

This repository accompanies the paper **"When the Gate Stays Closed: Empirical Evidence of Near-Zero Cross-Sectional Predictability in Large-Cap NASDAQ Equities Using an IC-Gated Machine Learning Framework"**.

We build a walk-forward cross-sectional conviction-ranking framework that combines isotonic-calibrated ensemble learning with an **Information Coefficient (IC) gate** — a statistical mechanism that prevents position-taking whenever no reliable cross-sectional signal is detected. Applied to a survivorship-bias-controlled universe of **30 large-cap NASDAQ-100 stocks** over a **6-year out-of-sample window** (October 2018 – October 2024, 1,512 trading days) using 49 causal technical indicators, the framework finds no exploitable predictive signal.

The IC gate stays closed throughout the evaluation window. We treat this as the correct outcome, not a failure — this study documents signal absence in a well-specified, methodologically rigorous setting.

Five independent robustness analyses confirm the finding: expanding the universe to N=100 stocks, testing gradient saliency stability across folds, applying the Diebold-Mariano test against a random baseline, conditioning IC on the VIX regime, and constructing circular block-bootstrap CIs for each fold's IC.

---

## Key Results

| Strategy | Ann. Return | Sharpe Ratio | Max Drawdown |
|---|---|---|---|
| **Equal-Weight Benchmark** | +25.0% | **0.96** | −32.4% |
| SPY Buy-and-Hold | +14.9% | 0.74 | −33.7% |
| TopK1 (ML Conviction) | −5.9% | −0.16 | −67.0% |
| Random Top-1 | −4.6% | −0.12 | −65.6% |

**IC Signal Test** — Mean IC = −0.0005 &nbsp;|&nbsp; ICIR = −0.0023 &nbsp;|&nbsp; HAC t = −0.090 &nbsp;|&nbsp; p = 0.464

**Permutation Test** — Observed TopK1 Sharpe = −0.16 vs null 95th pct = +0.44 &nbsp;|&nbsp; p = 0.742

**Diebold-Mariano Test** — TopK1 vs Random Top-1: DM = 0.42, p = 0.672 (no systematic advantage)

**Block Bootstrap** — All 12 fold-level IC 95% CIs span zero

**Calibration** — ECE < 0.025 across all 12 folds (well-calibrated despite zero discrimination)

> **Core finding:** The ensemble achieves well-calibrated probability estimates (ECE < 0.025) while providing no cross-sectional discrimination (IC ≈ 0). Calibration quality is orthogonal to predictive content. Five robustness checks — spanning universe size, feature stability, statistical testing, market regime, and bootstrap uncertainty — all confirm the absence of exploitable signal.

---

## The IC Gate Pipeline

```
[ Daily OHLCV — 30 NASDAQ-100 Stocks, Jan 2015 – Dec 2024 ]
        │
        ▼
[ Feature Engineering ] ──── 49 Strictly Causal Technical Indicators
        │
        ▼
[ Walk-Forward Engine ] ──── 12-Fold Expanding Window (10-day embargo)
        │
        ▼
[ Ensemble Model      ] ──── CatBoost + Random Forest + MLP (equal weight)
        │
        ▼
[ Isotonic Calibration] ──── Per-fold, fitted on 6-month held-out window
        │
        ▼
[ IC Gate             ] ──── HAC t-test (Newey-West, lag=9)
        │                    + Permutation test (1,000 shuffles)
        │                    CLOSED if p ≥ 0.05 or mean IC ≤ 0
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

> **PyTorch (optional):** The MLP component requires PyTorch. The ensemble degrades gracefully to CatBoost + Random Forest if PyTorch is unavailable. Install PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/).

---

## Quickstart

**Reproduce paper results (30-stock NASDAQ-100 dataset):**

```bash
python scripts/run_experiments.py --data-path data/nasdaq30_prices.parquet --use-cache
```

**Rapid test with a small download (7 stocks):**

```bash
python scripts/run_experiments.py --use-cache
```

**Expected runtime:** 30–90 minutes on modern hardware. Fold-level checkpoints write to `results/predictions/` — interrupted runs resume from the last completed fold.

**Fama-French 6-factor regression:**

```bash
python scripts/factor_regression.py
```

**Permutation test (1,000 shuffles, parallelised):**

```bash
python scripts/parallel_permutation.py
```

**Regenerate all 12 publication figures:**

```bash
python generate_figures.py
```

**Rebuild the manuscript docx:**

```bash
python build_manuscript_v2.py
```

---

## Robustness Analyses

Five independent checks confirm the gate closure finding:

### R1 — Expanded Universe (N=100)

```bash
python scripts/robustness/robustness_01_expanded_universe.py
```

Runs the full IC pipeline on a 100-stock NASDAQ universe. Mean IC and ICIR remain near zero; the gate closes at the same rate as N=30.

Output: `results/robustness/expanded_universe/`

### R2 — Gradient Saliency Stability

```bash
python scripts/robustness/robustness_02_shap_analysis.py
```

Computes gradient-based token saliency (‖∂P/∂E‖₂) per fold and measures inter-fold Spearman rank correlation of top-20 features. ρ = 0.13–0.40 confirms the model fits noise, not stable structure.

Output: `results/robustness/shap/`

> **Note:** The script is named `shap_analysis.py` for legacy reasons. The implementation uses gradient magnitude, not SHAP Shapley values.

### R3 — Diebold-Mariano Test

```bash
python scripts/robustness/robustness_03_04_05_dm_vix_bootstrap.py
```

HLN-corrected DM test (HAC Newey-West, 5 lags, squared loss) comparing TopK1 vs Random Top-1. DM = 0.42, p = 0.672 — the ML ranking offers no systematic improvement over random selection.

Output: `results/robustness/dm_test/`

### R4 — VIX-Conditioned IC

Same script as R3. Partitions fold-level IC by VIX tercile (low / medium / high volatility regime). The gate closes in all three regimes — regime conditioning does not reveal hidden pockets of predictability.

Output: `results/robustness/vix_ic/`

### R5 — Block Bootstrap CIs

Same script as R3. Circular block bootstrap (block size = 5 days, B = 2,000 resamples) constructs 95% CIs for each fold's IC. All 12 CIs span zero.

Output: `results/robustness/bootstrap/`

---

## Repository Structure

```
transaction-cost-trap/
├── README.md
├── requirements.txt
├── generate_figures.py           # Regenerate all 12 publication figures from CSVs
├── build_manuscript_v2.py        # Rebuild final manuscript docx
│
├── data/
│   └── nasdaq30_prices.parquet   # 30-stock NASDAQ-100 OHLCV, Jan 2015–Dec 2024
│                                 # Survivorship-bias controlled; SPY included
│
├── src/
│   ├── data/
│   │   └── data_loader.py        # OHLCV ingestion + 49-feature causal matrix
│   ├── training/
│   │   ├── models.py             # CatBoost, RF, MLP, Ensemble classes
│   │   ├── walk_forward.py       # Expanding-window generator with 10-day embargoes
│   │   └── calibration.py        # Isotonic regression, ECE, IC computation
│   └── backtesting/
│       └── backtester.py         # Vectorized strategy engine + transaction cost model
│
├── scripts/
│   ├── run_experiments.py        # End-to-end pipeline with checkpointing
│   ├── factor_regression.py      # Fama-French 6-factor regression
│   ├── parallel_permutation.py   # Parallelised permutation null distribution
│   └── robustness/
│       ├── robustness_01_expanded_universe.py    # R1: N=100 IC test
│       ├── robustness_02_shap_analysis.py        # R2: Gradient saliency fold stability
│       └── robustness_03_04_05_dm_vix_bootstrap.py  # R3–R5: DM, VIX-IC, bootstrap
│
├── paper/
│   ├── when_the_gate_stays_closed.docx           # Earlier draft
│   └── when_the_gate_stays_closed_FINAL.docx     # Final submission-ready manuscript
│
└── results/
    ├── metrics/                  # Core CSVs: IC test, strategy comparison, etc.
    │   ├── ic_test_results.csv
    │   ├── strategy_comparison.csv
    │   ├── k_sensitivity.csv
    │   ├── subperiod_analysis.csv
    │   ├── cost_sensitivity_topk1.csv
    │   └── factor_regression_topk1_specs.csv
    ├── permutation/
    │   ├── permutation_topk1.csv
    │   └── permutation_topk1_summary.csv
    ├── figures/
    │   └── pub/                  # 12 publication figures (300 DPI PNG + PDF)
    │       ├── fig01_strategy_comparison.{png,pdf}
    │       ├── fig02_permutation.{png,pdf}
    │       ├── fig03_ic_bootstrap.{png,pdf}
    │       ├── fig04_k_sensitivity.{png,pdf}
    │       ├── fig05_cost_sensitivity.{png,pdf}
    │       ├── fig06_subperiod_heatmap.{png,pdf}
    │       ├── fig07_ff_alpha.{png,pdf}
    │       ├── fig08_shap_importance.{png,pdf}
    │       ├── fig09_vix_ic.{png,pdf}
    │       ├── fig10_expanded_universe.{png,pdf}
    │       ├── fig11_dm_test.{png,pdf}
    │       └── fig12_gate_summary.{png,pdf}
    └── robustness/
        ├── expanded_universe/    # R1: ic_comparison_30vs100.csv, ic_results_100.csv
        ├── shap/                 # R2: shap_mean_abs_by_fold.csv, fold rank correlation
        ├── dm_test/              # R3: dm_test_results.csv
        ├── vix_ic/               # R4: vix_conditioned_ic.csv
        └── bootstrap/            # R5: fold_ic_with_bootstrap_ci.csv
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
| | random_seed / thread_count | 42 / 1 |
| **Random Forest** | n_estimators | 500 |
| | max_depth | 10 |
| | min_samples_leaf | 20 |
| | random_state | 42 |
| **MLP** | architecture | [256, 128, 64] |
| | dropout | 0.3 |
| | learning_rate | 1e-3 |
| | early_stopping patience | 10 |
| **Calibration** | method | Isotonic regression (per fold) |
| **IC Gate (parametric)** | test | HAC t-test, Newey-West lag=9 |
| | threshold | p < 0.05 AND mean IC > 0 |
| **IC Gate (non-parametric)** | test | Permutation (1,000 shuffles) |
| **Bootstrap** | method | Circular block bootstrap |
| | block size | 5 days |
| | resamples | 2,000 |

All random seeds are fixed to 42. `n_jobs=1` throughout for full reproducibility.

---

## Feature Set (49 Causal Technical Indicators)

All features are strictly backward-looking — no `center=True`, `min_periods` set to full window length.

| Category | Count | Features |
|---|---|---|
| Momentum | 8 | RSI-14/21, MACD line/signal/histogram, ROC-10/21, Williams %R |
| Bollinger Bands | 6 | Upper, lower, mid, bandwidth, %B position, CCI-20 |
| Volatility | 6 | ATR-14/21, rolling log-return vol 5/21/63d, HL range |
| Trend / MA | 8 | EMA-9/21/50/200, SMA-50/200, price-to-SMA200, price-to-SMA50 |
| Returns | 6 | 1d through 21d log returns |
| Volume | 5 | OBV, OBV-EMA, volume z-score 5d/21d, MFI-14 |
| Candle Structure | 4 | OC body, upper/lower shadow, DPO-20 |
| Directional | 6 | Stochastic %K/%D, ADX-14, DI+/DI−, VWAP deviation |

**Target variable:** `y_t = 1{Close(t+2) > Close(t+1)}` — execution at Close(t+1), return over the next holding day. The 2-day shift eliminates execution-at-signal lookahead bias.

---

## Walk-Forward Design

| Fold | Training window | Calibration window | Test window |
|---|---|---|---|
| 1 | Jan 2015 – Dec 2016 | Jan 2017 – Jun 2017 | Jul 2017 – Dec 2017 |
| 2 | Jan 2015 – Jun 2017 | Jul 2017 – Dec 2017 | Jan 2018 – Jun 2018 |
| … | expanding | fixed 6 months (held-out) | fixed 6 months |
| 12 | Jan 2015 – Mar 2024 | Apr 2024 – Jun 2024 | Jul 2024 – Dec 2024 |

A 10-day embargo separates each training set from its test set to prevent close-price leakage. The calibration window is carved from the tail of the training period and never overlaps with the test window.

---

## Interpreting the Results

| Output | What it tells you |
|---|---|
| `results/metrics/ic_test_results.csv` | `significant == False` across all folds — no exploitable cross-sectional signal |
| `results/metrics/strategy_comparison.csv` | TopK1 Sharpe < Equal-Weight Sharpe — the ranking is uninformative |
| `results/permutation/permutation_topk1_summary.csv` | `p_value > 0.05` — observed performance indistinguishable from random |
| `results/robustness/dm_test/dm_test_results.csv` | `p > 0.05` — ML ranking offers no systematic advantage over random selection |
| `results/robustness/bootstrap/fold_ic_with_bootstrap_ci.csv` | All 12 fold CIs span zero — IC is statistically indistinguishable from zero in every fold |
| `results/robustness/vix_ic/vix_conditioned_ic.csv` | Gate closed in all three VIX regimes — no regime reveals hidden predictability |
| `results/metrics/k_sensitivity.csv` | Sharpe increases monotonically with K — benchmark convergence signature of an uninformative ranker |

---

## Figures

All 12 figures use Times New Roman, 300 DPI, Ocean Dusk palette (`#264653`, `#2A9D8F`, `#E9C46A`, `#F4A261`, `#E76F51`). PDFs are vector-quality for journal submission.

| Figure | Description |
|---|---|
| fig01 | Strategy comparison — annualised return, Sharpe, max drawdown |
| fig02 | Permutation null distribution (1,000 shuffles) with observed Sharpe |
| fig03 | Fold-level IC with 95% block-bootstrap confidence intervals |
| fig04 | K-sensitivity: Sharpe vs TopK with benchmark convergence annotation |
| fig05 | Transaction cost sensitivity — Sharpe and return vs bps (0–50) |
| fig06 | Subperiod heatmap — 4 strategies × 3 periods (RdYlGn) |
| fig07 | Fama-French 6-factor alpha and t-statistics |
| fig08 | Gradient saliency — top-20 features by mean fold importance |
| fig09 | VIX-conditioned IC by regime (low/medium/high) with gate status |
| fig10 | Expanded universe: mean IC and ICIR for N=30 vs N=100 |
| fig11 | Diebold-Mariano test panel — all model pairs |
| fig12 | Three-panel gate summary: IC statistics, fold-level IC, gate decisions |

---

## Limitations

- The universe covers only mega-cap NASDAQ-100 stocks — results may not generalise to small-cap or international markets.
- Features are restricted to OHLCV-derived technical indicators. Fundamental data, NLP signals, order-book features, and cross-asset momentum were not tested.
- The sample period (2015–2024) is dominated by a prolonged NASDAQ bull market; a bear-market-dominated sample may produce different IC estimates.
- Daily rebalancing with a 2-day execution lag and 5 bps round-trip cost is one of several plausible market microstructure assumptions.
- The SamLowe-RoBERTa model used in feature experiments was pre-trained on GoEmotions Reddit data, creating partial overlap with the Reddit test set.

---

## Citation

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

- Bailey, D. H., Borwein, J., Lopez de Prado, M., & Zhu, Q. J. (2014). Pseudo-mathematics and financial charlatanism. *Notices of the AMS*, 61(5), 458–471.
- Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.
- Chordia, T., Subrahmanyam, A., & Tong, Q. (2014). Have capital market anomalies attenuated in the recent era of high liquidity and trading activity? *Journal of Financial Economics*, 114(2), 501–523.
- de Prado, M. L. (2018). *Advances in Financial Machine Learning*. Wiley.
- Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Fama, E. F. (1991). Efficient capital markets: II. *Journal of Finance*, 46(5), 1575–1617.
- Freyberger, J., Neuhierl, A., & Weber, M. (2020). Dissecting characteristics nonparametrically. *Review of Financial Studies*, 33(5), 2326–2377.
- Grinold, R., & Kahn, R. (1999). *Active Portfolio Management* (2nd ed.). McGraw-Hill.
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*.
- Harvey, C. R., Liu, Y., & Zhu, H. (2016). …and the cross-section of expected returns. *Review of Financial Studies*, 29(1), 5–68.
- Harvey, D. I., Leybourne, S. J., & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.
- Lo, A. W. (2000). Finance: A selective survey. *Journal of the American Statistical Association*, 95(450), 629–635.
- Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. *Econometrica*, 55(3), 703–708.
- Niculescu-Mizil, A., & Caruana, R. (2005). Predicting good probabilities with supervised learning. *ICML*, 625–632.
- Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased boosting with categorical features. *NeurIPS*, 6638–6648.

---

## License

MIT License — Copyright (c) 2025 Rajveer Singh Pall
