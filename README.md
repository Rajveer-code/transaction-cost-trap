# When the Gate Stays Closed

**Empirical Evidence of Near-Zero Cross-Sectional Predictability in Large-Cap NASDAQ Equities Using an IC-Gated Machine Learning Framework**

*Rajveer Singh Pall — Independent Researcher*

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status: Under Review](https://img.shields.io/badge/status-under%20review-orange.svg)]()

---

## Overview

This repository contains the complete, reproducible code and results for the working paper:

> **Pall, R. S. (2025).** *When the Gate Stays Closed: Empirical Evidence of Near-Zero Cross-Sectional Predictability in Large-Cap NASDAQ Equities Using an IC-Gated Machine Learning Framework.* Working paper.

The paper introduces the **IC-Gated Deployment Framework (ICGDF)** — a two-stage, statistically grounded pre-deployment filter for financial machine learning systems. The gate requires Newey-West HAC-corrected Information Coefficient (IC) significance *and* permutation confirmation before any capital is deployed, directly addressing the false-discovery and backtest-overfitting problems identified by Harvey, Liu and Zhu (2016) and Bailey et al. (2014).

Applied to 30 survivorship-bias-controlled NASDAQ-100 stocks over **1,512 consecutive out-of-sample trading days** (October 2018 – October 2024) using 49 strictly causal OHLCV indicators:

- **The gate stays closed in every fold.** Mean IC = −0.0005, HAC t = −0.09, p = 0.464. The ensemble is well-calibrated (ECE < 0.025) yet produces no exploitable cross-sectional discrimination — demonstrating that calibration quality and predictive content are orthogonal.
- **A momentum positive control** achieves Sharpe = 0.57 over the same window, confirming cross-sectional structure exists. Its daily IC is directionally positive but statistically insufficient under HAC-corrected inference — revealing that momentum's returns come from multi-week trend persistence, a mechanism distinct from daily IC significance.
- **An ablation study** demonstrates both gate components are necessary: naive t-test produces 11.8% false positive rate under a simulated AR(1) null; the full two-stage ICGDF reduces this to 0.0%.
- **Five independent robustness checks** — expanded universe (N=100), SHAP feature stability, Diebold-Mariano test, VIX-conditioned IC, and block bootstrap CIs — all confirm the null.

---

## Repository Structure

```
transaction-cost-trap/
│
├── paper/
│   └── when_the_gate_stays_closed_FINAL.docx   # Manuscript (submission version)
│
├── src/                                          # Core pipeline modules
│   ├── data/
│   │   └── data_loader.py                        # Yahoo Finance data acquisition
│   ├── training/
│   │   ├── models.py                             # CatBoost, Random Forest, MLP ensemble
│   │   ├── calibration.py                        # Isotonic probability calibration
│   │   └── walk_forward.py                       # 12-fold expanding-window validator
│   └── backtesting/
│       └── backtester.py                         # Vectorised portfolio backtest engine
│
├── scripts/
│   ├── run_experiments.py                        # Main pipeline entry point
│   ├── factor_regression.py                      # Fama-French factor regressions
│   ├── parallel_permutation.py                   # Permutation test (parallelised)
│   └── robustness/
│       ├── robustness_01_expanded_universe.py    # R1: N=100 universe check
│       ├── robustness_02_shap_analysis.py        # R2: SHAP feature attribution & stability
│       ├── robustness_03_04_05_dm_vix_bootstrap.py  # R3–R5: DM test, VIX IC, bootstrap CIs
│       ├── robustness_06_momentum_ic_gate.py     # Momentum positive control (IC gate)
│       └── robustness_07_ablation.py             # Gate component ablation study
│
├── results/
│   ├── figures/pub/                              # Publication-ready figures (PNG + PDF)
│   ├── metrics/                                  # IC statistics, strategy metrics (CSV)
│   ├── permutation/                              # Permutation null distributions (CSV)
│   ├── plots/reliability_diagrams/               # Per-fold calibration diagrams
│   └── robustness/
│       ├── expanded_universe/                    # R1 outputs
│       ├── shap/                                 # R2 outputs
│       ├── dm_test/                              # R3 outputs
│       ├── vix_ic/                               # R4 outputs
│       ├── bootstrap/                            # R5 outputs
│       ├── momentum_ic/                          # Momentum IC gate results
│       └── ablation/                             # Ablation study results
│
├── data/
│   └── nasdaq30_prices.parquet                   # Adjusted OHLCV, 30 stocks, 2015–2024
│
├── build_manuscript_v2.py                        # Manuscript builder (python-docx)
├── generate_figures.py                           # Publication figure generation
├── requirements.txt                              # Python dependencies
└── .gitignore
```

---

## Methodology

### ICGDF Algorithm

ICGDF applies a two-stage gate before every deployment decision. Both conditions must hold simultaneously; if either fails, no position is taken.

**Input:** Daily OHLCV panel for N stocks over T trading days; α = 0.05; HAC lag L = 9; permutation replicates B = 1,000.

**Stage 1 — Training and Calibration (per fold k)**
1. Construct expanding training window with 2-calendar-day embargo.
2. Engineer 49 strictly causal OHLCV features; no future-referencing windows.
3. Fit CatBoost, Random Forest, and MLP independently; combine by equal probability averaging.
4. Calibrate via isotonic regression on the last 20% of the training window (frozen before test).

**Stage 2 — IC Gate (applied before each deployment decision)**

5. Compute daily IC: IC_d = SpearmanRankCorr(p̂_d, r_{d+1}) for each day in the test fold.
6. **Gate Condition A** — Newey-West HAC t-test (lag = 9 days):

   `t_HAC = IC̄ / √(V̂_HAC / N) > 1.645  AND  IC̄ > 0`

7. **Gate Condition B** — Permutation test (B = 1,000 shuffles):

   `p_perm < 0.05`

8. Gate opens ⟺ Condition A AND Condition B.
9. Gate closed → no position.
10. Gate open → equal weight to the K highest-conviction stocks; 5 bps round-trip cost.

The gate is **model-agnostic**: any base learner producing a cross-sectional conviction ranking can be substituted at Step 3 without modifying the gate logic.

### Walk-Forward Design

| Parameter | Value |
|---|---|
| Out-of-sample period | October 2018 – October 2024 |
| OOS trading days | 1,512 |
| Folds | 12 expanding windows (6-month increments) |
| Embargo | 2 calendar days |
| Calibration window | Last 20% of each training period |
| HAC lag | 9 days (≈ 2 trading weeks) |
| Permutation replicates | B = 1,000 |
| Stocks | 30 survivorship-bias-controlled NASDAQ-100 members |
| Features | 49 strictly causal OHLCV technical indicators |

### Ensemble Configuration

| Component | Configuration |
|---|---|
| CatBoost | 500 trees · depth 6 · lr 0.05 · l2_leaf_reg 3.0 |
| Random Forest | 500 trees · max_depth 10 · min_samples_leaf 20 |
| MLP | [256→128→64] · dropout 0.3 · early stopping (patience 10) |
| Combination | Equal probability averaging |
| Calibration | Isotonic regression (per fold, frozen before test) |
| Random seed | 42 (all components) |

---

## Key Results

### IC Gate — Full Out-of-Sample Window

| Statistic | Value | Threshold | Decision |
|---|---|---|---|
| Mean IC | −0.0005 | > 0 | Negative |
| IC Std Dev | 0.2204 | — | — |
| ICIR | −0.0023 | > 0.5 (practice) | Near zero |
| HAC t-statistic | −0.090 | > 1.645 | Not significant |
| p-value (HAC, one-sided) | 0.464 | < 0.05 | **Gate CLOSED** |
| Permutation p-value | 0.742 | < 0.05 | **Gate CLOSED** |
| Gate-open folds | 0 / 12 | ≥ 1 | Never opened |

### Strategy Performance (October 2018 – October 2024)

| Strategy | Ann. Return | Sharpe | Sortino | Max DD | # Trades |
|---|---|---|---|---|---|
| Equal-Weight Benchmark | +25.0% | **0.96** | 1.28 | −32.4% | 1 |
| SPY Buy & Hold | +14.9% | 0.74 | 0.91 | −33.7% | 0 |
| Momentum Top-1 | +26.4% | 0.57 | 0.79 | −62.7% | 407 |
| TopK3 (ML) | +3.5% | 0.12 | 0.16 | −38.2% | 1,248 |
| TopK2 (ML) | −0.3% | −0.01 | −0.01 | −53.9% | 1,134 |
| Random Top-1 | −4.6% | −0.12 | −0.15 | −65.6% | 1,461 |
| **TopK1 (ML, gate closed)** | **−5.9%** | **−0.16** | **−0.21** | **−67.0%** | 833 |

The **benchmark convergence signature** — Sharpe increasing monotonically from TopK1 (−0.16) to TopK2 (−0.01) to TopK3 (+0.12) toward the equal-weight limit (0.96) — is the mathematical diagnostic of a cross-sectional ranker with zero information content.

### Ablation Study: Gate Component Necessity

Simulated AR(1) null IC process (φ = 0.30, N = 126 days per trial, 500 trials):

| Gate Variant | Null False Positive Rate | Assessment |
|---|---|---|
| Naive t-test only | **11.8%** | Severely inflated (> 2× nominal α) |
| HAC t-test only | 7.6% | Reduced but above α |
| Full ICGDF (paper) | **0.0%** | Correct under simulated conditions |

### Momentum Positive Control — Mechanism Analysis

| Signal | Mean IC | IC Std | ICIR | HAC t | p-value | Gate |
|---|---|---|---|---|---|---|
| Momentum (252-day trailing) | +0.0071 | 0.4747 | +0.015 | +0.60 | 0.276 | CLOSED |
| ML Ensemble (baseline) | −0.0005 | 0.2204 | −0.0023 | −0.09 | 0.464 | CLOSED |

Momentum's Sharpe advantage (0.57) comes from multi-week trend persistence, not daily IC significance — a mechanistically distinct predictive channel that the ICGDF daily IC criterion is not designed to screen.

### Robustness Checks

| Check | Key Result | Conclusion |
|---|---|---|
| R1: Expanded Universe (N=100) | p = 0.947; IC = −0.006 | Gate CLOSED — result not universe-specific |
| R2: SHAP Feature Attribution | Inter-fold rank ρ = 0.13–0.40 | Noise-fitting confirmed; no stable signal |
| R3: Diebold-Mariano Test | DM = 0.42, p = 0.672 (vs Random Top-1) | ML indistinguishable from random selection |
| R4: VIX-Conditioned IC | Min p-value = 0.136 (all regimes) | Gate CLOSED in all volatility regimes |
| R5: Block Bootstrap CIs | 0 / 12 folds exclude zero | All fold CIs consistent with null IC |

### Calibration vs. Discrimination (Orthogonality Finding)

| Metric | Value | Interpretation |
|---|---|---|
| Mean ECE (across 12 folds) | < 0.025 | Excellent calibration |
| Mean IC | −0.0005 | Zero discriminative content |

**The model is well-specified but the signal class contains no exploitable cross-sectional information in this setting.** Calibration quality alone is an insufficient criterion for deployment readiness.

---

## Reproducing Results

### 1. Clone and install

```bash
git clone https://github.com/Rajveer-code/transaction-cost-trap.git
cd transaction-cost-trap
pip install -r requirements.txt
```

### 2. Run the main pipeline

```bash
# Walk-forward training, IC gate evaluation, backtest
python scripts/run_experiments.py

# Factor regressions (CAPM, FF3, FF5, FF5+MOM)
python scripts/factor_regression.py

# Permutation null distribution (parallelised)
python scripts/parallel_permutation.py
```

### 3. Run robustness checks (in order)

```bash
python scripts/robustness/robustness_01_expanded_universe.py   # ~15 min (N=100 universe)
python scripts/robustness/robustness_02_shap_analysis.py       # ~5 min  (SHAP values)
python scripts/robustness/robustness_03_04_05_dm_vix_bootstrap.py  # ~10 min (DM, VIX, bootstrap)
python scripts/robustness/robustness_06_momentum_ic_gate.py    # ~3 min  (momentum IC gate)
python scripts/robustness/robustness_07_ablation.py            # ~2 min  (gate ablation)
```

All scripts run from the repository root and write outputs to `results/robustness/`.

### 4. Generate publication figures

```bash
python generate_figures.py
# Outputs: results/figures/pub/fig01_*.png through fig12_*.png
```

### 5. Build the manuscript

```bash
python build_manuscript_v2.py
# Output: paper/when_the_gate_stays_closed_FINAL.docx
```

> **Note on data:** `data/nasdaq30_prices.parquet` (3.2 MB) is included in the repository and contains adjusted OHLCV data for all 30 stocks from January 2015 through December 2024, obtained from Yahoo Finance. All scripts use this cached file by default; delete it to force a fresh download.

---

## Requirements

Tested on Python 3.10/3.11/3.12, Windows 11 and Ubuntu 22.04.

```
catboost>=1.2
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
scipy>=1.11
yfinance>=0.2.28
shap>=0.44
python-docx>=1.1
matplotlib>=3.7
seaborn>=0.13
statsmodels>=0.14
pyarrow>=14.0
```

Install with: `pip install -r requirements.txt`

---

## Citation

```bibtex
@article{pall2025icgdf,
  title   = {When the Gate Stays Closed: Empirical Evidence of Near-Zero
             Cross-Sectional Predictability in Large-Cap {NASDAQ} Equities
             Using an {IC}-Gated Machine Learning Framework},
  author  = {Pall, Rajveer Singh},
  year    = {2025},
  note    = {Working paper. Available at:
             \url{https://github.com/Rajveer-code/transaction-cost-trap}}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for full terms.

---

## Contact

**Rajveer Singh Pall**  
Independent Researcher  
rajveerpall04@gmail.com  
[github.com/Rajveer-code](https://github.com/Rajveer-code)
