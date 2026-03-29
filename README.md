# Overcoming the Transaction Cost Trap: Cross-Sectional Conviction Ranking in Machine Learning Equity Prediction

A publication-grade implementation of a cross-sectional equity ranking methodology designed to bypass the transaction cost trap routinely observed in financial machine learning classification architectures.

Dependencies: Python 3.10+ | License: MIT | Status: Research Execution

## Abstract

Binary classification models in financial machine learning consistently exhibit high directional accuracy while failing economically due to magnitude asymmetry and compounding transaction costs. This study identifies and quantifies the "transaction cost trap," wherein algorithms accurately predict minor price fluctuations but exhaust accumulated capital through excessive portfolio turnover. To bridge this gap, this pipeline introduces Cross-Sectional Conviction Ranking, treating raw probability outputs as relative conviction scores to dynamically allocate capital only to the highest-conviction assets globally. By isolating the Top-K standard models, the methodology establishes a statistically significant Information Coefficient while compressing turnover by approximately 85%, systematically outperforming equivalent binary threshold invariants when realistic institutional friction constraints are enforced.

## Research Motivation

The transaction cost trap occurs when predictive algorithms generate statistically significant classification accuracy that evaporates immediately upon exposure to real-world friction. Standard supervised learning architectures mapping directional probability boundaries (e.g., $P > 0.50$) inherently force algorithms to aggressively shift positions tracking microscopic predictive edges. For example, a model generating 55% accuracy that dictates 400 parameter rotations annually will invariably underperform a 51% accuracy model executing exactly 50 targeted structural trades, purely due to accumulated slippage. By recalibrating models away from independent binary thresholds and toward a cross-sectional competition framework, capital deployment mathematically limits holding rotations to periods of extreme relative distributional shift.

## Methodology Overview

```text
[ Daily OHLCV Data ]
         |
         v
[ Feature Engineering ] ---> 47 Causal Technical Indicators
         |
         v
[ Walk-Forward Engine ] ---> Expanding Window Validation + 2-Day Embargo
         |
         v
[ Model Optimization  ] ---> CatBoost / RF / DNN / Ensemble Architectures
         |
         v
[ Probability Calib.  ] ---> Isotonic Regression (Held-out Subsets)
         |
         v
[ Signal Validity     ] ---> Spearman IC Statistical Gate
         |
         v
[ Vectorized Backtest ] ---> Top-K Conviction Ranking Engine
```

* Data structuring constructs uniform MultiIndex tracking formats binding exogenous indicator persistence without yielding lookahead parameters.
* Walk-Forward engines enforce strict dynamic expanding window matrices validating parameters out-of-sample over ten years of historical density.
* Model Optimization yields diverse variance aggregation via gradient boosting and deep representations unified by uniform prediction interfaces.
* Probability calibration corrects the native extreme-confidence defects found in decision trees, adjusting raw outputs to empirical realities.
* Signal Validity halts execution entirely if the base rank distributions fail to achieve a statistically positive representation boundary against noise.
* Vectorized Backtesting transforms theoretical vectors into fractional capital weights, exacting structural transaction penalties across temporal changes.

## Repository Structure

```text
research_clean/
|-- src/
|   |-- data/
|   |   `-- data_loader.py         # OHLCV ingestion and 47-feature causal matrix builder
|   |-- training/
|   |   |-- walk_forward.py        # Expanding window generator with structural embargoes
|   |   |-- models.py              # CatBoost, RF, DNN, and Ensemble inference logic
|   |   `-- calibration.py         # Isotonic regression, ECE, and IC mapping evaluations
|   `-- backtesting/
|       `-- backtester.py          # Vectorized metrics engine enforcing transaction friction
|-- scripts/
|   `-- run_experiments.py         # End-to-end framework test orchestrator and checkpointing
|-- requirements.txt               # Deterministic structural environment lock
|-- .gitignore                     # Data omission and artifact masking
`-- README.md                      # Foundational documentation and methodological map
```

## Installation

```bash
git clone https://github.com/Rajveer-code/transaction-cost-trap.git
cd transaction-cost-trap
pip install -r requirements.txt
```
*Note: The DNN architecture defaults to CPU execution. PyTorch with CUDA acceleration is supported natively if hardware dependencies are configured.*

## Quickstart

Execute the entire data ingestion, calibration, and backtesting pipeline deterministically:

```bash
python scripts/run_experiments.py
```

Expected runtime on modern hardware is between 30 and 90 minutes. 
The script automatically builds stateful `.parquet` checkpoints directly in `results/predictions/` after each successfully tested walk-forward fold. If the execution is manually interrupted, subsequent executions will instantly reload the validation cache and resume processing at the next structural fold boundary.

## Experimental Design

| Experiment | Research Question | Null Hypothesis | Key Metric | Baseline |
|---|---|---|---|---|
| 1. IC Signal Check | Is cross-sectional separation valid prior to execution? | Mean IC <= 0 | Rank Spearman IC | N/A (Statistical Gate) |
| 2. Strategy Compare| Does cross-sectional ranking bypass friction drag? | Top-K Sharpe <= Baseline | HAC Sharpe Ratio | Threshold (P>0.50) |
| 3. Permutation Test| Is the rank ordering structurally superior to noise? | Obs Sharpe <= Null Sharpe | Empirical p-value | Random_Top1 (1000 iter) |
| 4. Regime Analysis | Does the cross-sectional edge persist across cycles? | Identical structural decay | Max Drawdown | BuyHold_SPY |
| 5. Cost Sensitivity| At what friction bound does the anomaly disintegrate? | Break-even <= 5 bps | Break-even bps | Equal_Weight |
| 6. K Concentration | Does portfolio concentration amplify net annualized yield? | Variance is fully stochastic | Annualized Return | Threshold (P>0.60) |

## Key Design Decisions (Lookahead Bias Prevention)

1. **Target Label Shift ($t+2$)**: 
The methodology targets yielding $Close_{t+2} / Close_{t+1} - 1$, guaranteeing that execution prices mimic actionable future market entries precisely matching realistic execution lags generated after signal creation at $t$.

2. **StandardScaler Fit Isolation**: 
Normalization transformers are rigorously formulated to measure and fit parameters exclusively over the 80% distribution phase of internal training intervals. Applying full-window normalization leaks terminal boundary variance backwards into structural predictions.

3. **In-Loop Hyperparameter Evolution**: 
Static optimal parameters are never established preemptively over the global dataset. Adjustments occur within dynamic validation walls, assuring tuning reactions behave identically to uninformed empirical observation states over time.

4. **Two-Day Exclusion Embargo**: 
Structural overlaps in temporal return formations inject artificial correlations between the right tail of train subsets and test beginnings. A strict 2-day deletion embargo systematically isolates training edges away from target realization paths.

5. **SMA(200) Strict Index Processing**: 
Moving averages natively project NaN values until the boundary is met. Implementation enforces `min_periods=200` to categorically deny trailing sparse-data calculations the capacity to influence trend-filtering mechanics.

6. **Held-Out Model Calibration**: 
Isotonic probability regression behaves aggressively over established data. Calibration engines are assigned an explicit unique 20% block of the designated training sequences, completely cordoned off from base model gradients and future OOS validations.

7. **Shifted DPO Configuration**: 
Calculations for Detrended Price Oscillators invoke backward-looking formulations uniquely isolated from classical centered moving combinations, explicitly suppressing the native lookahead bleed present in standard DPO formulations.

## Statistical Validity

Confidence and dispersion analysis are fundamentally constrained by non-normal asset return distribution properties. The Lo (2002) HAC-adjusted Sharpe Ratio enforces rigorous standard errors rectifying serial autocorrelation and conditional heteroskedasticity natively contaminating equity sequences.

The methodology utilizes non-parametric permutation evaluations as primary hypothesis boundaries. Shuffling output probability distributions cross-sectionally exactly 1000 times destroys predictive capability while maintaining chronological label frequency, effectively forcing the methodology to prove its excess returns survive independently of market beta drift.

Prior to constructing capital tracking simulations, Information Coefficient evaluation stands as an unyielding pass/fail framework gate. If cross-sectional probability ranks fail to exhibit significantly positive monotonic correlation against yield labels ($p < 0.05$), theoretical portfolio friction analysis constitutes statistical data-dredging and is structurally void.

Implementation handles immense strategic combinatorics by applying Benjamini-Hochberg False Discovery Rate procedures across output parameter branches, limiting spurious configuration anomalies from improperly establishing publication metrics.

## Strategy Variants

| Strategy | Type | Description | Null Baseline |
|---|---|---|---|
| Baseline_P50 | Threshold | Capital distributed linearly if P > 0.50 | - |
| Threshold_P60 | Threshold | Capital distributed linearly if P > 0.60 | Baseline_P50 |
| TopK1 | Ranking | 100% allocation specifically to maximum conviction probability | Random_Top1 |
| TopK2 | Ranking | Splitting capital equivalently between top two probabilities | Random_Top2 |
| TopK3 | Ranking | Tertiating capital cleanly amongst top three convictions | Equal_Weight |
| TopK1_Trend | Filtered | Top-ranking extraction post-SMA200 regime checks | TopK1 |
| Random_Top1 | Baseline | Stochastic random allocation identically mimicking TopK rules | Equal_Weight |
| Momentum_Top1| Baseline | Historic pure-momentum ranking allocations (trailing 21D map) | Random_Top1 |
| Equal_Weight | Index | Purely static division universally held constantly | - |
| BuyHold_SPY | Index | Standard S&P 500 benchmark tracking comparison | - |

## Results Interpretation Guide

* **`strategy_comparison.csv`**: Establish that `TopK1` yields positive compounding distributions safely surviving 5bps friction parameters while thoroughly invalidating `Baseline_P50` returns internally destroyed by identical transaction rates.
* **`ic_test_results.csv`**: Verify `significant == True`. A failure here designates the anomaly has structurally deteriorated in the historical data sample analyzed.
* **`permutation_topk1_summary.csv`**: Observe the `p_value`. Values below $0.05$ confirm the derived allocation sequences behave materially superior to 1000 separate localized stochastic approximations.
* **`reliability_diagrams/`**: Ensure the post-calibration blue mappings fall decisively closer to the diagonal `y=x` ideal variance boundary than the uncalibrated extreme-confidence red curve geometries.

## Known Limitations

The empirical verification relies entirely upon a constrained 7-stock institutional-grade large-cap technology universe. Substantial survivorship bias impacts inference generation; properties may wildly diverge against distressed small-cap boundaries holding minimal structural continuity.

Sector-specific concentration risk exposes the implementation to acute exogenous structural shocks. Tech-centric variance uniquely governs the distribution density, leaving utility against diversified macroeconomic sequences fundamentally untested.

Transaction executions assume static structural limits averaging 5 basis points. Frictional volatility matching order-book depths, bid/ask spread collapse dynamics, and gross systemic block execution slippage are natively idealized within standard constant limitations.

## Citation

```bibtex
@article{pall2025transaction,
  title={Overcoming the Transaction Cost Trap: Cross-Sectional Conviction Ranking in Machine Learning Equity Prediction},
  author={Pall, Rajveer Singh},
  journal={TBD},
  year={2025}
}
```

* Lo, A. W. (2002). *The statistics of Sharpe ratios*. Financial Analysts Journal, 58(4), 36-52.
* Niculescu-Mizil, A., & Caruana, R. (2005). *Predicting good probabilities with supervised learning*. Proceedings of the 22nd international conference on Machine learning.
* Jegadeesh, N., & Titman, S. (1993). *Returns to buying winners and selling losers: Implications for stock market efficiency*. The Journal of finance, 48(1), 65-91.

## License

MIT License

Copyright (c) 2025 Rajveer Singh Pall

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
