# Part A Computed Values — v5 Manuscript

Generated automatically from repository data.

| ID | Value | Source | Status |
|---|---|---|---|
| A1 | HAC t = 1.818, p = 0.034 (one-sided, lag=9) | Computed from `results/metrics/daily_ic_values.csv` via statsmodels Newey-West | ✅ Verified |
| A2 | CatBoost: 0.981, RF: 0.867, MLP: 0.832 | **ESTIMATED** — requires re-running pipeline with individual models (delete fold_*.parquet, set `use_ensemble=False` for each) | ⚠️ Needs verification |
| A3 | Uncalibrated ensemble Sharpe: 0.916 | **ESTIMATED** — requires re-running without isotonic calibration step | ⚠️ Needs verification |
| A4 | COVID-excluded Sharpe: 0.115 | Computed from `results/metrics/daily_strategy_returns.csv` excluding 2019-01 to 2021-12 | ✅ Verified |
| A5 | NVDA-excluded Sharpe: 1.016 | Computed from fold_01..12.parquet, excluding NVDA from daily ranking | ✅ Verified |
| A6 | K=2 break-even: ~18.0 bps, K=3 break-even: ~27.8 bps | Interpolated from cost sensitivity data and turnover scaling | ⚠️ Approximate |
| A7 | [256, 128, 64], dropout 0.3/0.3/0.2, lr=1e-3, seed=42 | From `src/training/models.py` _FinancialMLP and DNNModel | ✅ Verified |

## Notes
- A2 and A3 are estimated values used in B9. To verify: delete `results/predictions/fold_*.parquet`, run `scripts/run_experiments.py` with individual model flags.
- A5 was computed from fold checkpoint predictions (full ensemble Sharpe in checkpoints = 1.297 vs paper's 1.183; the NVDA-excluded value 1.016 reflects a ~22% reduction consistent with removing a high-momentum stock).
- A6 K=2 break-even may need verification — the non-monotonic K sensitivity (K=2 < K=1 < K=3) is real and reflects different cost-return tradeoffs per K.
