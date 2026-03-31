"""
run_experiments.py
==================
End-to-end orchestration scripting bridging the Financial ML Pipeline.

Executes sequential walk-forward training validation (Module 2), deep bounds model
aggregation (Module 3/4), strictly isolated OOS inference, transaction friction
evaluation profiles (Module 5), and generates publication-ready analysis figures.

Expected Runtime: 30-90 minutes (with integrated checkpoints to resume progress).

Author: Rajveer Singh Pall
Paper : "Overcoming the Transaction Cost Trap: Cross-Sectional Conviction
         Ranking in Machine Learning Equity Prediction"
"""

from __future__ import annotations

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import scipy.stats
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

# Import local modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.data.data_loader import load_all_data, get_feature_columns
from src.training.walk_forward import (
    WalkForwardFold, generate_folds, get_fold_arrays, get_cal_arrays, print_fold_summary
)
from src.training.models import CatBoostModel, EnsembleModel
from src.training.calibration import (
    fit_calibrator, calibrated_predict, compute_spearman_ic, test_ic_significance, plot_reliability_diagram
)
from src.backtesting.backtester import (
    STRATEGY_CONFIGS, run_backtest, run_all_strategies, run_permutation_test,
    run_subperiod_analysis, run_cost_sensitivity, run_spy_buyhold
)

OUTPUT_DIR = Path("results")

# ---------------------------------------------------------------------------
# ORCHESTRATION PIPELINE
# ---------------------------------------------------------------------------

def build_predictions_df(
    fold: WalkForwardFold,
    df: pd.DataFrame,
    actual_returns_series: pd.Series,
    feature_cols: List[str],
    fitted_model: Any,
    calibrator: IsotonicRegression,
    scaler: StandardScaler,
    model_name: str,
) -> pd.DataFrame:
    """
    Format prediction tracking subsets strictly confined within valid OOS boundary arrays.
    """
    # 1. Isolate test slice accurately tracking MultiIndex dates
    dates_level = df.index.get_level_values('date')
    test_mask = dates_level.isin(set(fold.test_dates))
    test_df = df.loc[test_mask].copy()

    # 2. Scale
    assert hasattr(scaler, 'mean_') and scaler.mean_ is not None, "Scaler inherently unfitted"
    X_test = scaler.transform(test_df[feature_cols].values)

    # 3. Predict & Calibrate
    probs = calibrated_predict(fitted_model, calibrator, X_test)
    assert probs.shape == (len(test_df),), "Shape misalignment on cross-sectional probabilities"

    # 4. Integrate required column references and actual yields (already subset cleanly)
    result = pd.DataFrame({
        'prob': probs,
        'Close': test_df['Close'].values,
        'SMA_200': test_df['SMA_200'].values,
        'target': test_df['target'].values,
        'trailing_return_21d': test_df['return_21d'].values,
    }, index=test_df.index)

    # Slice actual returns to only identical test boundaries natively aligned on MultiIndex
    aligned_actuals = actual_returns_series.loc[test_df.index]
    result['actual_return'] = aligned_actuals.values
    result['model'] = model_name

    # 5. Drop NaN rows (final 2 series instances per asset lacking t+2 tracking capabilities)
    result = result.dropna(subset=['actual_return'])

    # 6. Sanity invariants
    assert not result[['prob', 'Close', 'SMA_200', 'target']].isna().any().any(), "Structural NaN detection inside metrics"

    return result


def run_walk_forward_loop(
    df: pd.DataFrame,
    feature_cols: List[str],
    folds: List[WalkForwardFold],
    use_ensemble: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[float]]:
    """
    Execute exhaustive walk-forward calibration cycle logging incrementally.
    """
    all_predictions = []
    all_fold_ic = []
    
    # Pre-compute exact cross-sectional realization bounds iteratively to escape boundary limitations
    actual_returns_dict = {}
    tickers = df.index.get_level_values('ticker').unique()
    for ticker in tickers:
        close = df.xs(ticker, level='ticker')['Close']
        actual_returns_dict[ticker] = close.shift(-2) / close.shift(-1) - 1.0
        
    actual_returns_df = pd.concat({t: s for t, s in actual_returns_dict.items()}, axis=1).stack()
    actual_returns_df.index.names = ['date', 'ticker']
    # .stack() orders natively (date, ticker), identical to standard index layout map

    print("\n" + "=" * 50)
    print("Initiating Master Pipeline Sequential Runs")
    print("=" * 50)
    
    plots_dir = OUTPUT_DIR / "plots" / "reliability_diagrams"
    preds_dir = OUTPUT_DIR / "predictions"

    for fold in folds:
        print("\n" + "=" * 50)
        print(f"FOLD {fold.fold_number}/{len(folds)}")
        print(f"  Train: {fold.train_start.date()} to {fold.train_end.date()}")
        print(f"  Test : {fold.test_start.date()} to {fold.test_end.date()}")
        print(f"  Train rows: {len(fold.model_train_dates) * len(tickers)}")
        print(f"  Test rows : {len(fold.test_dates) * len(tickers)}")
        
        # Checkpoint Guard
        checkpoint_path = preds_dir / f"fold_{fold.fold_number:02d}.parquet"
        if checkpoint_path.exists():
            print(f"  [CHECKPOINT] Fold {fold.fold_number} loaded from cache. Skipping training.")
            fold_preds = pd.read_parquet(checkpoint_path)
            all_predictions.append(fold_preds)
            
            # Repopulate internal IC history bypassing loop checks
            dates_oos = fold_preds.index.get_level_values('date').unique()
            for date in dates_oos:
                day_slice = fold_preds.loc[date]
                # Fallback implementation directly matching self-tests
                if np.var(day_slice['prob'].values) > 1e-12 and np.var(day_slice['actual_return'].values) > 1e-12:
                    corr, _ = spearmanr(day_slice['prob'].values, day_slice['actual_return'].values)
                    all_fold_ic.append(0.0 if np.isnan(corr) else float(corr))
                else:
                    all_fold_ic.append(0.0)
            continue
            
        # Step 1
        X_train, X_test, y_train, y_test, scaler = get_fold_arrays(fold, df, feature_cols)
        print(f"  [1/5] Arrays extracted. Train={X_train.shape}")
        
        # Step 2
        model = EnsembleModel() if use_ensemble else CatBoostModel()
        model.fit(X_train, y_train)
        print(f"  [2/5] Model trained.")
        
        # Step 3
        X_cal, y_cal = get_cal_arrays(fold, df, feature_cols, scaler)
        calibrator = fit_calibrator(model, X_cal, y_cal)
        
        model_label = "Ensemble" if use_ensemble else "CatBoost"
        plot_reliability_diagram(
            model, calibrator, X_cal, y_cal,
            f"{model_label}_Fold{fold.fold_number:02d}",
            plots_dir / f"{model_label}_fold{fold.fold_number:02d}.png"
        )
        print(f"  [3/5] Calibration complete.")
        
        # Step 4
        fold_preds = build_predictions_df(
            fold, df, actual_returns_df, feature_cols,
            model, calibrator, scaler, model_name=model_label
        )
        print(f"  [4/5] Predictions built: {len(fold_preds)} rows.")
        
        # Step 5
        fold_ic_values = []
        for date, day_slice in fold_preds.groupby(level='date'):
            ic = compute_spearman_ic(
                day_slice['prob'].values,
                day_slice['actual_return'].values
            )
            fold_ic_values.append(ic)
            
        f_mean = np.mean(fold_ic_values)
        f_std = np.std(fold_ic_values) if np.std(fold_ic_values) > 0 else 1.0
        f_icir = f_mean / f_std
        print(f"  [5/5] Fold IC: mean={f_mean:.4f}, ICIR={f_icir:.4f}")
        
        all_predictions.append(fold_preds)
        all_fold_ic.extend(fold_ic_values)
        
        # Save sequence dynamically mapping states incrementally
        fold_preds.to_parquet(checkpoint_path)

    # Finale Configuration Aggregation
    print("\nProcessing final unification buffers...")
    predictions_df = pd.concat(all_predictions).sort_index()
    
    assert not predictions_df.index.duplicated().any(), "Index row duplication critically failed"
    
    all_oos_dates_expected = set()
    for f in folds:
        all_oos_dates_expected.update(f.test_dates)
    
    covered_dates = set(predictions_df.index.get_level_values('date').unique())
    # Subtraction allows for last 2 days naturally dropped because Close(t+2) doesn't exist
    missing = len(all_oos_dates_expected - covered_dates)
    if missing > 5:
        print(f"[WARN] {missing} expected test dates missing from aggregated predictions (usually <3 per fold acceptable).")
        
    ic_results = test_ic_significance(all_fold_ic)
    pd.DataFrame([ic_results]).to_csv(OUTPUT_DIR / "metrics" / "ic_test_results.csv", index=False)
    
    base_file_name = "predictions_ensemble.parquet" if use_ensemble else "predictions_catboost.parquet"
    predictions_df.to_parquet(preds_dir / base_file_name)
    
    print("-" * 60)
    print("Walk-forward loop complete.")
    print(f"Total OOS dates: {predictions_df.index.get_level_values('date').nunique()}")
    print(f"Total OOS rows : {len(predictions_df)}")
    print(f"Overall mean IC: {np.mean(all_fold_ic):.4f}")
    
    return predictions_df, pd.DataFrame([ic_results]), all_fold_ic


# ---------------------------------------------------------------------------
# EXTERNAL DATA / SPY
# ---------------------------------------------------------------------------

def load_spy_returns(start: str, end: str) -> pd.Series:
    """
    Standardize SPY baseline arrays isolated using precalculated persistence files.
    """
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    cache_path = raw_dir / "SPY_ohlcv.parquet"
    
    if cache_path.exists():
        spy_df = pd.read_parquet(cache_path)
    else:
        spy_df = yf.download("SPY", start=start, end=end)
        spy_df.to_parquet(cache_path)
        
    if "Close" in spy_df.columns:
        if isinstance(spy_df.columns, pd.MultiIndex):
            close_col = spy_df['Close']['SPY']
        else:
            close_col = spy_df['Close']
    else:
        # Fallback for simple Series loading
        close_col = spy_df.iloc[:, 0]
        
    spy_returns = close_col.pct_change(1).dropna()
    spy_returns.index = spy_returns.index.tz_localize(None)
    spy_returns.name = 'SPY'
    return spy_returns


# ---------------------------------------------------------------------------
# EXPERIMENT LAUNCHER
# ---------------------------------------------------------------------------

def run_all_experiments(predictions_df: pd.DataFrame, spy_returns: pd.Series) -> None:
    """
    Directly initiate evaluation constraints spanning standard experimental requirements.
    """
    metrics_dir = OUTPUT_DIR / "metrics"
    perm_dir = OUTPUT_DIR / "permutation"
    
    # Ex 1: Primary Comparison
    print("\nRunning Experiment 1: Strategy Comparison...")
    results_df = run_all_strategies(predictions_df, spy_returns)
    results_df.to_csv(metrics_dir / "strategy_comparison.csv", index=False)
    
    print("\nTop 3 Strategies by Sharpe Ratio:")
    print(results_df[['strategy_name', 'annual_return', 'sharpe_ratio']].head(3).to_string(index=False))

    # Ex 2: Isolated Distribution Noise Assessment
    print("\nRunning Experiment 2: Permutation Test (1000 runs)...")
    try:
        topk1_config = [c for c in STRATEGY_CONFIGS if c.name == 'TopK1'][0]
        
        def run_perm_with_progress():
            observed_result = run_backtest(predictions_df, topk1_config)
            obs_sharpe = float(observed_result['sharpe_ratio'])
            pred_sorted = predictions_df.sort_index()
            nulls = []
            
            for i in range(1000):
                if (i + 1) % 100 == 0:
                    print(f"  Permutation {i+1}/1000 complete...")
                
                shuffled = pred_sorted.copy()
                rng = np.random.default_rng(i)
                probs_arr = shuffled['prob'].values.copy()
                sizes = shuffled.groupby(level='date').size().values
                start = 0
                for size in sizes:
                    rng.shuffle(probs_arr[start: start + size])
                    start += size
                shuffled['prob'] = probs_arr
                
                res = run_backtest(shuffled, topk1_config)
                nulls.append(float(res['sharpe_ratio']))
                
            p_val = np.mean(np.array(nulls) >= obs_sharpe)
            rank = scipy.stats.percentileofscore(nulls, obs_sharpe)
            return {'observed_sharpe': obs_sharpe, 'null_sharpes': nulls,
                    'null_mean': float(np.mean(nulls)), 'null_95th': float(np.percentile(nulls, 95)),
                    'p_value': float(p_val), 'percentile_rank': float(rank), 'significant': bool(p_val < 0.05)}
            
        perm_results = run_perm_with_progress()
        
        pd.Series(perm_results['null_sharpes'], name='null_sharpe').to_csv(
            perm_dir / "permutation_topk1.csv", index=False
        )
        
        sum_dict = {k: v for k, v in perm_results.items() if k != 'null_sharpes'}
        pd.DataFrame([sum_dict]).to_csv(perm_dir / "permutation_topk1_summary.csv", index=False)
    except Exception as e:
        print(f"  [WARN] Permutation test disabled or failed: {str(e)}")

    # Ex 3: Sub-period Matrix Check
    print("\nRunning Experiment 3: Sub-period Analysis...")
    key_configs = [c for c in STRATEGY_CONFIGS if c.name in ['TopK1', 'TopK1_Trend', 'Equal_Weight', 'Random_Top1']]
    sub_df = run_subperiod_analysis(predictions_df, key_configs)
    
    # Override SPY handling via subperiod isolated bounds specifically mapping BuyHold
    periods = {
        'Period 1 - ZIRP Bull': ('2015-10-16', '2018-12-31'),
        'Period 2 - COVID/Growth': ('2019-01-01', '2021-12-31'),
        'Period 3 - Rate Shock': ('2022-01-01', '2024-12-31')
    }
    spy_recs = []
    spy_cfg = [c for c in STRATEGY_CONFIGS if c.name == 'BuyHold_SPY'][0]
    for p_name, (start_dt, end_dt) in periods.items():
        mask = (predictions_df.index.get_level_values('date') >= pd.to_datetime(start_dt)) & (predictions_df.index.get_level_values('date') <= pd.to_datetime(end_dt))
        dates_in_period = predictions_df.loc[mask].index.get_level_values('date').unique().tolist()
        if len(dates_in_period) > 0:
            res = run_spy_buyhold(spy_returns, dates_in_period)
            spy_recs.append({'strategy_name': spy_cfg.name, 'sub_period': p_name, 'annual_return': res['annual_return'], 'sharpe_ratio': res['sharpe_ratio'], 'max_drawdown': res['max_drawdown']})
    
    if len(spy_recs) > 0:
        spy_df = pd.DataFrame(spy_recs).set_index(['strategy_name', 'sub_period'])
        sub_df = pd.concat([sub_df, spy_df])
        
    sub_df.to_csv(metrics_dir / "subperiod_analysis.csv")
    print(sub_df.to_string())

    # Ex 4: Execution Delay Breaking Point
    print("\nRunning Experiment 4: Cost Sensitivity...")
    try:
        topk1_cfg = [c for c in STRATEGY_CONFIGS if c.name == 'TopK1'][0]
        cost_df = run_cost_sensitivity(
            predictions_df, topk1_cfg,
            cost_levels_bps=[0, 5, 10, 15, 20, 30, 50]
        )
        cost_df.to_csv(metrics_dir / "cost_sensitivity_topk1.csv")
        print("\nCost Sensitivities:\n")
        print(cost_df[['annual_return', 'sharpe_ratio']].to_string())
    except Exception as e:
        print(f"  [WARN] Cost Sensitivity test failed: {str(e)}")

    # Ex 5: Subset Scaling Dimensions
    print("\nRunning Experiment 5: K Sensitivity...")
    k_configs = [c for c in STRATEGY_CONFIGS if c.name in ['TopK1', 'TopK2', 'TopK3']]
    k_results = [run_backtest(predictions_df, c) for c in k_configs]
    k_df = pd.DataFrame(k_results)[
        ['strategy_name', 'annual_return', 'sharpe_ratio', 'max_drawdown', 'n_trades', 'mean_ic']
    ]
    k_df.to_csv(metrics_dir / "k_sensitivity.csv", index=False)
    print("\n" + k_df.to_string(index=False))

    print("\n" + "=" * 60)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 60)
    print("Results saved to: results/")
    print(" - results/predictions/predictions_ensemble.parquet  [Full Panel Inference]")
    print(" - results/metrics/strategy_comparison.csv           [Main Results Table]")
    print(" - results/metrics/subperiod_analysis.csv            [Cycle Performance]")
    print(" - results/metrics/cost_sensitivity_topk1.csv        [Latency/Fee Bounds]")
    print(" - results/metrics/ic_test_results.csv               [IC Distribution Significance]")
    print(" - results/plots/reliability_diagrams/ (nPNGs)       [Calibration Validations]")
    print(" - results/permutation/permutation_topk1_summary.csv [Null Bootstrap Outperformance]")


# ---------------------------------------------------------------------------
# MAIN START
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Step 0
    dirs_to_make = [
        "predictions",
        "metrics",
        "plots/reliability_diagrams",
        "permutation"
    ]
    for d in dirs_to_make:
        (OUTPUT_DIR / d).mkdir(parents=True, exist_ok=True)

    # Step 1
    print("Step 1/5: Loading data...")
    df = load_all_data(use_cache=True)
    feature_cols = get_feature_columns(df)
    print(f"  Loaded: {len(df)} rows, {len(feature_cols)} features")
    print(f"  Date range: {df.index.get_level_values('date').min().date()} to {df.index.get_level_values('date').max().date()}")

    # Step 2
    print("\nStep 2/5: Generating walk-forward folds...")
    folds = generate_folds(df)
    print_fold_summary(folds)
    print(f"  {len(folds)} folds generated")

    # Step 3
    print("\nStep 3/5: Loading SPY benchmark...")
    spy_returns = load_spy_returns("2015-01-01", "2025-01-01")
    print(f"  SPY: {len(spy_returns)} trading days loaded")

    # Step 4
    print("\nStep 4/5: Running walk-forward training loop...")
    print("  (This will take 30-90 minutes on CPU; automatic progression checkpoints are ACTIVE)")
    
    predictions_df, ic_results, all_fold_ic = run_walk_forward_loop(
        df, feature_cols, folds, use_ensemble=True
    )
    
    ic_sig = bool(ic_results.iloc[0]['significant'])
    
    if not ic_sig:
        print("\n" + "=" * 60)
        print("WARNING: IC test failed. Mean IC is not significant.")
        print("Results will still be saved but the cross-sectional")
        print("ranking contribution needs to be reviewed.")
        print("=" * 60 + "\n")
    else:
        print("\n[PASS] IC test passed. Proceeding to backtesting.\n")

    # Step 5
    print("Step 5/5: Running backtesting experiments...")
    run_all_experiments(predictions_df, spy_returns)

    print("\nrun_experiments.py complete. All results in results/")
