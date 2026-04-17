"""
parallel_permutation.py
=======================
Drop-in replacement for the permutation test block in run_experiments.py.
Uses all CPU cores via joblib. On i7-13650HX (14 cores), expect ~6-8x speedup
vs single-threaded: ~30 min -> ~4-5 min.

GPU (RTX 4060) is NOT used here — the backtest is numpy/pandas CPU work,
not a tensor operation. The CPU parallelism is the right tool.

HOW TO USE:
  In run_experiments.py, replace the entire permutation test block (Ex 2)
  with a call to run_parallel_permutation_test().

Author: Rajveer Singh Pall
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List
import time


def _single_permutation_worker(seed: int, pred_sorted_values: np.ndarray,
                                pred_sorted_index, group_sizes: np.ndarray,
                                topk1_config) -> float:
    """
    One permutation iteration. Designed to be called in parallel.
    Accepts pre-extracted numpy arrays to minimise pickling overhead.
    """
    from src.backtesting.backtester import run_backtest

    # Reconstruct minimal DataFrame needed by run_backtest
    probs_arr = pred_sorted_values[:, 0].copy()  # prob column
    rng = np.random.default_rng(seed)

    # Within-day shuffle of probabilities
    start = 0
    for size in group_sizes:
        rng.shuffle(probs_arr[start: start + size])
        start += size

    # Rebuild DataFrame with shuffled probs
    shuffled_values = pred_sorted_values.copy()
    shuffled_values[:, 0] = probs_arr

    shuffled = pd.DataFrame(
        shuffled_values,
        index=pred_sorted_index,
        columns=['prob', 'Close', 'SMA_200', 'target', 'trailing_return_21d', 'actual_return']
    )
    shuffled['model'] = 'Permutation'

    res = run_backtest(shuffled, topk1_config)
    return float(res['sharpe_ratio'])


def run_parallel_permutation_test(
    predictions_df: pd.DataFrame,
    topk1_config,
    n_permutations: int = 1000,
    n_jobs: int = -1,
) -> dict:
    """
    Run permutation test in parallel across all CPU cores.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Full OOS predictions (MultiIndex date, ticker).
    topk1_config : StrategyConfig
        TopK1 strategy config object from STRATEGY_CONFIGS.
    n_permutations : int
        Number of permutations. Default 1000.
    n_jobs : int
        Number of parallel workers. -1 = use all cores.
        On i7-13650HX: 14 cores -> set n_jobs=12 to leave 2 for OS.

    Returns
    -------
    dict with keys: observed_sharpe, null_sharpes, null_mean,
                    null_95th, p_value, percentile_rank, significant
    """
    try:
        from joblib import Parallel, delayed
    except ImportError:
        raise ImportError("joblib not found. Run: pip install joblib")

    import scipy.stats
    from src.backtesting.backtester import run_backtest

    print(f"\nRunning Experiment 2: Permutation Test ({n_permutations} runs, parallel)...")

    # Compute observed Sharpe
    observed_result = run_backtest(predictions_df, topk1_config)
    obs_sharpe = float(observed_result['sharpe_ratio'])
    print(f"  Observed TopK1 Sharpe: {obs_sharpe:.4f}")

    # Pre-extract arrays to minimise per-worker pickling cost
    pred_sorted = predictions_df.sort_index()

    # Extract numeric columns only (model column is string, handle separately)
    numeric_cols = ['prob', 'Close', 'SMA_200', 'target', 'trailing_return_21d', 'actual_return']
    pred_values = pred_sorted[numeric_cols].values.astype(np.float64)
    pred_index = pred_sorted.index

    # Pre-compute group sizes (same for every permutation)
    group_sizes = pred_sorted.groupby(level='date').size().values.astype(np.int64)

    print(f"  Total rows: {len(pred_sorted):,} | Days: {len(group_sizes):,} | "
          f"Workers: {'all cores' if n_jobs == -1 else n_jobs}")
    print(f"  Starting parallel permutations...")

    t0 = time.time()

    # Run in parallel — joblib handles Windows spawn correctly
    # verbose=5 prints progress every ~10% of jobs
    nulls: List[float] = Parallel(
        n_jobs=n_jobs,
        backend='loky',          # most robust backend, works on Windows
        verbose=5,               # prints progress batches
        batch_size='auto',       # joblib decides optimal batch size
    )(
        delayed(_single_permutation_worker)(
            i, pred_values, pred_index, group_sizes, topk1_config
        )
        for i in range(n_permutations)
    )

    elapsed = time.time() - t0
    print(f"\n  Completed {n_permutations} permutations in {elapsed:.1f}s "
          f"({elapsed/n_permutations*1000:.1f}ms/iter)")

    nulls_arr = np.array(nulls)
    p_val = float(np.mean(nulls_arr >= obs_sharpe))
    rank = float(scipy.stats.percentileofscore(nulls_arr, obs_sharpe))

    print(f"  Null mean Sharpe   : {nulls_arr.mean():.4f}")
    print(f"  Null 95th pctile   : {np.percentile(nulls_arr, 95):.4f}")
    print(f"  p-value            : {p_val:.4f}")
    print(f"  Significant (p<0.05): {p_val < 0.05}")

    return {
        'observed_sharpe': obs_sharpe,
        'null_sharpes': nulls,
        'null_mean': float(nulls_arr.mean()),
        'null_95th': float(np.percentile(nulls_arr, 95)),
        'p_value': p_val,
        'percentile_rank': rank,
        'significant': bool(p_val < 0.05),
    }
