#!/usr/bin/env python3
"""
Validate that pipeline results show meaningful per-ticker variation.
Saves nothing — prints a clear pass/fail summary and exits 0/1.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np


def validate_results():
    # Prefer new centralized results/ but fall back to research_outputs/results
    candidates = [Path('results'), Path('research_outputs/results'), Path('research_outputs/results')]
    results_dir = None
    for c in candidates:
        if c.exists() and any(c.glob('*.csv')):
            results_dir = c
            break
    if results_dir is None:
        print("ERROR: No results directory with CSV files found. Checked: results/, research_outputs/results/")
        return False

    print("="*80)
    print("RESULTS VALIDATION - Checking Per-Ticker Variation")
    print("="*80)

    checks_passed = 0
    total_checks = 6

    # Helper: safe read
    def read_csv_safe(name):
        p = results_dir / name
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")
        return pd.read_csv(p)

    # 1. Walk-Forward Validation
    print("\n1. WALK-FORWARD VALIDATION")
    print("-" * 60)
    df_wf = read_csv_safe('walk_forward_results.csv')
    if 'ticker' not in df_wf.columns or 'accuracy' not in df_wf.columns:
        raise KeyError("walk_forward_results.csv missing required columns ('ticker','accuracy')")

    wf_by_ticker = df_wf.groupby('ticker')['accuracy'].agg(['mean', 'std', 'min', 'max', 'count'])
    print(wf_by_ticker.round(4))

    overall_std = float(df_wf['accuracy'].std())
    print(f"\nOverall std dev: {overall_std:.4f}")
    print(f"Accuracy range: [{df_wf['accuracy'].min():.4f}, {df_wf['accuracy'].max():.4f}]")

    if overall_std >= 0.02:
        print("✅ PASS: Results show meaningful variation (std >= 0.02)")
        checks_passed += 1
    else:
        print("❌ FAIL: Results too uniform (std < 0.02)")

    # 2. Cross-Ticker Generalization
    print("\n2. CROSS-TICKER GENERALIZATION")
    print("-" * 60)
    df_cross = read_csv_safe('cross_ticker_results.csv')
    required = {'test_ticker', 'accuracy', 'roc_auc'}
    if not required.issubset(df_cross.columns):
        raise KeyError(f"cross_ticker_results.csv missing one of {required}")
    print(df_cross[['test_ticker', 'accuracy', 'roc_auc']].round(4))

    cross_std = float(df_cross['accuracy'].std())
    cross_mean = float(df_cross['accuracy'].mean())
    print(f"\nMean accuracy: {cross_mean:.4f} ± {cross_std:.4f}")

    if cross_std >= 0.01 and cross_mean < 0.80:
        print("✅ PASS: Realistic cross-ticker performance")
        checks_passed += 1
    else:
        print("❌ FAIL: Unrealistic generalization (too high or too uniform)")

    # 3. Ablation Studies
    print("\n3. ABLATION STUDIES")
    print("-" * 60)
    df_ablation = read_csv_safe('ablation_studies.csv')
    if not {'ticker', 'feature_set', 'accuracy'}.issubset(df_ablation.columns):
        raise KeyError("ablation_studies.csv missing required columns (ticker, feature_set, accuracy)")

    ablation_pivot = df_ablation.pivot_table(index='ticker', columns='feature_set', values='accuracy')
    print(ablation_pivot.round(4))

    ablation_std = df_ablation.groupby('feature_set')['accuracy'].std()
    print(f"\nStd dev by feature set:")
    print(ablation_std.round(4))

    if ablation_std.mean() >= 0.01:
        print("✅ PASS: Feature sets show different performance")
        checks_passed += 1
    else:
        print("❌ FAIL: All feature sets identical")

    # 4. Statistical Significance
    print("\n4. STATISTICAL SIGNIFICANCE")
    print("-" * 60)
    if 'binom_significant' in df_wf.columns:
        significant_folds = df_wf[df_wf['binom_significant'] == True]
        pct_significant = 100 * len(significant_folds) / max(len(df_wf), 1)
        print(f"Significant folds (p < 0.05): {len(significant_folds)}/{len(df_wf)} ({pct_significant:.1f}%)")
        if pct_significant > 0:
            print("✅ PASS: Some folds achieve statistical significance")
            checks_passed += 1
        else:
            print("❌ FAIL: No significant results")
    else:
        print("⚠️  SKIP: 'binom_significant' column missing in walk_forward_results.csv")

    # 5. Baseline Comparison
    print("\n5. BASELINE MODELS")
    print("-" * 60)
    df_random = read_csv_safe('baseline_random.csv')
    df_lr = read_csv_safe('baseline_logistic_regression.csv')
    df_tech = read_csv_safe('baseline_technical_only.csv')

    # mean per ticker
    baseline_summary = pd.DataFrame({
        'Random': df_random.groupby('ticker')['accuracy'].mean(),
        'Logistic': df_lr.groupby('ticker')['accuracy'].mean(),
        'Technical': df_tech.groupby('ticker')['accuracy'].mean(),
        'WalkForward': df_wf.groupby('ticker')['accuracy'].mean()
    })
    print(baseline_summary.round(4))

    baseline_variation = float(baseline_summary.std(axis=0).mean())
    if baseline_variation > 0.01:
        print(f"✅ PASS: Baselines show variation (std={baseline_variation:.4f})")
        checks_passed += 1
    else:
        print(f"❌ FAIL: Baselines too uniform (std={baseline_variation:.4f})")

    # 6. Backtest Returns
    print("\n6. BACKTEST SIMULATION")
    print("-" * 60)
    df_backtest = read_csv_safe('backtest_results.csv')
    if not {'ticker', 'excess_return', 'strategy_total_return', 'buy_hold_total_return'}.issubset(df_backtest.columns):
        raise KeyError("backtest_results.csv missing required columns")

    print(df_backtest[['ticker', 'strategy_total_return', 'buy_hold_total_return', 'excess_return']].round(4))

    backtest_std = float(df_backtest['excess_return'].std())
    print(f"\nExcess return std dev: {backtest_std:.4f}")

    if backtest_std > 0.01:
        print("✅ PASS: Returns vary across tickers")
        checks_passed += 1
    else:
        print("❌ FAIL: Returns identical across tickers")

    # Final Verdict
    print("\n" + "="*80)
    print(f"VALIDATION SCORE: {checks_passed}/{total_checks} checks passed")
    print("="*80)

    if checks_passed >= 5:
        print("✅ RESULTS VALID - Ready for publication!")
        return True
    elif checks_passed >= 3:
        print("⚠️  RESULTS MIXED - Some variation but investigate further")
        return False
    else:
        print("❌ RESULTS INVALID - Major issues remain")
        return False


if __name__ == '__main__':
    try:
        ok = validate_results()
        sys.exit(0 if ok else 1)
    except Exception as e:
        print("ERROR during validation:", str(e))
        sys.exit(1)
