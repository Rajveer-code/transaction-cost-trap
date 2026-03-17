"""
Statistical Significance Tests - Phase 5 (Day 32-33)
====================================================
Rigorous statistical validation of model performance.

Tests:
1. T-Test: Model accuracy vs random baseline (50%)
2. McNemar's Test: Compare CatBoost vs Baseline
3. Bootstrapped Confidence Intervals
4. Permutation Test: Verify model learns real patterns
"""

import os
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Tuple
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append("src/evaluation")
from time_series_cv import TimeSeriesSplit


class SignificanceTests:
    """
    Statistical significance testing for model validation.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize significance tester.
        
        Args:
            alpha: Significance level (default: 0.05)
        """
        self.alpha = alpha
        self.results = {}
    
    def load_cv_predictions(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate CV predictions for statistical tests.
        
        Returns:
            Tuple of (y_true, y_pred_catboost, y_pred_baseline)
        """
        print("📂 Loading data and generating CV predictions...")
        
        # Load data
        df = pd.read_csv("data/final/model_ready_full.csv")
        df = df.sort_values('date').reset_index(drop=True)
        
        # Load models
        with open("models/catboost_best.pkl", "rb") as f:
            model_catboost = pickle.load(f)
        
        with open("models/baseline_lr.pkl", "rb") as f:
            model_baseline = pickle.load(f)
        
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        
        # Prepare features
        exclude_cols = ['date', 'ticker', 'movement']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['movement'].values
        
        # Load CV config
        cv_config_path = Path("models/cv_config.json")
        if cv_config_path.exists():
            with open(cv_config_path, 'r') as f:
                cfg = json.load(f)
            n_splits = cfg['n_splits']
            test_size = cfg['test_size']
        else:
            n_splits = 5
            test_size = 30
        
        # Generate CV predictions
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
        
        y_true_all = []
        y_pred_catboost_all = []
        y_pred_baseline_all = []
        
        print(f"\n🔄 Generating CV predictions ({n_splits} folds)...")
        
        fold = 1
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # CatBoost predictions
            model_catboost.fit(X_train, y_train, verbose=False)
            y_pred_cat = model_catboost.predict(X_test)
            
            # Baseline predictions (scaled)
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model_baseline.fit(X_train_scaled, y_train)
            y_pred_base = model_baseline.predict(X_test_scaled)
            
            y_true_all.extend(y_test)
            y_pred_catboost_all.extend(y_pred_cat)
            y_pred_baseline_all.extend(y_pred_base)
            
            print(f"   Fold {fold}/{n_splits} complete")
            fold += 1
        
        print(f"✅ Generated {len(y_true_all)} CV predictions")
        
        return (
            np.array(y_true_all),
            np.array(y_pred_catboost_all),
            np.array(y_pred_baseline_all)
        )
    
    def test_accuracy_vs_random(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict:
        """
        One-sample t-test: Model accuracy vs 50% (random baseline).
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary with test results
        """
        print(f"\n{'='*60}")
        print("TEST 1: ACCURACY VS RANDOM BASELINE (50%)")
        print(f"{'='*60}\n")
        
        # Calculate per-sample correctness
        correct = (y_true == y_pred).astype(int)
        
        # Calculate accuracy
        accuracy = correct.mean()
        
        # One-sample t-test against 0.5
        t_stat, p_value = stats.ttest_1samp(correct, 0.5)
        
        # Effect size (Cohen's d)
        cohen_d = (accuracy - 0.5) / correct.std()
        
        result = {
            'test': 'One-sample t-test (vs 50%)',
            'accuracy': float(accuracy),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'cohen_d': float(cohen_d),
            'significant': p_value < self.alpha,
            'n_samples': len(y_true)
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Cohen's d: {cohen_d:.4f}")
        print(f"Significant: {'✅ YES' if result['significant'] else '❌ NO'} (α={self.alpha})")
        
        self.results['accuracy_vs_random'] = result
        return result
    
    def mcnemar_test(
        self,
        y_true: np.ndarray,
        y_pred1: np.ndarray,
        y_pred2: np.ndarray,
        model1_name: str = "CatBoost",
        model2_name: str = "Baseline"
    ) -> Dict:
        """
        McNemar's test: Compare two models on same data.
        
        Args:
            y_true: True labels
            y_pred1: Predictions from model 1
            y_pred2: Predictions from model 2
            model1_name: Name of model 1
            model2_name: Name of model 2
            
        Returns:
            Dictionary with test results
        """
        print(f"\n{'='*60}")
        print(f"TEST 2: MCNEMAR'S TEST ({model1_name} vs {model2_name})")
        print(f"{'='*60}\n")
        
        # Create contingency table
        correct1 = (y_true == y_pred1)
        correct2 = (y_true == y_pred2)
        
        # McNemar contingency table
        # [[both correct, model1 wrong & model2 correct],
        #  [model1 correct & model2 wrong, both wrong]]
        
        both_correct = np.sum(correct1 & correct2)
        only_model1 = np.sum(correct1 & ~correct2)
        only_model2 = np.sum(~correct1 & correct2)
        both_wrong = np.sum(~correct1 & ~correct2)
        
        contingency_table = np.array([
            [both_correct, only_model2],
            [only_model1, both_wrong]
        ])
        
        # McNemar's test (uses only off-diagonal elements)
        if only_model1 + only_model2 > 0:
            chi2_stat = (abs(only_model1 - only_model2) - 1) ** 2 / (only_model1 + only_model2)
            p_value = 1 - stats.chi2.cdf(chi2_stat, df=1)
        else:
            chi2_stat = 0.0
            p_value = 1.0
        
        result = {
            'test': "McNemar's test",
            'model1': model1_name,
            'model2': model2_name,
            'chi2_statistic': float(chi2_stat),
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'contingency_table': contingency_table.tolist(),
            'interpretation': {
                'both_correct': int(both_correct),
                'only_model1_correct': int(only_model1),
                'only_model2_correct': int(only_model2),
                'both_wrong': int(both_wrong)
            }
        }
        
        print(f"Contingency Table:")
        print(f"  Both correct: {both_correct}")
        print(f"  Only {model1_name} correct: {only_model1}")
        print(f"  Only {model2_name} correct: {only_model2}")
        print(f"  Both wrong: {both_wrong}")
        print(f"\nχ² statistic: {chi2_stat:.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Significant: {'✅ YES' if result['significant'] else '❌ NO'} (α={self.alpha})")
        
        self.results['mcnemar_test'] = result
        return result
    
    def bootstrap_confidence_intervals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_iterations: int = 1000,
        confidence: float = 0.95
    ) -> Dict:
        """
        Bootstrap confidence intervals for accuracy and F1.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            n_iterations: Number of bootstrap samples
            confidence: Confidence level
            
        Returns:
            Dictionary with confidence intervals
        """
        print(f"\n{'='*60}")
        print(f"TEST 3: BOOTSTRAP CONFIDENCE INTERVALS")
        print(f"{'='*60}\n")
        
        from sklearn.metrics import accuracy_score, f1_score
        from sklearn.utils import resample
        
        print(f"Running {n_iterations} bootstrap iterations...")
        
        accuracies = []
        f1_scores = []
        
        for i in range(n_iterations):
            # Resample with replacement
            indices = resample(range(len(y_true)), replace=True, random_state=i)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]
            
            # Calculate metrics
            accuracies.append(accuracy_score(y_true_boot, y_pred_boot))
            f1_scores.append(f1_score(y_true_boot, y_pred_boot))
            
            if (i + 1) % 200 == 0:
                print(f"   Completed {i + 1}/{n_iterations} iterations")
        
        # Calculate confidence intervals
        alpha = 1 - confidence
        
        acc_lower = np.percentile(accuracies, alpha/2 * 100)
        acc_upper = np.percentile(accuracies, (1 - alpha/2) * 100)
        
        f1_lower = np.percentile(f1_scores, alpha/2 * 100)
        f1_upper = np.percentile(f1_scores, (1 - alpha/2) * 100)
        
        result = {
            'test': 'Bootstrap confidence intervals',
            'n_iterations': n_iterations,
            'confidence_level': confidence,
            'accuracy': {
                'mean': float(np.mean(accuracies)),
                'lower': float(acc_lower),
                'upper': float(acc_upper),
                'ci': f"[{acc_lower:.4f}, {acc_upper:.4f}]"
            },
            'f1_score': {
                'mean': float(np.mean(f1_scores)),
                'lower': float(f1_lower),
                'upper': float(f1_upper),
                'ci': f"[{f1_lower:.4f}, {f1_upper:.4f}]"
            }
        }
        
        print(f"\n✅ Bootstrap complete!")
        print(f"\nAccuracy:")
        print(f"  Mean: {np.mean(accuracies):.4f}")
        print(f"  {confidence*100:.0f}% CI: [{acc_lower:.4f}, {acc_upper:.4f}]")
        print(f"\nF1-Score:")
        print(f"  Mean: {np.mean(f1_scores):.4f}")
        print(f"  {confidence*100:.0f}% CI: [{f1_lower:.4f}, {f1_upper:.4f}]")
        
        self.results['bootstrap_ci'] = result
        return result
    
    def permutation_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        y_pred: np.ndarray,
        n_permutations: int = 1000
    ) -> Dict:
        """
        Permutation test: Verify model learns real patterns.
        
        Args:
            X: Feature matrix
            y: True labels
            y_pred: Model predictions
            n_permutations: Number of permutations
            
        Returns:
            Dictionary with test results
        """
        print(f"\n{'='*60}")
        print(f"TEST 4: PERMUTATION TEST")
        print(f"{'='*60}\n")
        
        from sklearn.metrics import accuracy_score
        
        # True accuracy
        true_accuracy = accuracy_score(y, y_pred)
        
        print(f"True accuracy: {true_accuracy:.4f}")
        print(f"Running {n_permutations} permutations...")
        
        # Permutation scores
        perm_scores = []
        
        for i in range(n_permutations):
            # Shuffle labels
            y_perm = np.random.RandomState(seed=i).permutation(y)
            
            # Calculate accuracy with shuffled labels
            perm_accuracy = accuracy_score(y_perm, y_pred)
            perm_scores.append(perm_accuracy)
            
            if (i + 1) % 200 == 0:
                print(f"   Completed {i + 1}/{n_permutations} permutations")
        
        # Calculate p-value
        p_value = (np.sum(np.array(perm_scores) >= true_accuracy) + 1) / (n_permutations + 1)
        
        result = {
            'test': 'Permutation test',
            'true_accuracy': float(true_accuracy),
            'n_permutations': n_permutations,
            'p_value': float(p_value),
            'significant': p_value < self.alpha,
            'permutation_mean': float(np.mean(perm_scores)),
            'permutation_std': float(np.std(perm_scores))
        }
        
        print(f"\n✅ Permutation test complete!")
        print(f"\nTrue accuracy: {true_accuracy:.4f}")
        print(f"Permutation mean: {np.mean(perm_scores):.4f}")
        print(f"p-value: {p_value:.6f}")
        print(f"Significant: {'✅ YES' if result['significant'] else '❌ NO'} (α={self.alpha})")
        
        self.results['permutation_test'] = result
        return result
    
    def save_results(self, output_dir: str = "results/metrics"):
        """
        Save all test results to CSV and JSON.
        Converts NumPy types to native Python types to avoid JSON errors.
        """
        os.makedirs(output_dir, exist_ok=True)

        print(f"\n💾 Saving statistical test results...")

        # ---- FIX: Convert all numpy types to Python native types ----
        def convert(o):
            if isinstance(o, (np.integer, np.int64)): return int(o)
            if isinstance(o, (np.floating, np.float64)): return float(o)
            if isinstance(o, (np.bool_, bool)): return bool(o)
            if isinstance(o, dict): return {k: convert(v) for k, v in o.items()}
            if isinstance(o, list): return [convert(i) for i in o]
            return o

        clean_results = convert(self.results)
        # --------------------------------------------------------------

        # Save JSON
        json_path = f"{output_dir}/statistical_tests.json"
        with open(json_path, 'w') as f:
            json.dump(clean_results, f, indent=2)

        print(f"   ✅ Saved JSON: {json_path}")

        # Create summary table
        summary_data = []

        for test_name, test_result in clean_results.items():
            row = {
                'test': test_result.get('test', test_name),
                'statistic': test_result.get('t_statistic') or 
                             test_result.get('chi2_statistic') or 
                             'N/A',
                'p_value': test_result.get('p_value', 'N/A'),
                'significant': '✅' if test_result.get('significant', False) else '❌'
            }
            summary_data.append(row)

        summary_df = pd.DataFrame(summary_data)
        csv_path = f"{output_dir}/statistical_tests.csv"
        summary_df.to_csv(csv_path, index=False)

        print(f"   ✅ Saved CSV: {csv_path}")
        print(f"\n📊 Statistical Tests Summary:")
        print(summary_df.to_string(index=False))

def main():
    """
    Main execution for statistical significance tests.
    """
    print("\n" + "="*60)
    print("PHASE 5 - DAY 32-33: STATISTICAL SIGNIFICANCE TESTS")
    print("="*60 + "\n")
    
    # Initialize tester
    tester = SignificanceTests(alpha=0.05)
    
    # Load CV predictions
    y_true, y_pred_catboost, y_pred_baseline = tester.load_cv_predictions()
    
    # Run all tests
    
    # Test 1: Accuracy vs Random
    tester.test_accuracy_vs_random(y_true, y_pred_catboost)
    
    # Test 2: McNemar's Test
    tester.mcnemar_test(y_true, y_pred_catboost, y_pred_baseline)
    
    # Test 3: Bootstrap CI
    tester.bootstrap_confidence_intervals(y_true, y_pred_catboost, n_iterations=1000)
    
    # Test 4: Permutation Test
    # Load X for permutation test
    df = pd.read_csv("data/final/model_ready_full.csv")
    df = df.sort_values('date').reset_index(drop=True)
    exclude_cols = ['date', 'ticker', 'movement']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].values
    
    tester.permutation_test(X, y_true, y_pred_catboost, n_permutations=1000)
    
    # Save all results
    tester.save_results()
    
    print("\n" + "="*60)
    print("✅ STATISTICAL TESTS COMPLETE!")
    print("="*60)
    print("\nOutputs saved:")
    print("  • results/metrics/statistical_tests.json")
    print("  • results/metrics/statistical_tests.csv")


if __name__ == "__main__":
    main()