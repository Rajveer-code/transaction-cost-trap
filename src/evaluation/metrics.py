"""
Evaluation metrics module for financial sentiment ML models.

Provides robust metrics computation with proper edge case handling,
bootstrap confidence intervals, and statistical significance testing.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve
)
import warnings


def compute_metrics(y_true, y_pred, y_pred_proba=None, threshold=0.5):
    """
    Compute classification metrics with edge case handling.
    
    Parameters
    ----------
    y_true : array-like
        True binary labels
    y_pred : array-like
        Predicted binary labels (already thresholded)
    y_pred_proba : array-like, optional
        Predicted probabilities for positive class
    threshold : float
        Decision threshold (used for reference, not applied here since y_pred is pre-thresholded)
    
    Returns
    -------
    dict
        Dictionary of metrics (accuracy, ROC-AUC, precision, recall, F1)
        Returns NaN for metrics that cannot be computed (e.g., single-class folds)
    """
    metrics = {
        'accuracy': np.nan,
        'roc_auc': np.nan,
        'precision': np.nan,
        'recall': np.nan,
        'f1': np.nan,
        'n_samples': len(y_true),
        'n_positive': int(np.sum(y_true)),
        'n_negative': int(np.sum(1 - y_true))
    }
    
    # Check for single-class folds
    if len(np.unique(y_true)) < 2:
        warnings.warn(f"Single-class fold detected: {np.unique(y_true)}. Metrics set to NaN.")
        return metrics
    
    # Accuracy
    try:
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
    except Exception as e:
        warnings.warn(f"Could not compute accuracy: {e}")
    
    # ROC-AUC (requires probabilities)
    if y_pred_proba is not None and len(np.unique(y_true)) >= 2:
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        except Exception as e:
            warnings.warn(f"Could not compute ROC-AUC: {e}")
    
    # Precision
    try:
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    except Exception as e:
        warnings.warn(f"Could not compute precision: {e}")
    
    # Recall
    try:
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    except Exception as e:
        warnings.warn(f"Could not compute recall: {e}")
    
    # F1
    try:
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    except Exception as e:
        warnings.warn(f"Could not compute F1: {e}")
    
    return metrics


def binomial_significance_test(accuracy, n_samples, baseline=0.5, alpha=0.05, alternative='greater'):
    """
    One-sided binomial test for accuracy significance.
    
    H0: accuracy = 0.5 (random baseline)
    H1: accuracy > 0.5 (our model beats random)
    
    Parameters
    ----------
    accuracy : float
        Observed accuracy
    n_samples : int
        Number of test samples
    baseline : float
        Null hypothesis accuracy (default 0.5)
    alpha : float
        Significance level
    alternative : str
        'greater', 'less', or 'two-sided'
    
    Returns
    -------
    dict
        Dictionary with p-value, is_significant, and interpretation
    """
    n_correct = int(np.round(accuracy * n_samples))
    # Handle scipy version differences (binomtest vs binom_test)
    if hasattr(stats, 'binomtest'):
        p_value = stats.binomtest(n_correct, n_samples, baseline, alternative=alternative).pvalue
    else:
        p_value = stats.binom_test(n_correct, n_samples, baseline, alternative=alternative)
    
    return {
        'p_value': p_value,
        'is_significant': p_value < alpha,
        'n_correct': n_correct,
        'n_samples': n_samples,
        'accuracy': accuracy,
        'interpretation': f"p={p_value:.4f}. " + 
            ("Accuracy significantly > 50%" if p_value < alpha else "Cannot reject H0: accuracy = 50%")
    }


def bootstrap_confidence_interval(y_true, y_pred, y_pred_proba=None, metric='accuracy', 
                                   n_iterations=1000, ci=0.95, random_state=42):
    """
    Compute bootstrap confidence intervals for metrics.
    
    Handles missing values (NaN) properly by ignoring them.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    y_pred_proba : array-like, optional
        Predicted probabilities
    metric : str
        Metric to compute: 'accuracy', 'roc_auc', 'precision', 'recall', 'f1'
    n_iterations : int
        Number of bootstrap samples
    ci : float
        Confidence interval level (0.95 = 95%)
    random_state : int
        Random seed
    
    Returns
    -------
    dict
        Dictionary with mean, lower, upper bounds and bootstrap samples
    """
    np.random.seed(random_state)
    
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Remove rows with NaN values
    valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[valid_idx]
    y_pred_clean = y_pred[valid_idx]
    
    if y_pred_proba is not None:
        y_pred_proba = np.asarray(y_pred_proba).flatten()
        y_pred_proba_clean = y_pred_proba[valid_idx]
    else:
        y_pred_proba_clean = None
    
    if len(y_true_clean) < 2:
        return {
            'mean': np.nan,
            'lower': np.nan,
            'upper': np.nan,
            'n_samples': len(y_true_clean),
            'bootstrap_samples': []
        }
    
    bootstrap_scores = []
    
    for _ in range(n_iterations):
        # Resample with replacement
        idx = np.random.choice(len(y_true_clean), size=len(y_true_clean), replace=True)
        y_true_boot = y_true_clean[idx]
        y_pred_boot = y_pred_clean[idx]
        
        if y_pred_proba_clean is not None:
            y_pred_proba_boot = y_pred_proba_clean[idx]
        else:
            y_pred_proba_boot = None
        
        # Check for single-class fold
        if len(np.unique(y_true_boot)) < 2:
            continue
        
        # Compute metric
        try:
            if metric == 'accuracy':
                score = accuracy_score(y_true_boot, y_pred_boot)
            elif metric == 'roc_auc':
                if y_pred_proba_boot is not None:
                    score = roc_auc_score(y_true_boot, y_pred_proba_boot)
                else:
                    continue
            elif metric == 'precision':
                score = precision_score(y_true_boot, y_pred_boot, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true_boot, y_pred_boot, zero_division=0)
            elif metric == 'f1':
                score = f1_score(y_true_boot, y_pred_boot, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            bootstrap_scores.append(score)
        except Exception as e:
            warnings.warn(f"Could not compute {metric} for bootstrap sample: {e}")
            continue
    
    if len(bootstrap_scores) == 0:
        return {
            'mean': np.nan,
            'lower': np.nan,
            'upper': np.nan,
            'n_samples': len(y_true_clean),
            'bootstrap_samples': []
        }
    
    bootstrap_scores = np.array(bootstrap_scores)
    mean = np.mean(bootstrap_scores)
    lower = np.percentile(bootstrap_scores, (1 - ci) / 2 * 100)
    upper = np.percentile(bootstrap_scores, (1 + ci) / 2 * 100)
    
    return {
        'mean': mean,
        'lower': lower,
        'upper': upper,
        'n_samples': len(y_true_clean),
        'bootstrap_samples': bootstrap_scores.tolist()
    }


def find_optimal_threshold(y_true, y_pred_proba, metric='f1', penalty=0.1):
    """
    Find optimal decision threshold using penalized metric to prevent degenerate solutions.
    
    Parameters
    ----------
    y_true : array-like
        True labels
    y_pred_proba : array-like
        Predicted probabilities
    metric : str
        Metric to optimize: 'f1' or 'balanced_accuracy'
    penalty : float
        Penalty for extreme thresholds (0, 1) to prevent degenerate solutions
    
    Returns
    -------
    dict
        Dictionary with optimal threshold, metric value, and threshold curve
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_proba = np.asarray(y_pred_proba).flatten()
    
    # Remove NaN values
    valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred_proba))
    y_true_clean = y_true[valid_idx]
    y_pred_proba_clean = y_pred_proba[valid_idx]
    
    if len(np.unique(y_true_clean)) < 2:
        return {
            'optimal_threshold': 0.5,
            'metric_value': np.nan,
            'thresholds': [],
            'metric_values': []
        }
    
    thresholds = np.arange(0, 1.01, 0.01)
    metric_values = []
    
    for thresh in thresholds:
        y_pred_thresh = (y_pred_proba_clean >= thresh).astype(int)
        
        # Check for single-class predictions
        if len(np.unique(y_pred_thresh)) < 2:
            # Penalize degenerate solutions
            if metric == 'f1':
                score = -penalty
            else:
                score = 0.5 - penalty
        else:
            if metric == 'f1':
                score = f1_score(y_true_clean, y_pred_thresh, zero_division=0)
            else:  # balanced_accuracy
                tn = np.sum((y_pred_thresh == 0) & (y_true_clean == 0))
                tp = np.sum((y_pred_thresh == 1) & (y_true_clean == 1))
                fn = np.sum((y_pred_thresh == 0) & (y_true_clean == 1))
                fp = np.sum((y_pred_thresh == 1) & (y_true_clean == 0))
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = (sens + spec) / 2
        
        metric_values.append(score)
    
    best_idx = np.nanargmax(metric_values)
    optimal_threshold = thresholds[best_idx]
    
    return {
        'optimal_threshold': optimal_threshold,
        'metric_value': metric_values[best_idx],
        'thresholds': thresholds.tolist(),
        'metric_values': metric_values
    }


def diebold_mariano_test(y_true, y_pred_proba1, y_pred_proba2, loss='squared_error', 
                          small_sample_correction=True):
    """
    Diebold-Mariano test to compare two forecasting models.
    
    Tests H0: MSE1 = MSE2 (models have equal forecast accuracy)
    
    Parameters
    ----------
    y_true : array-like
        True labels (0/1)
    y_pred_proba1 : array-like
        Predicted probabilities from model 1
    y_pred_proba2 : array-like
        Predicted probabilities from model 2
    loss : str
        Loss function: 'squared_error' or 'absolute_error'
    small_sample_correction : bool
        Apply small-sample correction (Harvey, Leybourne, Newbold 1997)
    
    Returns
    -------
    dict
        Dictionary with DM statistic, p-value, and interpretation
    """
    y_true = np.asarray(y_true).flatten()
    y_pred_proba1 = np.asarray(y_pred_proba1).flatten()
    y_pred_proba2 = np.asarray(y_pred_proba2).flatten()
    
    # Remove NaN values
    valid_idx = ~(np.isnan(y_true) | np.isnan(y_pred_proba1) | np.isnan(y_pred_proba2))
    y_true_clean = y_true[valid_idx]
    y_pred_proba1_clean = y_pred_proba1[valid_idx]
    y_pred_proba2_clean = y_pred_proba2[valid_idx]
    
    n = len(y_true_clean)
    
    if n < 2:
        return {
            'dm_statistic': np.nan,
            'p_value': np.nan,
            'interpretation': 'Insufficient samples for test'
        }
    
    # Compute loss for each prediction
    if loss == 'squared_error':
        loss1 = (y_true_clean - y_pred_proba1_clean) ** 2
        loss2 = (y_true_clean - y_pred_proba2_clean) ** 2
    else:  # absolute_error
        loss1 = np.abs(y_true_clean - y_pred_proba1_clean)
        loss2 = np.abs(y_true_clean - y_pred_proba2_clean)
    
    # Compute loss differential
    d = loss1 - loss2
    mean_d = np.mean(d)
    
    # Variance of loss differential (Newey-West estimator with lag 1)
    cov0 = np.mean(d ** 2)
    cov1 = np.mean(d[:-1] * d[1:]) if n > 1 else 0
    var_d = cov0 + 2 * cov1
    
    if var_d <= 0:
        return {
            'dm_statistic': np.nan,
            'p_value': np.nan,
            'interpretation': 'Variance non-positive; test inconclusive'
        }
    
    # DM statistic
    dm_stat = mean_d / np.sqrt(var_d / n)
    
    # P-value (two-sided test)
    if small_sample_correction:
        # Harvey, Leybourne, Newbold (1997) correction
        dof = n - 1
        t_crit = stats.t.ppf(0.975, dof)
        dm_stat_corrected = dm_stat * np.sqrt((dof + 1 - 2 + dof ** (-1)) / dof)
        p_value = 2 * (1 - stats.t.cdf(np.abs(dm_stat_corrected), dof))
    else:
        p_value = 2 * (1 - stats.norm.cdf(np.abs(dm_stat)))
    
    return {
        'dm_statistic': dm_stat,
        'p_value': p_value,
        'mse1': np.mean(loss1),
        'mse2': np.mean(loss2),
        'interpretation': f"DM={dm_stat:.4f}, p={p_value:.4f}. " +
            ("Model 1 significantly better" if p_value < 0.05 and dm_stat > 0 else
             "Model 2 significantly better" if p_value < 0.05 and dm_stat < 0 else
             "No significant difference")
    }


if __name__ == '__main__':
    # Quick test
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, 100)
    y_pred = np.random.binomial(1, 0.55, 100)
    y_pred_proba = np.random.uniform(0, 1, 100)
    
    metrics = compute_metrics(y_true, y_pred, y_pred_proba)
    print("Metrics:", metrics)
    
    binom_test = binomial_significance_test(0.55, 100)
    print("\nBinomial test:", binom_test)
    
    ci = bootstrap_confidence_interval(y_true, y_pred, y_pred_proba, metric='accuracy')
    print("\nBootstrap CI:", ci)
