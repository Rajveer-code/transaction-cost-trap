"""
train_trend_rf.py

Evaluate trend features with RandomForest using walk-forward validation.
Focus on "Precision @ Top 20%" - win rate of the most confident predictions.

Key metric: Top 20% Win Rate vs Buy & Hold Baseline.

Outputs:
 - results/predictions_trend_rf.csv

Usage:
    python scripts/train_trend_rf.py
"""
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]
IN_FILE = ROOT / 'data' / 'extended' / 'features_trend_5d.parquet'
OUT_PRED = ROOT / 'results' / 'predictions_trend_rf.csv'


def precision_at_top_k_percent(y_true, y_proba, k=20):
    """
    Calculate precision for top k% most confident predictions.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities for class 1
        k: Top percentage (default 20)
    
    Returns:
        precision of top k% predictions
    """
    # Sort by confidence (descending)
    sorted_idx = np.argsort(-y_proba)
    
    # Select top k% of samples
    n_top = max(1, int(np.ceil(len(y_true) * k / 100)))
    top_idx = sorted_idx[:n_top]
    
    # Calculate precision on top k%
    if len(top_idx) == 0:
        return np.nan
    
    top_correct = (y_true[top_idx] == 1).sum()
    return top_correct / len(top_idx)


def main():
    if not IN_FILE.exists():
        print('Features file not found:', IN_FILE)
        return

    print('Loading features:', IN_FILE)
    df = pd.read_parquet(IN_FILE)
    
    # Identify columns
    col_names = [c for c in df.columns]
    date_col = next((c for c in col_names if 'date' == c.lower()), None)
    if date_col is None:
        date_col = next((c for c in col_names if 'date' in c.lower()), None)
    
    ticker_col = next((c for c in col_names if 'ticker' == c.lower()), None)
    if ticker_col is None:
        ticker_col = next((c for c in col_names if 'ticker' in c.lower()), None)

    target_col = 'binary_label'
    if target_col not in df.columns:
        raise RuntimeError('binary_label target not found')

    if date_col is None or ticker_col is None:
        raise RuntimeError('Date or Ticker column not found')

    # Ensure sorted
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)

    # Identify feature columns
    id_cols = {date_col, ticker_col, 'target_return_5d', target_col, 'sma_50', 'sma_200'}
    feature_cols = [c for c in df.columns if c not in id_cols]
    # Drop any feature columns that are entirely NaN
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]

    print(f'Using {len(feature_cols)} features for training')
    print(f'Features: {feature_cols}')

    # Containers for results
    all_preds = []
    metrics_rows = []

    tickers = sorted(df[ticker_col].unique())
    for ticker in tickers:
        df_t = df[df[ticker_col] == ticker].copy().reset_index(drop=True)
        X = df_t[feature_cols]
        y = df_t[target_col]

        # Drop rows with NaNs
        mask_valid = X.notna().all(axis=1) & y.notna()
        X = X[mask_valid]
        y = y[mask_valid]
        
        if len(y) < 20:
            print(f'  Skipping {ticker} - insufficient data ({len(y)} rows)')
            continue

        print(f'Processing {ticker} with {len(y)} valid samples...')
        
        tscv = TimeSeriesSplit(n_splits=10)
        model = RandomForestClassifier(
            n_estimators=500,
            min_samples_leaf=50,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )

        preds_t = []
        top20_wrs = []
        baselines = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit model
            model.fit(X_train, y_train)

            # Predictions
            y_hat = model.predict(X_test)
            proba = model.predict_proba(X_test)[:, 1]

            # Buy & Hold baseline: proportion of class 1 in test fold
            baseline = y_test.mean()
            baselines.append(baseline)
            
            # Precision @ Top 20%
            top20_wr = precision_at_top_k_percent(y_test.values, proba, k=20)
            top20_wrs.append(top20_wr)

            preds_block = pd.DataFrame({
                date_col: df_t.loc[X_test.index, date_col].values,
                ticker_col: ticker,
                'fold': fold_idx,
                'y_true': y_test.values,
                'y_pred': y_hat,
                'y_proba': proba,
                'baseline': baseline,
                'top20_win_rate': top20_wr,
            })
            preds_t.append(preds_block)

        if not preds_t:
            continue

        df_preds_t = pd.concat(preds_t, axis=0).reset_index(drop=True)
        
        # Per-ticker metrics
        y_true_t = df_preds_t['y_true'].values
        y_proba_t = df_preds_t['y_proba'].values
        
        try:
            overall_acc = accuracy_score(y_true_t, (y_proba_t > 0.5).astype(int))
            overall_top20 = precision_at_top_k_percent(y_true_t, y_proba_t, k=20)
            baseline_wr = df_preds_t['baseline'].mean()
            improvement = overall_top20 - baseline_wr if not np.isnan(overall_top20) else np.nan
        except Exception as e:
            print(f'  Error computing metrics for {ticker}: {e}')
            continue

        metrics_rows.append({
            'Ticker': ticker,
            'n_test': len(df_preds_t),
            'accuracy': overall_acc,
            'baseline_win_rate': baseline_wr,
            'top20_win_rate': overall_top20,
            'improvement': improvement,
        })

        all_preds.append(df_preds_t)

    if not all_preds:
        print('No predictions were produced.')
        return

    df_all_preds = pd.concat(all_preds, axis=0).reset_index(drop=True)

    # Global metrics
    y_true_global = df_all_preds['y_true'].values
    y_proba_global = df_all_preds['y_proba'].values
    
    global_acc = accuracy_score(y_true_global, (y_proba_global > 0.5).astype(int))
    global_top20 = precision_at_top_k_percent(y_true_global, y_proba_global, k=20)
    global_baseline = df_all_preds['baseline'].mean()
    global_improvement = global_top20 - global_baseline if not np.isnan(global_top20) else np.nan

    # Print results table per ticker
    metrics_df = pd.DataFrame(metrics_rows).sort_values('Ticker').reset_index(drop=True)
    print('\n' + '='*110)
    print('Results Table (per ticker):')
    print('='*110)
    print(metrics_df.to_string(index=False))

    print('\n' + '='*110)
    print('Global Performance:')
    print('='*110)
    print(f'  overall_accuracy: {global_acc:.6f}')
    print(f'  buy_hold_baseline (% positive days): {global_baseline:.6f}')
    print(f'  top_20_percent_win_rate: {global_top20:.6f}')
    print(f'  improvement_vs_baseline: {global_improvement:.6f}')
    print(f'  total_test_samples: {len(df_all_preds)}')

    # Save predictions
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    df_all_preds.to_csv(OUT_PRED, index=False)
    print(f'\nSaved predictions -> {OUT_PRED}')


if __name__ == '__main__':
    main()
