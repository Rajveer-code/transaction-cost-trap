"""
train_model_5d_optimized.py

Evaluate the 5-day horizon dataset with optimized CatBoost config:
- iterations=1500, learning_rate=0.01, depth=4, l2_leaf_reg=5
- NO auto_class_weights (learns bull market prior naturally)
- Per-ticker walk-forward validation (10 folds)
- Win Rate (Precision of Class 1) vs Buy & Hold (Baseline Class 1 %)

Outputs:
 - results/predictions_5d_optimized.csv

Usage:
    python scripts/train_model_5d_optimized.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False
    warnings.warn('CatBoost not available; falling back to RandomForestClassifier')
    from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
IN_FILE = ROOT / 'data' / 'extended' / 'features_5d_horizon.parquet'
OUT_PRED = ROOT / 'results' / 'predictions_5d_optimized.csv'


def choose_model(random_state=42):
    """Create optimized CatBoost model."""
    if HAS_CATBOOST:
        return CatBoostClassifier(
            iterations=1500,
            learning_rate=0.01,
            depth=4,
            l2_leaf_reg=5,
            loss_function='Logloss',
            eval_metric='AUC',
            random_state=random_state,
            verbose=0
        )
    warnings.warn('CatBoost not available; falling back to RandomForestClassifier')
    return RandomForestClassifier(n_estimators=300, random_state=random_state, max_depth=8)


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
    id_cols = {date_col, ticker_col, 'target_return_5d', target_col}
    feature_cols = [c for c in df.columns if c not in id_cols]
    # Drop any feature columns that are entirely NaN
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]

    print(f'Using {len(feature_cols)} features for training')

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
        model = choose_model(random_state=42)

        preds_t = []
        baseline_pcts = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit model
            model.fit(X_train, y_train)

            # Predictions
            y_hat = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
            else:
                proba = y_hat.astype(float)

            # Buy & Hold baseline: proportion of class 1 (bullish) in test fold
            baseline_pct = y_test.mean()
            baseline_pcts.append(baseline_pct)

            preds_block = pd.DataFrame({
                date_col: df_t.loc[X_test.index, date_col].values,
                ticker_col: ticker,
                'fold': fold_idx,
                'y_true': y_test.values,
                'y_pred': y_hat,
                'y_proba': proba,
                'baseline': baseline_pct,
            })
            preds_t.append(preds_block)

        if not preds_t:
            continue

        df_preds_t = pd.concat(preds_t, axis=0).reset_index(drop=True)
        
        # Per-ticker metrics
        y_true_t = df_preds_t['y_true'].values
        y_pred_t = df_preds_t['y_pred'].values
        proba_t = df_preds_t['y_proba'].values
        
        try:
            acc = accuracy_score(y_true_t, y_pred_t)
            try:
                roc = roc_auc_score(y_true_t, proba_t)
            except Exception:
                roc = np.nan
            
            # Win Rate: Precision of Class 1 (when we predict 1, how often correct?)
            win_rate = precision_score(y_true_t, y_pred_t, zero_division=0)
            
            # Buy & Hold baseline: % of positive days
            baseline = df_preds_t['baseline'].mean()
            
            # Accuracy vs Baseline
            acc_vs_baseline = acc - baseline
        except Exception as e:
            print(f'  Error computing metrics for {ticker}: {e}')
            continue

        metrics_rows.append({
            'Ticker': ticker,
            'n_test': len(df_preds_t),
            'accuracy': acc,
            'roc_auc': roc,
            'win_rate': win_rate,
            'buy_hold_baseline': baseline,
            'accuracy_vs_baseline': acc_vs_baseline,
        })

        all_preds.append(df_preds_t)

    if not all_preds:
        print('No predictions were produced.')
        return

    df_all_preds = pd.concat(all_preds, axis=0).reset_index(drop=True)

    # Global metrics
    y_true_global = df_all_preds['y_true'].values
    y_pred_global = df_all_preds['y_pred'].values
    proba_global = df_all_preds['y_proba'].values
    
    global_acc = accuracy_score(y_true_global, y_pred_global)
    try:
        global_roc = roc_auc_score(y_true_global, proba_global)
    except Exception:
        global_roc = np.nan
    
    global_win_rate = precision_score(y_true_global, y_pred_global, zero_division=0)
    global_baseline = df_all_preds['baseline'].mean()
    global_acc_vs_baseline = global_acc - global_baseline

    # Print results table per ticker
    metrics_df = pd.DataFrame(metrics_rows).sort_values('Ticker').reset_index(drop=True)
    print('\n' + '='*120)
    print('Results Table (per ticker):')
    print('='*120)
    print(metrics_df.to_string(index=False))

    print('\n' + '='*120)
    print('Global Performance:')
    print('='*120)
    print(f'  accuracy: {global_acc:.6f}')
    print(f'  roc_auc: {global_roc:.6f}')
    print(f'  win_rate (precision class 1): {global_win_rate:.6f}')
    print(f'  buy_hold_baseline (% positive days): {global_baseline:.6f}')
    print(f'  accuracy_vs_baseline: {global_acc_vs_baseline:.6f}')
    print(f'  total_test_samples: {len(df_all_preds)}')

    # Save predictions
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    df_all_preds.to_csv(OUT_PRED, index=False)
    print(f'\nSaved predictions -> {OUT_PRED}')


if __name__ == '__main__':
    main()
