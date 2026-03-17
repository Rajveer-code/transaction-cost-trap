"""
train_model_5d.py

Evaluate the 5-day horizon engineered dataset using strict per-ticker walk-forward
validation with CatBoostClassifier. Includes baseline accuracy and binomial test.

Outputs:
 - results/predictions_5d_final.csv

Usage:
    python scripts/train_model_5d.py
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

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
IN_FILE = ROOT / 'data' / 'extended' / 'features_5d_horizon.parquet'
OUT_PRED = ROOT / 'results' / 'predictions_5d_final.csv'


def choose_model(random_state=42):
    """Create CatBoost or RandomForest model with appropriate config."""
    if HAS_CATBOOST:
        return CatBoostClassifier(
            iterations=1000,
            learning_rate=0.03,
            depth=6,
            random_state=random_state,
            verbose=0,
            auto_class_weights='Balanced'
        )
    warnings.warn('CatBoost not available; falling back to RandomForestClassifier')
    return RandomForestClassifier(n_estimators=200, random_state=random_state, class_weight='balanced')


def evaluate_global(df_preds):
    """Compute global metrics and binomial test."""
    y_true = df_preds['y_true'].values
    y_pred = df_preds['y_pred'].values
    proba = df_preds['y_proba'].values

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, proba)
    except Exception:
        metrics['roc_auc'] = np.nan

    # Baseline: proportion of class 1 in test set
    metrics['baseline_accuracy'] = y_true.mean()
    
    # Excess return
    metrics['excess_return'] = metrics['accuracy'] - metrics['baseline_accuracy']

    # Binomial test: are correct predictions significantly better than 50% random?
    successes = int((y_true == y_pred).sum())
    n = len(y_true)
    bt = binomtest(successes, n, p=0.5)
    metrics['binom_pvalue'] = bt.pvalue
    metrics['n'] = n
    metrics['correct'] = successes

    return metrics


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

    # Identify feature columns (exclude ID, target, and derived columns)
    id_cols = {date_col, ticker_col, 'target_return_5d', target_col}
    feature_cols = [c for c in df.columns if c not in id_cols]
    # Drop any feature columns that are entirely NaN
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]

    # Containers for results
    all_preds = []
    metrics_rows = []

    tickers = sorted(df[ticker_col].unique())
    for ticker in tickers:
        df_t = df[df[ticker_col] == ticker].copy().reset_index(drop=True)
        X = df_t[feature_cols]
        y = df_t[target_col]

        # Drop rows with NaNs in features or target
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
        baseline_accs = []
        
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

            # Baseline: proportion of class 1 in this test fold
            baseline_acc = y_test.mean()
            baseline_accs.append(baseline_acc)

            preds_block = pd.DataFrame({
                date_col: df_t.loc[X_test.index, date_col].values,
                ticker_col: ticker,
                'fold': fold_idx,
                'y_true': y_test.values,
                'y_pred': y_hat,
                'y_proba': proba,
                'baseline': baseline_acc,
            })
            preds_t.append(preds_block)

        if not preds_t:
            continue

        df_preds_t = pd.concat(preds_t, axis=0).reset_index(drop=True)
        
        # Per-ticker metrics
        try:
            acc = accuracy_score(df_preds_t['y_true'], df_preds_t['y_pred'])
            prec = precision_score(df_preds_t['y_true'], df_preds_t['y_pred'], zero_division=0)
            rec = recall_score(df_preds_t['y_true'], df_preds_t['y_pred'], zero_division=0)
            try:
                roc = roc_auc_score(df_preds_t['y_true'], df_preds_t['y_proba'])
            except Exception:
                roc = np.nan
            
            baseline = df_preds_t['baseline'].mean()
            excess = acc - baseline
        except Exception as e:
            print(f'Error computing metrics for {ticker}: {e}')
            continue

        metrics_rows.append({
            'Ticker': ticker,
            'n_test': len(df_preds_t),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'roc_auc': roc,
            'baseline_accuracy': baseline,
            'excess_return': excess,
        })

        all_preds.append(df_preds_t)

    if not all_preds:
        print('No predictions were produced.')
        return

    df_all_preds = pd.concat(all_preds, axis=0).reset_index(drop=True)

    # Global evaluation
    global_metrics = evaluate_global(df_all_preds)

    # Print results table per ticker
    metrics_df = pd.DataFrame(metrics_rows).sort_values('Ticker').reset_index(drop=True)
    print('\n' + '='*100)
    print('Results Table (per ticker):')
    print('='*100)
    print(metrics_df.to_string(index=False))

    print('\n' + '='*100)
    print('Global Performance:')
    print('='*100)
    for k, v in global_metrics.items():
        if isinstance(v, float):
            print(f'  {k}: {v:.6f}')
        else:
            print(f'  {k}: {v}')

    # Save predictions
    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    df_all_preds.to_csv(OUT_PRED, index=False)
    print(f'\nSaved predictions -> {OUT_PRED}')


if __name__ == '__main__':
    main()
