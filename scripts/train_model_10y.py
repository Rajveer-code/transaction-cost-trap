"""
train_model_10y.py

Evaluate the engineered 10-year dataset using strict per-ticker walk-forward
validation. Trains CatBoostClassifier on expanding time-series splits per ticker
and aggregates predictions for global evaluation.

Outputs:
 - results/metrics_10y_final.csv
 - results/predictions_10y_final.csv

Usage:
    python scripts/train_model_10y.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
IN_FILE = ROOT / 'data' / 'extended' / 'features_final_10y.parquet'
OUT_METRICS = ROOT / 'results' / 'metrics_10y_final.csv'
OUT_PRED = ROOT / 'results' / 'predictions_10y_final.csv'


def choose_model(random_state=42):
    if HAS_CATBOOST:
        return CatBoostClassifier(random_state=random_state, verbose=0)
    warnings.warn('CatBoost not available; falling back to RandomForestClassifier')
    return RandomForestClassifier(n_estimators=200, random_state=random_state)


def evaluate_global(df_preds):
    y_true = df_preds['y_true'].values
    y_pred = df_preds['y_pred'].values
    proba = df_preds['proba'].values

    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    try:
        metrics['roc_auc'] = roc_auc_score(y_true, proba)
    except Exception:
        metrics['roc_auc'] = np.nan

    # binomial test vs p=0.5
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

    # Robust read: try default, then try fastparquet if available
    try:
        df = pd.read_parquet(IN_FILE)
    except Exception as e:
        print('Warning: default parquet read failed:', e)
        try:
            df = pd.read_parquet(IN_FILE, engine='fastparquet')
            print('Read parquet with fastparquet engine')
        except Exception as e2:
            print('Failed to read parquet with fastparquet as well:', e2)
            raise
    # expect columns: Date, Ticker, features..., next_day_return, binary_label
    col_names = [c for c in df.columns]
    date_col = next((c for c in col_names if 'date' == c.lower()), None)
    if date_col is None:
        date_col = next((c for c in col_names if 'date' in c.lower()), None)
    ticker_col = next((c for c in col_names if 'ticker' == c.lower()), None)
    if ticker_col is None:
        ticker_col = next((c for c in col_names if 'ticker' in c.lower()), None)

    if date_col is None or ticker_col is None:
        raise RuntimeError('Date or Ticker column not found in features file')

    # ensure sorted
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)

    # Identify feature columns
    target_col = 'binary_label'
    if target_col not in df.columns:
        raise RuntimeError('binary_label target not found in features file')

    id_cols = {date_col, ticker_col, 'next_day_return', target_col}
    feature_cols = [c for c in df.columns if c not in id_cols]
    # Drop any feature columns that are entirely NaN
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]

    # Containers for aggregated predictions and per-ticker metrics
    all_preds = []
    metrics_rows = []

    tickers = df[ticker_col].unique()
    for ticker in tickers:
        df_t = df[df[ticker_col] == ticker].copy().reset_index(drop=True)
        X = df_t[feature_cols]
        y = df_t[target_col]

        # drop rows with NaNs in features or target
        mask_valid = X.notna().all(axis=1) & y.notna()
        X = X[mask_valid]
        y = y[mask_valid]
        if len(y) < 20:
            print(f'  Skipping {ticker} - insufficient data after cleaning ({len(y)} rows)')
            continue

        tscv = TimeSeriesSplit(n_splits=10)
        model = choose_model(random_state=42)

        preds_t = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit model
            model.fit(X_train, y_train)

            # Predictions
            y_hat = model.predict(X_test)
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test)[:, 1]
            else:
                # fallback: use predict as 0/1 probability
                proba = y_hat.astype(float)

            preds_block = pd.DataFrame({
                date_col: df_t.loc[X_test.index, date_col].values,
                ticker_col: ticker,
                'y_true': y_test.values,
                'y_pred': y_hat,
                'proba': proba,
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
            f1 = f1_score(df_preds_t['y_true'], df_preds_t['y_pred'], zero_division=0)
            try:
                roc = roc_auc_score(df_preds_t['y_true'], df_preds_t['proba'])
            except Exception:
                roc = np.nan
        except Exception as e:
            print('Error computing metrics for', ticker, e)
            continue

        metrics_rows.append({
            'Ticker': ticker,
            'n_test': len(df_preds_t),
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1,
            'roc_auc': roc,
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
    print('\nResults Table (per ticker):')
    print(metrics_df.to_string(index=False))

    print('\nGlobal Performance:')
    for k, v in global_metrics.items():
        print(f'  {k}: {v}')

    # Save outputs
    OUT_METRICS.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(OUT_METRICS, index=False)
    df_all_preds.to_csv(OUT_PRED, index=False)
    print('\nSaved metrics ->', OUT_METRICS)
    print('Saved predictions ->', OUT_PRED)


if __name__ == '__main__':
    main()
