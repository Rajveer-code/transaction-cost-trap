"""
train_ensemble.py

Ensemble evaluation combining CatBoost and RandomForest on the ensemble
feature set with per-ticker walk-forward validation.

Outputs:
 - results/predictions_ensemble.csv

Usage:
    python scripts/train_ensemble.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False
    warnings.warn('CatBoost not available; Model A will use RandomForest fallback')
    from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier as RF

ROOT = Path(__file__).resolve().parents[1]
IN_FILE = ROOT / 'data' / 'extended' / 'features_ensemble.parquet'
OUT_PRED = ROOT / 'results' / 'predictions_ensemble.csv'


def make_models(random_state=42):
    # Model A
    if HAS_CATBOOST:
        model_a = CatBoostClassifier(
            iterations=1000,
            depth=5,
            learning_rate=0.02,
            random_state=random_state,
            verbose=0
        )
    else:
        model_a = RF(n_estimators=500, random_state=random_state, n_jobs=-1)

    # Model B: RandomForest
    model_b = RF(n_estimators=500, min_samples_leaf=40, random_state=random_state, n_jobs=-1)
    return model_a, model_b


def main():
    if not IN_FILE.exists():
        print('Ensemble feature file not found:', IN_FILE)
        return

    print('Loading ensemble features:', IN_FILE)
    df = pd.read_parquet(IN_FILE)

    cols = [c for c in df.columns]
    date_col = next((c for c in cols if 'date' == c.lower()), None)
    if date_col is None:
        date_col = next((c for c in cols if 'date' in c.lower()), None)
    ticker_col = next((c for c in cols if 'ticker' == c.lower()), None)
    if ticker_col is None:
        ticker_col = next((c for c in cols if 'ticker' in c.lower()), None)

    if date_col is None or ticker_col is None:
        raise RuntimeError('Date or Ticker column not found in ensemble features')

    target_col = 'binary_label'
    if target_col not in df.columns:
        raise RuntimeError('binary_label target not found in ensemble features')

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([ticker_col, date_col]).reset_index(drop=True)

    # Feature columns: exclude ids and targets
    id_cols = {date_col, ticker_col, 'target_return_5d', target_col}
    feature_cols = [c for c in df.columns if c not in id_cols]
    feature_cols = [c for c in feature_cols if not df[c].isna().all()]

    print(f'Using {len(feature_cols)} features')

    all_preds = []
    rows = []

    model_a, model_b = make_models(random_state=42)

    tickers = sorted(df[ticker_col].unique())
    for ticker in tickers:
        df_t = df[df[ticker_col] == ticker].copy().reset_index(drop=True)
        X = df_t[feature_cols]
        y = df_t[target_col]

        mask_valid = X.notna().all(axis=1) & y.notna()
        X = X[mask_valid]
        y = y[mask_valid]

        if len(y) < 20:
            print(f'  Skipping {ticker} - insufficient data ({len(y)} rows)')
            continue

        print(f'Processing {ticker} ({len(y)} samples)')
        tscv = TimeSeriesSplit(n_splits=10)

        preds_blocks = []
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Fit both models
            model_a.fit(X_train, y_train)
            model_b.fit(X_train, y_train)

            # Probabilities
            if hasattr(model_a, 'predict_proba'):
                pa = model_a.predict_proba(X_test)[:, 1]
            else:
                pa = model_a.predict(X_test).astype(float)

            if hasattr(model_b, 'predict_proba'):
                pb = model_b.predict_proba(X_test)[:, 1]
            else:
                pb = model_b.predict(X_test).astype(float)

            p_ens = (pa + pb) / 2.0
            y_pred = (p_ens > 0.5).astype(int)

            block = pd.DataFrame({
                date_col: df_t.loc[X_test.index, date_col].values,
                ticker_col: ticker,
                'fold': fold_idx,
                'y_true': y_test.values,
                'prob_a': pa,
                'prob_b': pb,
                'prob_ens': p_ens,
                'y_pred': y_pred,
            })

            # include ratio_sma200 if present
            if 'ratio_sma200' in df_t.columns:
                block['ratio_sma200'] = df_t.loc[X_test.index, 'ratio_sma200'].values
            else:
                block['ratio_sma200'] = np.nan

            preds_blocks.append(block)

        if not preds_blocks:
            continue

        df_preds_t = pd.concat(preds_blocks, axis=0).reset_index(drop=True)
        all_preds.append(df_preds_t)

        # Per-ticker metrics
        y_true = df_preds_t['y_true'].values
        y_pred = df_preds_t['y_pred'].values
        p_ens = df_preds_t['prob_ens'].values

        ensemble_win = precision_score(y_true, y_pred, zero_division=0)

        # Regime filter: ratio_sma200 > 1.0
        regime_mask = False
        if 'ratio_sma200' in df_preds_t.columns:
            regime_mask = df_preds_t['ratio_sma200'].fillna(-np.inf) > 1.0

        if regime_mask is not False and regime_mask.sum() > 0:
            regime_win = precision_score(y_true[regime_mask.values], y_pred[regime_mask.values], zero_division=0)
        else:
            regime_win = np.nan

        improvement = regime_win - ensemble_win if not np.isnan(regime_win) else np.nan

        rows.append({
            'Ticker': ticker,
            'n_test': len(df_preds_t),
            'ensemble_win_rate': ensemble_win,
            'regime_win_rate': regime_win,
            'improvement': improvement,
        })

    if not all_preds:
        print('No predictions produced.')
        return

    df_all = pd.concat(all_preds, axis=0).reset_index(drop=True)

    # Global metrics
    y_true_g = df_all['y_true'].values
    y_pred_g = df_all['y_pred'].values
    prob_g = df_all['prob_ens'].values

    global_acc = accuracy_score(y_true_g, y_pred_g)
    try:
        global_auc = roc_auc_score(y_true_g, prob_g)
    except Exception:
        global_auc = np.nan

    global_ensemble_win = precision_score(y_true_g, y_pred_g, zero_division=0)
    if 'ratio_sma200' in df_all.columns:
        mask_reg = df_all['ratio_sma200'].fillna(-np.inf) > 1.0
        if mask_reg.sum() > 0:
            global_regime_win = precision_score(y_true_g[mask_reg.values], y_pred_g[mask_reg.values], zero_division=0)
        else:
            global_regime_win = np.nan
    else:
        global_regime_win = np.nan

    # Per-ticker table
    metrics_df = pd.DataFrame(rows).sort_values('Ticker').reset_index(drop=True)
    print('\n' + '='*120)
    print('Per-ticker results:')
    print('='*120)
    print(metrics_df.to_string(index=False))

    print('\n' + '='*120)
    print('Global Performance:')
    print('='*120)
    print(f'  accuracy: {global_acc:.6f}')
    print(f'  roc_auc: {global_auc:.6f}')
    print(f'  ensemble_win_rate: {global_ensemble_win:.6f}')
    print(f'  regime_win_rate: {global_regime_win if not np.isnan(global_regime_win) else "N/A"}')
    print(f'  total_test_samples: {len(df_all)}')

    OUT_PRED.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUT_PRED, index=False)
    print(f'\nSaved predictions -> {OUT_PRED}')


if __name__ == '__main__':
    main()
