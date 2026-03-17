"""Retrain CatBoost across walk-forward folds and produce IEEE-style learning curves.

Strict behavior:
- Trains CatBoost per walk-forward fold using eval_set and records per-iteration TRAIN and VALIDATION accuracy.
- Aggregates mean TRAIN and mean VALIDATION per iteration across folds (handles folds that stopped early).
- Shaded ±1 std for validation only.
- Performs validation checks and raises RuntimeError if any requirement is violated.

Output:
- `results/learning_curves_catboost_accuracy.png` (DPI=300)
- Per-fold histories saved to `catboost_info/fold_histories/` as JSON files.
"""

import math
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import time

try:
    from catboost import CatBoostClassifier
except Exception:
    raise RuntimeError('CatBoost is required to run this script. Install it in your Python environment.')

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'data' / 'combined' / 'all_features.parquet'
OUT_PLOT = ROOT / 'results' / 'learning_curves_catboost_accuracy.png'
HIST_DIR = ROOT / 'catboost_info' / 'fold_histories'

# Model hyperparameters (match original experiments)
CB_PARAMS = dict(
    iterations=1000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_seed=42,
    verbose=0,
    thread_count=4,
    use_best_model=False,
    od_type='Iter',
    od_wait=50,
    allow_writing_files=False,
)


def load_data(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Data file not found: {path}')
    df = pd.read_parquet(path)
    if 'date' not in df.columns and 'Date' in df.columns:
        df = df.rename(columns={'Date': 'date'})
    if 'ticker' not in df.columns and 'Ticker' in df.columns:
        df = df.rename(columns={'Ticker': 'ticker'})
    return df


def walk_forward_folds(ticker_df, min_train_size=150, step_size=20):
    ticker_df = ticker_df.sort_values('date').reset_index(drop=True)
    n = len(ticker_df)
    folds = []
    train_end = min_train_size
    fold_num = 0
    while train_end < n:
        train_df = ticker_df.iloc[:train_end].copy()
        test_df = ticker_df.iloc[train_end:n].copy()
        if len(test_df) > 0:
            folds.append((fold_num, train_df, test_df))
            fold_num += 1
        train_end += step_size
    return folds


def split_train_val(train_df, val_ratio=0.15):
    n = len(train_df)
    if n < 10:
        return train_df, pd.DataFrame(columns=train_df.columns)
    val_size = max(1, int(math.ceil(n * val_ratio)))
    if val_size >= n:
        val_size = 1
    train_part = train_df.iloc[: n - val_size].copy()
    val_part = train_df.iloc[n - val_size :].copy()
    return train_part, val_part


def train_fold(X_tr, y_tr, X_val, y_val, params):
    model = CatBoostClassifier(**params)
    if X_val is not None and len(X_val) > 0:
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    else:
        model.fit(X_tr, y_tr)
    evals = model.get_evals_result()

    train_acc = None
    val_acc = None
    if 'learn' in evals and isinstance(evals['learn'], dict) and len(evals['learn']) > 0:
        metric_name = list(evals['learn'].keys())[0]
        train_acc = np.array(evals['learn'][metric_name], dtype=float)
    if 'validation' in evals and isinstance(evals['validation'], dict) and len(evals['validation']) > 0:
        metric_name = list(evals['validation'].keys())[0]
        val_acc = np.array(evals['validation'][metric_name], dtype=float)

    return model, train_acc, val_acc


def main():
    warnings.filterwarnings('ignore')
    print('Loading data...')
    df = load_data(DATA_PATH)
    if 'binary_label' not in df.columns:
        raise RuntimeError('Expected column `binary_label` in data for target labels')
    tickers = sorted(df['ticker'].unique())
    print(f'Found tickers: {tickers}')

    feature_cols = [c for c in df.columns if c not in ('date', 'ticker', 'binary_label', 'next_day_return')]

    per_fold_train = []  # list of (fold_id, train_acc_array)
    per_fold_val = []
    per_fold_test = []  # list of dicts: fold, test_acc, final_iter

    HIST_DIR.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    for ticker in tickers:
        print(f'Processing ticker: {ticker}')
        ticker_df = df[df['ticker'] == ticker].sort_values('date').reset_index(drop=True)
        folds = walk_forward_folds(ticker_df, min_train_size=150, step_size=20)
        for fold_num, train_df, test_df in folds:
            global_fold_id = f"{ticker}_fold{fold_num}"
            print(f'  Fold {fold_num}: train={len(train_df)}, test={len(test_df)}')

            train_part, val_part = split_train_val(train_df, val_ratio=0.15)

            X_tr = train_part[feature_cols].fillna(0)
            y_tr = train_part['binary_label'].values
            X_val = val_part[feature_cols].fillna(0) if len(val_part) > 0 else pd.DataFrame()
            y_val = val_part['binary_label'].values if len(val_part) > 0 else np.array([])
            X_test = test_df[feature_cols].fillna(0)
            y_test = test_df['binary_label'].values

            model, train_acc, val_acc = train_fold(X_tr, y_tr, X_val, y_val, CB_PARAMS)

            # compute test accuracy
            y_pred = model.predict(X_test)
            test_acc = float((y_pred == y_test).mean()) if len(y_test) > 0 else float('nan')

            # determine final iteration reached for this fold
            lens = [len(arr) for arr in ( [train_acc] if train_acc is not None else []) + ([val_acc] if val_acc is not None else [])]
            final_iter = max(lens) - 1 if len(lens) > 0 else CB_PARAMS['iterations'] - 1

            # Save histories
            hist_path = HIST_DIR / f"{global_fold_id}.json"
            with open(hist_path, 'w', encoding='utf-8') as h:
                json.dump({
                    'train_acc': None if train_acc is None else train_acc.tolist(),
                    'val_acc': None if val_acc is None else val_acc.tolist(),
                    'test_acc': test_acc,
                    'final_iter': int(final_iter)
                }, h)

            per_fold_train.append((global_fold_id, train_acc))
            per_fold_val.append((global_fold_id, val_acc))
            per_fold_test.append({'fold': global_fold_id, 'test_acc': test_acc, 'final_iter': final_iter})

    elapsed = time.time() - start_time
    print(f'Done training all folds. Elapsed: {elapsed/60:.1f} minutes')

    # Determine maximum iterations across folds (for x-axis range)
    max_iter = 0
    for _, arr in per_fold_train:
        if arr is not None:
            max_iter = max(max_iter, len(arr))
    for _, arr in per_fold_val:
        if arr is not None:
            max_iter = max(max_iter, len(arr))
    if max_iter == 0:
        raise RuntimeError('No per-iteration metrics were recorded from any fold.')

    iters = np.arange(max_iter)

    # Build matrices with NaN for missing iterations
    n_folds = len(per_fold_train)
    train_matrix = np.full((n_folds, max_iter), np.nan)
    val_matrix = np.full((n_folds, max_iter), np.nan)

    for i, (_, arr) in enumerate(per_fold_train):
        if arr is not None:
            length = len(arr)
            train_matrix[i, :length] = arr
    for i, (_, arr) in enumerate(per_fold_val):
        if arr is not None:
            length = len(arr)
            val_matrix[i, :length] = arr

    # Mean across folds per iteration (ignore NaNs)
    train_mean = np.nanmean(train_matrix, axis=0)
    val_mean = np.nanmean(val_matrix, axis=0)
    val_std = np.nanstd(val_matrix, axis=0)

    # Final iteration index (use last index where at least one fold had a value)
    valid_iters = ~np.isnan(val_mean)
    if not valid_iters.any():
        raise RuntimeError('No validation accuracy recorded for any iteration across folds.')
    last_valid_idx = np.where(valid_iters)[0][-1]

    # Validation checks (raise RuntimeError if violated)
    mean_train_final = float(train_mean[last_valid_idx])
    mean_val_final = float(val_mean[last_valid_idx])
    gap = mean_train_final - mean_val_final

    # No test accuracy < 0.45
    test_accs = [p['test_acc'] for p in per_fold_test]
    if any([acc < 0.45 for acc in test_accs if not math.isnan(acc)]):
        raise RuntimeError('Validation failed: At least one fold has test accuracy < 0.45')
    if mean_train_final < 0.90:
        raise RuntimeError(f'Validation failed: Mean TRAIN accuracy at final iteration < 0.90 ({mean_train_final:.4f})')
    if not (0.52 <= mean_val_final <= 0.65):
        raise RuntimeError(f'Validation failed: Mean VALIDATION accuracy at final iteration not in [0.52,0.65] ({mean_val_final:.4f})')
    if gap > 0.12:
        raise RuntimeError(f'Validation failed: Train-validation gap at final iteration > 12% ({gap:.4f})')

    # Plotting per strict requirements
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(iters, train_mean, color='blue', linewidth=2, label='Train')
    ax.plot(iters, val_mean, color='orange', linewidth=2, label='Validation')
    ax.fill_between(iters, val_mean - val_std, val_mean + val_std, color='orange', alpha=0.2)

    ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Random Baseline')

    # Green dots: one per fold at that fold's final iteration (use final_iter and corresponding test_acc)
    xs = []
    ys = []
    for rec in per_fold_test:
        if not math.isnan(rec['test_acc']):
            x = int(rec['final_iter'])
            # clamp to available x range
            x = min(x, max_iter - 1)
            xs.append(x)
            ys.append(rec['test_acc'])
    ax.scatter(xs, ys, color='green', s=40, label='Test')

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Accuracy')
    ax.set_title('Learning Curves: CatBoost Accuracy (Walk-Forward Validation)')
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PLOT, dpi=300, bbox_inches='tight')
    print(f'Saved plot to {OUT_PLOT}')


if __name__ == '__main__':
    main()
